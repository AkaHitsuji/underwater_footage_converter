import cv2
import numpy as np
import os
import argparse
import shutil
import tempfile
from pathlib import Path
import subprocess
import piexif
from PIL import Image
from tqdm import tqdm
import logging
import time
import multiprocessing
import signal
import sys
import re
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import KMeans

# Constants for file types
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
QUICKTIME_EXTENSIONS = ('.mov', '.mp4')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('UnderwaterConverter')

# Global flag for graceful termination
terminate_requested = False

# Signal handler for graceful termination
def signal_handler(sig, frame):
    global terminate_requested
    if terminate_requested:
        logger.info("Forced termination requested. Exiting immediately.")
        sys.exit(1)
    else:
        logger.info("Termination requested. Finishing current frame and cleaning up...")
        logger.info("Press Ctrl+C again to force immediate exit.")
        terminate_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Utility functions
def is_image_file(file_path):
    """Check if a file is a supported image format"""
    return file_path.lower().endswith(IMAGE_EXTENSIONS)

def is_video_file(file_path):
    """Check if a file is a supported video format"""
    return file_path.lower().endswith(VIDEO_EXTENSIONS)

def is_quicktime_format(file_path):
    """Check if a file is in a QuickTime format (MOV/MP4)"""
    return file_path.lower().endswith(QUICKTIME_EXTENSIONS)

def check_exiftool_available():
    """Check if exiftool is available on the system"""
    try:
        subprocess.run(['exiftool', '-ver'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("exiftool not found, metadata preservation capabilities will be limited")
        return False

class AutoColorUnderwaterImageProcessor:
    """Enhanced processor for underwater images with automatic color parameter selection"""
    
    def __init__(self, contrast_limit=2.0, saturation_factor=0.9, 
                 brightness_boost=1.12, white_balance_strength=0.7,
                 auto_tune_strength=0.9, sharpness=0.0):
        """
        Initialize the underwater image processor with automatic color parameter selection
        
        Args:
            contrast_limit: Clip limit for CLAHE contrast enhancement
            saturation_factor: Saturation adjustment factor
            brightness_boost: Brightness multiplier
            white_balance_strength: Strength of white balancing
            auto_tune_strength: Strength of the automatic parameter tuning (0-1)
                               Higher values apply more aggressive automatic adjustments
            sharpness: Sharpness enhancement factor (0.0-1.0)
        """
        # Core parameters (we'll auto-tune RGB values)
        self.contrast_limit = contrast_limit
        self.saturation_factor = saturation_factor
        self.brightness_boost = brightness_boost
        self.white_balance_strength = white_balance_strength
        self.auto_tune_strength = auto_tune_strength
        self.sharpness = sharpness
        
        # Default RGB values (will be overridden by auto-tuning)
        # More balanced boost with reduced red
        self.red_boost = 0.95  # Changed from 1.0 to slightly reduce red by default
        self.blue_boost = 0.98  # Further reduced from 1.05 to eliminate blue sheen
        self.green_factor = 1.0  # Reduced from 1.05 to fix green tinge
        
        # NEW: Flag to indicate if the scene has red-tinted rocks
        self.has_red_tinted_rocks = False
        
        # NEW: Flag to indicate if scene has deep blue water
        self.is_deep_blue = False
        
        # Create CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=self.contrast_limit, tileGridSize=(8, 8))
        
        # Track if we've optimized parameters for the current source
        self.has_optimized_params = False
        self.sample_size = (640, 480)  # Size to resize images for analysis
        
        logger.info(f"Initialized auto-color processor (v5) with auto-tune strength: {auto_tune_strength}")
    
    def _create_ffmpeg_extraction_command(self, input_path, frames_path, num_workers):
        """Create FFmpeg command for extracting frames from video"""
        return [
            'ffmpeg', '-i', input_path, 
            '-qscale:v', '1',  # Highest quality (1-31, lower is better)
            '-threads', str(num_workers),
            frames_path
        ]

    def _create_ffmpeg_encoding_command(self, frames_path, output_path, fps, width, height, num_workers, crf=24):
        """Create FFmpeg command for encoding frames into video"""
        return [
            'ffmpeg', '-r', str(fps),
            '-i', frames_path,
            '-c:v', 'libx265',  # Use HEVC codec 
            '-preset', 'slow',  # Use slow preset for better quality/compression ratio
            '-crf', str(crf),  # Lower CRF value for better quality
            '-pix_fmt', 'yuv420p',
            '-threads', str(num_workers),
            '-vf', f'scale=-2:{height},setdar=1/1,setsar=1/1',
            '-tag:v', 'hvc1',  # Ensure compatibility with Apple devices
            '-movflags', '+faststart',
            output_path
        ]

    def _create_ffmpeg_metadata_command(self, input_path, processed_path, output_path, is_quicktime=False):
        """Create FFmpeg command for copying metadata from original to processed video"""
        cmd = [
            'ffmpeg',
            '-i', input_path,          # Original file with metadata (first input)
            '-i', processed_path,      # Processed video (second input)
            '-map', '1:v',             # Use video from the processed file
            '-map_metadata', '0',      # Use metadata from the original file
            '-map', '0:a?',            # Copy audio if present
            '-map', '0:s?',            # Copy subtitles if present
            '-c', 'copy',              # Copy all streams (no re-encoding)
            '-movflags', 'use_metadata_tags+faststart'
        ]
        
        # Add QuickTime-specific mapping options if needed
        if is_quicktime:
            cmd.extend([
                '-map', '0:d?',        # Copy data if present
                '-map', '0:t?'         # Copy timecodes if present
            ])
        
        cmd.append(output_path)
        return cmd

    def _run_ffmpeg_with_progress(self, command, total_frames, description="Processing"):
        """Run FFmpeg command with progress monitoring"""
        # Create a temporary file for FFmpeg progress
        progress_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        progress_file.close()
        
        # Add progress monitoring to the command
        full_command = command.copy()  # Create a copy to avoid modifying the original
        
        # Add stats and progress options
        if '-y' not in full_command:
            full_command = [full_command[0], '-y'] + full_command[1:]  # Add overwrite flag
        full_command.extend(['-stats', '-progress', progress_file.name])
        
        # Start FFmpeg process
        process = subprocess.Popen(
            full_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Setup progress bar
        pbar = tqdm(total=total_frames, desc=description)
        
        # Monitor progress
        last_frame = 0
        stalled_count = 0  # Counter to detect if progress is stuck
        
        try:
            while process.poll() is None:
                try:
                    time.sleep(0.5)  # Check progress every half second
                    
                    if os.path.exists(progress_file.name) and os.path.getsize(progress_file.name) > 0:
                        with open(progress_file.name, 'r') as f:
                            progress_text = f.read()
                        
                        frame_match = re.search(r'frame=\s*(\d+)', progress_text)
                        if frame_match:
                            current_frame = int(frame_match.group(1))
                            if current_frame > last_frame:
                                pbar.update(current_frame - last_frame)
                                last_frame = current_frame
                                stalled_count = 0  # Reset stall counter
                            else:
                                stalled_count += 1
                except (IOError, FileNotFoundError):
                    pass  # File not ready yet
                
                # If progress appears stuck but process is still running,
                # just update the progress bar a little to show activity
                if stalled_count > 10 and last_frame < total_frames:  
                    pbar.set_description(f"{description} (estimating...)")
                    # Avoid updating beyond total
                    if last_frame < total_frames - 1:
                        pbar.update(1)
                        last_frame += 1
                    stalled_count = 0
            
            # Get the process result
            stdout, stderr = process.communicate()
            stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
            
            # Ensure progress bar reaches 100% when process completes successfully
            if process.returncode == 0 and last_frame < total_frames:
                pbar.update(total_frames - last_frame)
            
            # Check if process completed successfully
            if process.returncode != 0:
                logger.error(f"FFmpeg process failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr_text}")
                raise subprocess.CalledProcessError(process.returncode, full_command, 
                                                output=stdout, stderr=stderr)
                
            return process.returncode
            
        finally:
            # Always clean up, even if there's an exception
            pbar.close()
            # Kill the process if it's still running
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            # Remove temporary file
            try:
                if os.path.exists(progress_file.name):
                    os.unlink(progress_file.name)
            except:
                pass 

    def analyze_underwater_image(self, image):
        """
        Analyze underwater image to determine optimal color correction parameters
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple containing optimal red_boost, blue_boost, and green_factor values
        """
        # Step 1: Resize for faster analysis
        h, w = image.shape[:2]
        analysis_img = cv2.resize(image, self.sample_size) if max(h, w) > max(self.sample_size) else image.copy()
        
        # Step 2: Convert to LAB color space and extract channels
        lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Analyze a channel (red-green axis) to detect red tinting
        # Higher values in 'a' channel indicate more red
        mean_a = np.mean(a) - 128  # Centered around 0, positive = red, negative = green
        
        # Step 3: Analyze water color cast using K-means clustering
        # Reshape for clustering
        pixels = analysis_img.reshape(-1, 3)
        
        # Choose K based on the complexity of the underwater scene
        k = 8  # Increased from 5 to better detect different elements in the scene
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by cluster size (largest first)
        counts = np.bincount(kmeans.labels_)
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = colors[sorted_indices]
        
        # Calculate average water color using weighted average of dominant colors
        water_color_weight = 0.6  # Weight for most dominant color
        second_color_weight = 0.3  # Weight for second most dominant color
        other_colors_weight = 0.1  # Weight for other colors
        
        # Initialize weighted average with most dominant color
        water_color = dominant_colors[0] * water_color_weight
        
        # Add weighted second dominant color
        if len(dominant_colors) > 1:
            water_color += dominant_colors[1] * second_color_weight
        
        # Add weighted average of remaining colors
        if len(dominant_colors) > 2:
            other_colors = np.mean(dominant_colors[2:], axis=0)
            water_color += other_colors * other_colors_weight
        
        # Normalize to 0-1 range
        water_color = water_color / 255.0
        b_water, g_water, r_water = water_color
        
        # Analyze color balance 
        r_avg = np.mean(analysis_img[:,:,2]) / 255.0
        g_avg = np.mean(analysis_img[:,:,1]) / 255.0
        b_avg = np.mean(analysis_img[:,:,0]) / 255.0
        
        # Calculate channel ratios
        avg_color = (r_avg + g_avg + b_avg) / 3.0
        r_ratio = r_avg / avg_color if avg_color > 0 else 1.0
        g_ratio = g_avg / avg_color if avg_color > 0 else 1.0
        b_ratio = b_avg / avg_color if avg_color > 0 else 1.0

        # Red tint detection - using color ratios and LAB analysis
        is_red_tinted = (r_ratio > 1.02 and r_ratio > g_ratio and mean_a > 1)  # More sensitive detection (was 1.05 and mean_a > 2)
        
        # NEW: Improved detection of neutral colors with red tint (rocks, sand, etc.)
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # Function to detect rock-like colors
        def detect_red_tinted_rocks():
            """Enhanced detection of red-tinted rock-like elements in the scene"""
            red_rock_count = 0
            total_clusters = len(dominant_colors)
            
            for i, cluster_color in enumerate(dominant_colors):
                # Skip tiny clusters
                if counts[sorted_indices[i]] < len(pixels) * 0.03:  # Skip clusters smaller than 3%
                    continue
                    
                # Convert to BGR
                b, g, r = cluster_color
                
                # Convert to HSV for better color analysis
                hsv_color = cv2.cvtColor(np.uint8([[cluster_color]]), cv2.COLOR_BGR2HSV)[0][0]
                hue, sat, val = hsv_color
                
                # Check for colors that would be gray/neutral except for red tinting:
                # 1. Low saturation (grayish)
                is_low_saturation = sat < 100
                
                # 2. Rock-like value (not too dark, not too bright)
                is_rock_brightness = 50 < val < 210
                
                # 3. But with red hue bias or red > other channels
                is_reddish = (hue < 15 or hue > 165) or (r > g * 1.05 and r > b * 1.05)
                
                # 4. Relatively balanced RGB values (would be neutral without tint)
                rgb_range = max(r, g, b) - min(r, g, b)
                is_potentially_neutral = rgb_range < 60
                
                if (is_low_saturation or is_potentially_neutral) and is_rock_brightness and is_reddish:
                    red_rock_count += 1
                    logger.debug(f"Detected red-tinted rock-like cluster: {cluster_color}, HSV: {hsv_color}")
            
            # Consider it a red-tinted rock scene if enough clusters match
            has_red_rocks = red_rock_count >= 2
            
            # Also check if overall scene has low saturation but red tint
            overall_low_sat = np.mean(s) < 90  # Increased from 80
            overall_red_tint = r_ratio > 1.02 and mean_a > 0  # More sensitive (was 1.05 and mean_a > 0)
            
            return has_red_rocks or (overall_low_sat and overall_red_tint)
            
        # Run the enhanced rock detection
        has_red_tinted_rocks = detect_red_tinted_rocks()
        self.has_red_tinted_rocks = has_red_tinted_rocks  # Store this for later use
        
        # Final red tint detection combines multiple methods
        needs_red_correction = is_red_tinted or has_red_tinted_rocks
        
        # Determine scene type
        is_deep_blue = (b_avg > 0.45 and b_avg > r_avg * 1.3 and b_avg > g_avg * 1.15)  # Relaxed criteria to catch more blue scenes
        self.is_deep_blue = is_deep_blue  # Store for later use
        is_greenish = (g_avg > r_avg * 1.2 and g_avg > b_avg * 1.1)
        is_balanced = (max(r_ratio, g_ratio, b_ratio) / min(r_ratio, g_ratio, b_ratio) < 1.3)
        
        # Calculate optimal color boosts based on scene type
        if needs_red_correction:
            # Strong correction for red-tinted scenes, especially with neutral objects like rocks
            optimal_red_boost = 0.8 if has_red_tinted_rocks else 0.85  # Stronger red reduction (was 0.85/0.9)
            optimal_blue_boost = 0.95  # Further reduced from 1.0 to eliminate blue sheen
            optimal_green_factor = 1.05  # Reduced from 1.15 to fix green tinge
            logger.info(f"Detected red-tinted scene with rocks={has_red_tinted_rocks}, applying strong red reduction (red_ratio={r_ratio:.2f}, a_mean={mean_a:.2f})")
        elif is_deep_blue:
            # Deep blue water
            optimal_red_boost = 1.0  # No red boost
            optimal_blue_boost = 0.95 + b_avg * 0.03  # Significantly reduced from 1.0 to eliminate blue sheen
            optimal_green_factor = 1.0 + (g_avg - r_avg) * 0.05  # Reduced from 1.05 to fix green tinge
            logger.info(f"Detected deep blue scene (b_avg={b_avg:.2f}), reducing blue intensity")
        elif is_greenish:
            # Greenish water
            optimal_red_boost = 1.0  # No red boost
            optimal_blue_boost = 0.98 + (0.5 - b_avg) * 0.05  # Reduced from 1.05 to eliminate blue sheen
            optimal_green_factor = 0.9 - (g_avg - r_avg) * 0.1  # Reduced from 0.95 to address green tinge
        elif is_balanced:
            # Already balanced scene
            optimal_red_boost = 0.98  # Slight red reduction (was 1.0)
            optimal_blue_boost = 0.98  # Further reduced from 1.02 to eliminate blue sheen
            optimal_green_factor = 0.98  # Reduced from 1.02 to fix green tinge
        else:
            # General case
            optimal_red_boost = 0.95  # Slight red reduction (was 1.0)
            optimal_blue_boost = 0.98  # Reduced from 1.05 to eliminate blue sheen
            optimal_green_factor = 1.0 + (1.0 - g_ratio) * 0.05  # Reduced coefficient from 0.1 to fix green tinge
        
        # Default values with no red boost
        default_red = 0.9   # Increased red reduction (was 0.95)
        default_blue = 0.95  # Further reduced blue boost from 1.0 to eliminate blue sheen
        default_green = 0.98  # Reduced from 1.05 to fix green tinge
        
        # Apply blending
        red_boost = default_red * (1.0 - self.auto_tune_strength) + optimal_red_boost * self.auto_tune_strength
        blue_boost = default_blue * (1.0 - self.auto_tune_strength) + optimal_blue_boost * self.auto_tune_strength
        green_factor = default_green * (1.0 - self.auto_tune_strength) + optimal_green_factor * self.auto_tune_strength
        
        # Clamp to reasonable ranges - allow more red reduction
        red_boost = np.clip(red_boost, 0.75, 1.1)  # Allow even more red reduction (was 0.8)
        blue_boost = np.clip(blue_boost, 0.9, 1.15)  # Lower both min and max (was 0.95-1.25) to eliminate blue sheen
        green_factor = np.clip(green_factor, 0.85, 1.15)  # Reduced upper limit from 1.25 to fix green tinge
        
        return red_boost, blue_boost, green_factor

    def optimize_color_parameters(self, image):
        """
        Optimize color parameters based on the image analysis
        
        Args:
            image: Input BGR image to analyze
            
        Returns:
            Tuple of optimized parameters (red_boost, blue_boost, green_factor)
        """
        # Use image analysis to determine optimal parameters
        optimal_red, optimal_blue, optimal_green = self.analyze_underwater_image(image)
        
        # Update processor parameters
        self.red_boost = optimal_red
        self.blue_boost = optimal_blue
        self.green_factor = optimal_green
        
        logger.info(f"Auto-optimized color parameters: red={optimal_red:.2f}, blue={optimal_blue:.2f}, green={optimal_green:.2f}")
        
        self.has_optimized_params = True
        
        return optimal_red, optimal_blue, optimal_green

    def restore_color(self, image, optimize_params=False):
        """
        Restore natural colors in underwater image with automatic parameter optimization
        
        Args:
            image: Input BGR image (numpy array)
            optimize_params: Whether to re-optimize parameters for this image
            
        Returns:
            Corrected BGR image (numpy array)
        """
        # Optimize parameters if requested or if first time
        if optimize_params or not self.has_optimized_params:
            self.optimize_color_parameters(image)
        
        # Step 1: Enhanced white balance correction for more neutral colors
        wb_img = self._apply_white_balance(image)
        
        # Step 2: Convert to LAB for contrast enhancement on luminance channel
        lab = cv2.cvtColor(wb_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for adaptive contrast enhancement
        l_enhanced = self.clahe.apply(l)
        
        # Step 3: Color correction in LAB space
        # Get mean a value (red-green axis)
        mean_a = np.mean(a) - 128  # Center around 0
        
        # Determine a-channel shift based on red tint detection
        if self.has_red_tinted_rocks:
            # Strong green shift for red-tinted rocks
            a_shift_value = -10.0  # Stronger correction for rock scenes (was -8.0)
        elif mean_a > 3:
            # Moderate green shift for general red tint
            a_shift_value = -6.0  # Increased from -5.0
        else:
            # Slight green shift otherwise to prevent reddish tones
            a_shift_value = -3.0  # Increased from -2.0
            
        # Apply shift to a channel (green-red axis)
        a_shifted = cv2.addWeighted(a, 1.0, np.ones_like(a) * 128, 0.0, a_shift_value)
        
        # Shift b channel (blue-yellow axis) based on detection of deep blue scenes
        if self.is_deep_blue:
            # For very blue scenes, add stronger yellow shift
            b_shift_value = 3.0  # Increased positive shift towards yellow to counter blue (was 2.0)
            logger.debug(f"Deep blue scene detected, applying yellow shift: {b_shift_value}")
        else:
            # Mild shift for regular scenes
            b_shift_value = 1.0  # Slightly positive shift towards yellow (was 0.0)
            
        b_shifted = cv2.addWeighted(b, 1.0, np.ones_like(b) * 128, 0.0, b_shift_value)
        
        # Merge LAB channels and convert back to BGR
        enhanced_lab = cv2.merge([l_enhanced, a_shifted, b_shifted])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 4: Apply automatically tuned RGB channel adjustments
        enhanced_bgr = self._boost_rgb_channels(enhanced_bgr)
        
        # Step 5: Apply vibrance enhancement (smart saturation)
        enhanced_bgr = self._apply_vibrance(enhanced_bgr, factor=0.4)  # Reduced from 0.5
        
        # Step 6: Adjust saturation and brightness in HSV space
        enhanced_hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(enhanced_hsv)
        
        # NEW: Adjust hue to further address red tint if needed
        if self.has_red_tinted_rocks:
            # Create a mask for reddish hues (0-10 and 170-180)
            lower_red_mask = (h < 10)
            upper_red_mask = (h > 170)
            red_mask = lower_red_mask | upper_red_mask
            
            # Shift red hues slightly towards yellow/orange for more natural rock colors
            h_modified = h.copy()
            # For lower red hues, shift towards orange
            h_modified[lower_red_mask] = np.clip(h_modified[lower_red_mask] + 8, 0, 179)  # Increased from +5
            # For upper red hues, shift towards orange
            h_modified[upper_red_mask] = np.clip(h_modified[upper_red_mask] - 8, 0, 179)  # Increased from -5
            
            # Apply the modified hues
            h = h_modified
            
        # Apply enhanced saturation with protection for already saturated areas
        adjusted_saturation = self.saturation_factor * 1.1  # Reduced from 1.15
        s = np.clip(s * adjusted_saturation, 0, 255).astype(np.uint8)
        
        # Apply color-specific adjustments to fix green tinge
        if self.is_deep_blue:
            # For deep blue scenes, apply a slight red-blue boost to counter any green tinge
            hsv_back = cv2.merge([h, s, v])
            bgr = cv2.cvtColor(hsv_back, cv2.COLOR_HSV2BGR)
            b, g, r = cv2.split(bgr)
            
            # Very subtle reduction of green channel
            g = np.clip(g * 0.97, 0, 255).astype(np.uint8)
            # Additional blue reduction for deep blue scenes
            b = np.clip(b * 0.96, 0, 255).astype(np.uint8)
            
            # Merge back and convert to HSV again
            bgr_adjusted = cv2.merge([b, g, r])
            hsv_adjusted = cv2.cvtColor(bgr_adjusted, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_adjusted)
        else:
            # For other scenes, just apply a mild blue reduction
            hsv_back = cv2.merge([h, s, v])
            bgr = cv2.cvtColor(hsv_back, cv2.COLOR_HSV2BGR)
            b, g, r = cv2.split(bgr)
            
            # Subtle reduction of blue channel
            b = np.clip(b * 0.98, 0, 255).astype(np.uint8)
            
            # Merge back and convert to HSV again
            bgr_adjusted = cv2.merge([b, g, r])
            hsv_adjusted = cv2.cvtColor(bgr_adjusted, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_adjusted)
        
        # Boost brightness with highlights protection
        adjusted_brightness = self.brightness_boost * 1.02  # Reduced from 1.05
        v = np.clip(v * adjusted_brightness, 0, 255).astype(np.uint8)
        
        # Merge HSV channels and convert back to BGR
        adjusted_hsv = cv2.merge([h, s, v])
        adjusted_bgr = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        
        # Apply final tone mapping for better color balance
        final_image = self._tone_matching(adjusted_bgr)
        
        # Apply sharpening if requested
        if self.sharpness > 0:
            final_image = self._apply_sharpening(final_image)
        
        return final_image
    
    def _apply_white_balance(self, image):
        """Apply white balancing to correct color cast"""
        # Simple white balance using gray world assumption
        if self.white_balance_strength <= 0:
            return image
            
        result = np.zeros_like(image, dtype=np.float32)
            
        # Compute average for each channel
        b_avg = np.mean(image[:, :, 0])
        g_avg = np.mean(image[:, :, 1])
        r_avg = np.mean(image[:, :, 2])
        
        # Calculate average gray value
        avg_gray = (b_avg + g_avg + r_avg) / 3.0
        
        # Calculate scaling factors to achieve gray balance
        b_scale = avg_gray / b_avg if b_avg > 0 else 1.0
        g_scale = avg_gray / g_avg if g_avg > 0 else 1.0
        r_scale = avg_gray / r_avg if r_avg > 0 else 1.0
        
        # Apply weighted scaling with parameter control
        scales = np.array([
            1.0 + (b_scale - 1.0) * self.white_balance_strength,
            1.0 + (g_scale - 1.0) * self.white_balance_strength,
            1.0 + (r_scale - 1.0) * self.white_balance_strength
        ])
        
        # Apply scaling to each channel
        for i in range(3):
            result[:, :, i] = np.clip(image[:, :, i] * scales[i], 0, 255)
            
        return result.astype(np.uint8)
    
    def _boost_rgb_channels(self, image):
        """Boost RGB channels with automatically tuned parameters"""
        b, g, r = cv2.split(image)
        
        # Apply channel-specific boosts with auto-tuned parameters
        r_boosted = np.clip(r * self.red_boost, 0, 255).astype(np.uint8)
        g_adjusted = np.clip(g * self.green_factor, 0, 255).astype(np.uint8)
        b_boosted = np.clip(b * self.blue_boost, 0, 255).astype(np.uint8)
        
        return cv2.merge([b_boosted, g_adjusted, r_boosted])
    
    def _tone_matching(self, image):
        """Apply tone mapping to enhance color vibrance and reduce red tint"""
        # Split into channels
        b, g, r = cv2.split(image)
        
        # Apply mild histogram equalization to match target distribution
        b_eq = cv2.equalizeHist(b)
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        
        # Blending settings depend on detection results
        if self.has_red_tinted_rocks:
            # For rock scenes, minimize red equalization and boost greens more
            r_blend = 0.95  # Only 5% red equalization 
            g_blend = 0.55   # 45% green equalization (increased from 0.5)
            b_blend = 0.6   # 40% blue equalization
        elif self.is_deep_blue:
            # For deep blue scenes, minimize blue equalization
            r_blend = 0.85   # 15% red equalization for better balance
            g_blend = 0.65   # 35% green equalization (increased from 0.55 to reduce green tinge)
            b_blend = 0.88   # Only 12% blue equalization to reduce blue intensity (increased from 0.8)
        else:
            # Standard blending for normal scenes
            r_blend = 0.88   # 12% red equalization (slightly less than previous 0.9)
            g_blend = 0.7    # 30% green equalization (increased from 0.6 to reduce green)
            b_blend = 0.75   # 25% blue equalization (increased from 0.65)
            
        # Apply the blending
        b_final = cv2.addWeighted(b, b_blend, b_eq, 1.0 - b_blend, 0)
        r_final = cv2.addWeighted(r, r_blend, r_eq, 1.0 - r_blend, 0)
        g_final = cv2.addWeighted(g, g_blend, g_eq, 1.0 - g_blend, 0)
        
        # Additional blue reduction for all scenes to eliminate the blue sheen
        b_final = np.clip(b_final * 0.97, 0, 255).astype(np.uint8)
        
        # Create the final image with adjusted channels
        result = cv2.merge([b_final, g_final, r_final])
        
        # Apply enhanced contrast curve for richer colors
        result_float = result.astype(np.float32) / 255.0
        result_curve = result_float ** 0.95  # Slight contrast enhancement
        result = (result_curve * 255).astype(np.uint8)
        
        # Apply a very slight bilateral filter to reduce noise while preserving edges
        result = cv2.bilateralFilter(result, 5, 25, 25)
        
        return result

    def _apply_vibrance(self, image, factor=0.4):
        """Apply vibrance adjustment (smart saturation) to make colors pop while protecting already saturated areas"""
        # Convert to HSV for easy saturation manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create a mask of less-saturated pixels
        # Less saturated pixels get more saturation boost
        max_s = np.max(s)
        if max_s > 0:
            # Normalized saturation (0-1 scale)
            norm_s = s.astype(np.float32) / 255.0
            
            # Calculate adjustment factor based on existing saturation
            # Less saturated pixels get more boost, already saturated get less
            adjustment = (1.0 - norm_s) * factor
            
            # Apply adjustment
            s_float = s.astype(np.float32)
            s = np.clip(s_float * (1.0 + adjustment), 0, 255).astype(np.uint8)
        
        # Recombine the channels
        enhanced_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    def _apply_sharpening(self, image):
        """Apply sharpening to the image with adjustable strength"""
        # No sharpening if strength is 0
        if self.sharpness <= 0:
            return image
            
        # Create a blurred version for unsharp mask
        blur = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Create the unsharp mask by subtracting the blurred image
        unsharp_mask = cv2.addWeighted(image, 1.0 + self.sharpness, blur, -self.sharpness, 0)
        
        # Apply a slight bilateral filter to reduce noise enhancement from sharpening
        if self.sharpness > 0.3:
            unsharp_mask = cv2.bilateralFilter(unsharp_mask, 5, 20, 20)
            
        return unsharp_mask

    def process_image(self, input_path, output_path):
        """
        Process a single image file with auto-optimized color parameters
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image
        """
        start_time = time.time()
        logger.info(f"Processing image: {input_path}")
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Error loading image: {input_path}")
        
        logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
        
        # Reset optimization for each new image
        self.has_optimized_params = False
            
        # Process image with auto-optimization
        logger.info("Analyzing image and applying auto color restoration...")
        corrected = self.restore_color(image, optimize_params=True)
        
        # Save with OpenCV first
        cv2.imwrite(output_path, corrected)
        logger.info(f"Saved corrected image: {output_path}")
        
        # Now transfer metadata from original to new image
        try:
            # Try to preserve EXIF data
            if input_path.lower().endswith(('.jpg', '.jpeg')):
                logger.info("Preserving EXIF metadata...")
                original_exif = piexif.load(input_path)
                piexif.insert(piexif.dump(original_exif), output_path)
                logger.info("EXIF metadata transferred successfully")
                
                # Additionally try to preserve location data with exiftool
                self._preserve_metadata(input_path, output_path)
        except Exception as e:
            logger.warning(f"Could not preserve metadata: {e}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

    def _process_frame(self, frame_path, output_frame_path, is_sample_frame=False):
        """
        Process a single frame
        
        Args:
            frame_path: Path to input frame
            output_frame_path: Path to save processed frame
            is_sample_frame: Whether this is a sample frame for parameter optimization
        """
        # Check for termination request
        if terminate_requested:
            return False
            
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return False
        
        # For sample frames or if parameters aren't optimized yet, do full optimization
        optimize = is_sample_frame or not self.has_optimized_params
            
        corrected = self.restore_color(frame, optimize_params=optimize)
        cv2.imwrite(output_frame_path, corrected)
        return True

    def _preserve_metadata(self, source_file, target_file):
        """Preserve metadata from source file to target file"""
        logger.info("Ensuring all metadata is preserved...")

        if not check_exiftool_available():
            logger.warning("Cannot preserve metadata: exiftool not available")
            return False
            
        # First, check if source has GPS data
        gps_info_cmd = ['exiftool', '-s', '-G', '-Location', '-GPS*', source_file]
        result = subprocess.run(gps_info_cmd, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
        
        if result.stdout.strip():
            logger.info(f"Found location data in source: {result.stdout.strip()}")
            
            # Get appropriate exiftool command for this file type
            cmd = get_exiftool_cmd_for_metadata_copy(source_file, target_file)
            if cmd:
                # Run the command
                result = subprocess.run(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
                
                success = result.returncode == 0
                if success:
                    logger.info("Successfully preserved metadata")
                    
                    # Apply QuickTime-specific fix if needed
                    if is_quicktime_format(target_file):
                        fix_quicktime_metadata(target_file)
                else:
                    logger.warning(f"Error preserving metadata: {result.stderr}")
                return success
        else:
            logger.warning(f"No location data found in source file: {source_file}")
            
        return False

    def process_video(self, input_path, output_path, fps=None, num_workers=None, crf=24):
        """
        Process a video file with auto-optimized color parameters
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            fps: Frames per second (if None, detect from source)
            num_workers: Number of parallel workers (defaults to CPU count)
            crf: Constant Rate Factor for video quality (lower = better quality, higher = smaller size)
        """
        start_time = time.time()
        logger.info(f"Starting video processing for: {input_path}")
        logger.info("Press Ctrl+C at any time to gracefully cancel processing")
        
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        
        logger.info(f"Using {num_workers} parallel workers for frame processing")
        
        # Special handling for Apple formats (MOV)
        is_quicktime = is_quicktime_format(input_path)
        
        # Create temporary directory for frames and intermediate files
        temp_dir = tempfile.mkdtemp()
        try:
            # Get video properties
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {input_path}")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps is None:
                fps = source_fps
            
            logger.info(f"Video info: {width}x{height} at {source_fps} fps, {frame_count} frames")
                
            cap.release()
            
            # Extract frames
            logger.info("Extracting frames with ffmpeg...")
            frames_path = os.path.join(temp_dir, "frame_%04d.jpg")
            extraction_command = self._create_ffmpeg_extraction_command(input_path, frames_path, num_workers)
            self._run_ffmpeg_with_progress(extraction_command, frame_count, "Extracting frames")
            
            # Get all extracted frames
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_")])
            
            if not frame_files:
                raise ValueError("No frames were extracted from the video")
            
            # Reset optimization state for new video
            self.has_optimized_params = False
            
            # First, analyze a sample of frames to determine optimal parameters
            logger.info("Analyzing sample frames to determine optimal color parameters...")
            # Use a subset of frames for analysis
            sample_interval = max(1, len(frame_files) // 10)
            sample_frames = frame_files[::sample_interval][:5]  # Take up to 5 frames
            
            # Process sample frames to determine optimal parameters
            for i, filename in enumerate(sample_frames):
                sample_frame_path = os.path.join(temp_dir, filename)
                sample_output_path = os.path.join(temp_dir, f"sample_{i}.jpg")
                self._process_frame(sample_frame_path, sample_output_path, is_sample_frame=True)
            
            # Now process all frames in parallel with the optimized parameters
            logger.info(f"Processing {len(frame_files)} frames with optimized parameters...")
            
            # Process frames in parallel using ThreadPoolExecutor
            completed_frames = self._process_frames_parallel(temp_dir, frame_files, num_workers)
            
            if terminate_requested:
                logger.info(f"Processing cancelled. {completed_frames} frames were processed.")
                return
            
            # Create a temporary processed video file
            temp_processed = os.path.join(temp_dir, "processed_video.mp4")
            
            # Create video from processed frames
            logger.info("Creating processed video...")
            corrected_frames_path = os.path.join(temp_dir, "corrected_%04d.jpg")
            encoding_command = self._create_ffmpeg_encoding_command(
                corrected_frames_path, temp_processed, fps, width, height, num_workers, crf=crf
            )
            self._run_ffmpeg_with_progress(encoding_command, completed_frames, "Creating video")
            
            # Copy metadata to the processed video
            self._copy_video_metadata(input_path, temp_processed, output_path, is_quicktime, temp_dir)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Saved corrected video: {output_path}")
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                logger.info("Cleaning up temporary files...")
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_frames_parallel(self, temp_dir, frame_files, num_workers):
        """Process frames in parallel and return the count of successfully processed frames"""
        completed_frames = 0
        futures = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all frame processing tasks
            for i, filename in enumerate(frame_files):
                if terminate_requested:
                    break
                    
                futures.append((
                    executor.submit(
                        self._process_frame,
                        os.path.join(temp_dir, filename),
                        os.path.join(temp_dir, f"corrected_{i+1:04d}.jpg")
                    ),
                    i,
                    filename
                ))
            
            # Process results as they complete
            progress_bar = tqdm(total=len(futures), desc="Processing frames")
            for future, i, filename in [(f[0], f[1], f[2]) for f in futures]:
                if future.done() or future.running():
                    try:
                        success = future.result()
                        if success:
                            completed_frames += 1
                        else:
                            logger.warning(f"Failed to process frame {i+1}")
                    except Exception as e:
                        logger.error(f"Error processing frame {i+1}: {e}")
                        
                    progress_bar.update(1)
                    
                if terminate_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            
            progress_bar.close()
        
        return completed_frames

    def _copy_video_metadata(self, input_path, temp_processed, output_path, is_quicktime, temp_dir):
        """Copy metadata from original video to processed video"""
        if is_quicktime:
            logger.info("Using optimized approach for preserving metadata in QuickTime format")
            
            # Create a temporary file path for the output
            temp_output_path = os.path.join(temp_dir, "final_with_metadata.mov")
            
            # Command for copying metadata
            metadata_command = self._create_ffmpeg_metadata_command(
                input_path, temp_processed, temp_output_path, is_quicktime=True
            )
            
            # Run the command to copy metadata
            logger.info("Copying metadata from original to processed video...")
            self._run_ffmpeg_with_progress(metadata_command, 1, "Copying metadata")
            
            # For Apple MOV files, apply additional fixes with exiftool
            logger.info("Applying exiftool fixes for Apple QuickTime metadata...")
            try:
                # Copy all metadata first
                if copy_all_metadata_with_exiftool(input_path, temp_output_path):
                    # Then apply the specialized QuickTime fix
                    if fix_quicktime_metadata(temp_output_path):
                        # Move the final version to the requested output path
                        shutil.copy2(temp_output_path, output_path)
                        logger.info("Successfully applied metadata fixes with exiftool")
                    else:
                        logger.warning("QuickTime metadata fix failed, using basic metadata copy")
                        shutil.copy2(temp_output_path, output_path)
                else:
                    logger.warning("Metadata copy with exiftool failed, using basic metadata copy")
                    shutil.copy2(temp_output_path, output_path)
            except Exception as e:
                logger.warning(f"Error applying exiftool fixes: {e}")
                # If exiftool fails, just use the ffmpeg output
                shutil.copy2(temp_output_path, output_path)
        else:
            # For non-QuickTime files, use the standard approach
            logger.info("Using standard metadata preservation approach")
            
            # Create and run command for copying metadata
            metadata_command = self._create_ffmpeg_metadata_command(
                input_path, temp_processed, output_path, is_quicktime=False
            )
            self._run_ffmpeg_with_progress(metadata_command, 1, "Copying metadata")
        
        # Verify location data was preserved
        verify_location_metadata(input_path, output_path)

    def process_batch(self, input_folder, output_folder, fps=None, num_workers=None, crf=24, output_format=None):
        """
        Process all supported files in a folder
        
        Args:
            input_folder: Path to input folder containing images and/or videos
            output_folder: Path to save processed files
            fps: Frames per second for videos (if None, detect from source)
            num_workers: Number of parallel workers (defaults to CPU count)
            crf: Constant Rate Factor for video quality (lower = better quality, higher = smaller size)
            output_format: If specified, convert all output files to this format (e.g., 'mp4')
        """
        start_time = time.time()
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all supported files in the input folder
        input_files = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.lower().endswith(SUPPORTED_EXTENSIONS):
                    input_files.append(file_path)
        
        if not input_files:
            logger.warning(f"No supported files found in {input_folder}")
            return
            
        logger.info(f"Found {len(input_files)} files to process in {input_folder}")
        
        # Process each file
        successful = 0
        failed = 0
        
        for input_file in tqdm(input_files, desc="Processing files"):
            if terminate_requested:
                logger.info("Termination requested. Stopping batch processing.")
                break
                
            try:
                # Determine relative path to maintain folder structure
                rel_path = os.path.relpath(input_file, input_folder)
                output_file = os.path.join(output_folder, rel_path)
                
                # Modify output extension if output_format is specified
                if output_format and output_format.lower() == 'mp4' and is_video_file(input_file):
                    output_file = os.path.splitext(output_file)[0] + '.mp4'
                
                # Create necessary subdirectories
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Reset optimization state for each new file
                self.has_optimized_params = False
                
                if is_image_file(input_file):
                    logger.info(f"Processing image: {input_file}")
                    self.process_image(input_file, output_file)
                    successful += 1
                elif is_video_file(input_file):
                    logger.info(f"Processing video: {input_file}")
                    self.process_video(input_file, output_file, fps=fps, num_workers=num_workers, crf=crf)
                    successful += 1
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
                failed += 1
        
        elapsed_time = time.time() - start_time
        if terminate_requested:
            logger.info(f"Batch processing terminated. Processed {successful} files successfully, {failed} failed.")
        else:
            logger.info(f"Batch processing completed. Processed {successful} files successfully, {failed} failed.")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds") 

# Utility functions copied from v4
def verify_location_metadata(input_file, output_file):
    """
    Verify that location metadata was preserved by checking both files
    
    Args:
        input_file: Original input file path
        output_file: Processed output file path
        
    Returns:
        bool: True if location metadata was preserved, False otherwise
    """
    logger.info("Verifying location metadata preservation...")
    
    if not check_exiftool_available():
        logger.warning("Cannot verify location metadata: exiftool not available")
        return False
    
    try:
        # Use exiftool to check GPS data in both files
        cmd_original = ['exiftool', '-s', '-G', '-Location', '-GPS*', input_file]
        cmd_processed = ['exiftool', '-s', '-G', '-Location', '-GPS*', output_file]
        
        result_original = subprocess.run(cmd_original, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
        
        result_processed = subprocess.run(cmd_processed, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        # Check if original has GPS data
        original_gps = result_original.stdout.strip()
        if not original_gps:
            logger.warning(f"No location data found in original file: {input_file}")
            return False
            
        # Check if processed file has GPS data
        processed_gps = result_processed.stdout.strip()
        if not processed_gps:
            logger.error(f"Location data NOT preserved in processed file: {output_file}")
            logger.error(f"Original GPS data: {original_gps}")
            return False
            
        # Compare data (exact match not required, just ensure processed file has GPS data)
        logger.info(f"Original GPS data: {original_gps}")
        logger.info(f"Processed GPS data: {processed_gps}")
        
        # Basic verification that key GPS tags are present
        if "GPSLatitude" in processed_gps and "GPSLongitude" in processed_gps:
            logger.info("✅ Location metadata successfully preserved")
            return True
        else:
            logger.error("❌ Location metadata not fully preserved")
            return False
    except Exception as e:
        logger.error(f"Error verifying location metadata: {e}")
        return False

def get_exiftool_cmd_for_metadata_copy(source_file, target_file):
    """Generate appropriate exiftool command for copying metadata"""
    if is_quicktime_format(target_file):
        # Extract GPS data from source
        extract_cmd = [
            'exiftool', '-j', 
            '-GPS:GPSLatitude',
            '-GPS:GPSLatitudeRef',
            '-GPS:GPSLongitude',
            '-GPS:GPSLongitudeRef',
            source_file
        ]
        
        extract_result = subprocess.run(extract_cmd, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
        
        if extract_result.returncode != 0:
            logger.warning(f"Failed to extract GPS data: {extract_result.stderr}")
            return None
        
        try:
            # Parse JSON result
            gps_data = json.loads(extract_result.stdout)[0]
            
            # Check if we have GPS data
            if not ('GPSLatitude' in gps_data and 'GPSLongitude' in gps_data):
                logger.warning("GPS data extraction returned empty results")
                return None
                
            # Apply GPS data directly to target in the QuickTime format
            lat = gps_data.get('GPSLatitude', '')
            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
            lon = gps_data.get('GPSLongitude', '')
            lon_ref = gps_data.get('GPSLongitudeRef', 'E')
            
            logger.info(f"Found GPS: Lat: {lat} {lat_ref}, Lon: {lon} {lon_ref}")
            
            location_string = f"{lat} {lat_ref}, {lon} {lon_ref}"
            
            # Apply to the target file with various tag formats for maximum compatibility
            return [
                'exiftool',
                '-GPSLatitude=%s' % lat,
                '-GPSLatitudeRef=%s' % lat_ref,
                '-GPSLongitude=%s' % lon,
                '-GPSLongitudeRef=%s' % lon_ref,
                # Add QuickTime specific location tags
                '-XMP:GPSLatitude=%s' % lat,
                '-XMP:GPSLongitude=%s' % lon,
                # Add as a human-readable location string
                f'-LocationCreated="{location_string}"',
                f'-LocationTaken="{location_string}"',
                f'-Location="{location_string}"',
                # Update metadata from original
                '-TagsFromFile', source_file,
                '-xmp:all',
                '-iptc:all',
                '-P',  # Preserve existing tags
                '-overwrite_original',
                target_file
            ]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error processing GPS data: {e}")
            return None
    else:
        # For non-MOV/MP4 files, use the standard approach
        return [
            'exiftool', '-TagsFromFile', source_file,
            '-gps:all',
            '-location:all',
            '-coordinates:all',
            '-xmp:geotag',
            '-P',
            '-overwrite_original',
            target_file
        ]

def fix_quicktime_metadata(file_path):
    """Apply QuickTime metadata fix to ensure GPS data compatibility with Apple devices"""
    if not check_exiftool_available():
        return False

    cmd = [
        'exiftool',
        '-m',                   # Ignore minor errors
        '-overwrite_original',  # Overwrite the original file
        '-api', 'LargeFileSupport=1',  # Support for large video files
        '-Keys:All=',           # First clear the Keys to avoid conflicts
        '-tagsFromFile', '@',   # Copy tags from the same file (self-reference)
        '-Keys:All',            # Then copy back all the Keys (including location)
        file_path               # Target file to modify
    ]
    
    # Execute the command
    result = subprocess.run(cmd, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         text=True)
    
    return result.returncode == 0

def copy_all_metadata_with_exiftool(source_file, target_file):
    """Copy all metadata from source to target file using exiftool"""
    if not check_exiftool_available():
        return False

    cmd = [
        'exiftool',
        '-m',                        # Ignore minor errors
        '-overwrite_original',       # Overwrite the output file
        '-api', 'LargeFileSupport=1',  # Support for large video files
        '-tagsFromFile', source_file, # Copy tags from the input file
        '-all:all',                  # Copy all tags
        '-GPS:all',                  # Specifically ensure GPS tags are copied
        '-Apple:all',                # Copy all Apple-specific tags
        '-quicktime:all',            # Copy all QuickTime tags
        target_file                  # The output file to modify
    ]
    
    result = subprocess.run(cmd, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=False)
    
    return result.returncode == 0

def test_location_preservation(input_file=None):
    """
    Run test for location metadata preservation
    
    Args:
        input_file: Optional specific file to test. If not provided, a sample file will be used.
                  
    Returns:
        bool: True if test passes, False otherwise
    """
    import tempfile
    import os
    
    if not input_file:
        logger.error("No test file provided.")
        return False
        
    logger.info(f"Running location metadata preservation test with: {input_file}")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_file)[1], delete=False) as temp_file:
        output_file = temp_file.name
        
    try:
        # Process the file
        processor = AutoColorUnderwaterImageProcessor()
        
        if is_image_file(input_file):
            processor.process_image(input_file, output_file)
        elif is_video_file(input_file):
            processor.process_video(input_file, output_file)
        else:
            logger.error(f"Unsupported file format: {input_file}")
            os.unlink(output_file)
            return False
            
        # Verify location metadata was preserved
        result = verify_location_metadata(input_file, output_file)
        
        if result:
            logger.info("✅ TEST PASSED: Location metadata was correctly preserved")
        else:
            logger.error("❌ TEST FAILED: Location metadata was NOT correctly preserved")
            
        return result
    finally:
        # Clean up
        try:
            os.unlink(output_file)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Enhance underwater images and videos by restoring natural colors with automatic color parameter optimization (v5).")
    parser.add_argument("input", help="Path to the input image, video file, or folder containing images/videos.")
    parser.add_argument("--output", "-o", help="Path to save the corrected output file or folder. If not provided, defaults to '<original_file_name>_converted.<original_filetype>' or '<original_folder_name>_converted'.")
    parser.add_argument("--contrast", "-c", type=float, default=2.0, help="CLAHE clip limit for contrast enhancement")
    parser.add_argument("--saturation", "-s", type=float, default=0.9, help="Saturation factor (0-1 range)")
    parser.add_argument("--brightness", "-B", type=float, default=1.12, help="Brightness boost factor")
    parser.add_argument("--white-balance", "-wb", type=float, default=0.7, help="White balance strength (0-1 range)")
    parser.add_argument("--auto-tune", "-a", type=float, default=0.9, help="Auto-tuning strength (0-1 range). Controls how aggressively the algorithm applies optimized color parameters.")
    parser.add_argument("--sharpness", "-sh", type=float, default=0.3, help="Sharpness enhancement factor (0-1 range). 0 for no sharpening.")
    parser.add_argument("--fps", type=float, help="Output frames per second (default: same as input)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker threads for parallel processing (default: CPU count)")
    parser.add_argument("--batch", action="store_true", help="Process input as a folder containing multiple files")
    parser.add_argument("--test-location", "-t", action="store_true", help="Run test for location metadata preservation")
    parser.add_argument("--quality", "-q", type=int, default=24, help="Video quality (CRF value: 18-28 range, lower = better quality, higher = smaller size)")
    parser.add_argument("--high-quality", "-hq", action="store_true", help="Use high quality preset (CRF 20, slow preset) for best quality output")
    parser.add_argument("--output-format", "-f", choices=["mp4"], help="Convert output videos to the specified format. Currently only supports mp4.")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('UnderwaterConverter').setLevel(logging.DEBUG)
    
    logger.info("Underwater Footage Converter v5.0 with Automatic Color Optimization")
    
    # Test mode for location metadata preservation
    if args.test_location:
        logger.info("Running in TEST MODE for location metadata preservation")
        result = test_location_preservation(args.input)
        return 0 if result else 1
    
    logger.info(f"Input: {args.input}")
    
    # Generate output filename/folder if not provided
    if not args.output:
        input_path = Path(args.input)
        if os.path.isdir(args.input) or args.batch:
            args.output = str(input_path.with_name(f"{input_path.name}_converted"))
        else:
            output_suffix = input_path.suffix
            # If output format is specified and is a video file, use that format
            if args.output_format and args.output_format.lower() == 'mp4' and is_video_file(args.input):
                output_suffix = '.mp4'
            args.output = str(input_path.with_name(f"{input_path.stem}_converted{output_suffix}"))
    
    logger.info(f"Output: {args.output}")
    
    # Apply high quality settings if requested
    if args.high_quality:
        args.quality = 20
        logger.info("Using high quality preset (CRF 20)")
    
    # Initialize processor with command-line parameters
    processor = AutoColorUnderwaterImageProcessor(
        contrast_limit=args.contrast,
        saturation_factor=args.saturation,
        brightness_boost=args.brightness,
        white_balance_strength=args.white_balance,
        auto_tune_strength=args.auto_tune,
        sharpness=args.sharpness
    )
    
    logger.info(f"Processing parameters: contrast={args.contrast}, saturation={args.saturation}, " +
                f"brightness={args.brightness}, white_balance={args.white_balance}, " + 
                f"auto_tune_strength={args.auto_tune}, sharpness={args.sharpness}, " +
                f"video quality={args.quality}")
    if args.output_format:
        logger.info(f"Output format set to: {args.output_format}")
    logger.info("RGB color parameters will be automatically optimized for each image/video")
    
    # Process based on input type
    if os.path.isdir(args.input) or args.batch:
        # Process as batch
        logger.info(f"Processing folder: {args.input}")
        processor.process_batch(args.input, args.output, fps=args.fps, num_workers=args.workers, 
                               crf=args.quality, output_format=args.output_format)
    else:
        # Process single file
        if is_image_file(args.input):
            processor.process_image(args.input, args.output)
        elif is_video_file(args.input):
            # Check if we should force MP4 output
            if args.output_format and args.output_format.lower() == 'mp4':
                output_path = args.output
                # Ensure the output has .mp4 extension
                if not output_path.lower().endswith('.mp4'):
                    output_path = os.path.splitext(output_path)[0] + '.mp4'
                processor.process_video(args.input, output_path, fps=args.fps, num_workers=args.workers, crf=args.quality)
            else:
                processor.process_video(args.input, args.output, fps=args.fps, num_workers=args.workers, crf=args.quality)
        else:
            logger.error(f"Unsupported file format: {args.input}")
            return 1
        
    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    main() 