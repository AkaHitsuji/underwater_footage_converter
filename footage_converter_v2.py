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

class UnderwaterImageProcessor:
    """Class for processing underwater images and videos to restore natural colors"""
    
    def __init__(self, dehaze_strength=0.5, clahe_clip=3.0, saturation_boost=1.5, red_boost=1.05):
        """
        Initialize the underwater image processor
        
        Args:
            dehaze_strength: Strength of dehazing effect (0.0-1.0)
            clahe_clip: Clip limit for CLAHE algorithm
            saturation_boost: Multiplier for saturation enhancement
            red_boost: Red channel multiplier (1.0 = no change, >1.0 = more red)
        """
        self.dehaze_strength = dehaze_strength
        self.clahe_clip = clahe_clip
        self.saturation_boost = saturation_boost
        self.red_boost = red_boost
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
    
    def _create_ffmpeg_extraction_command(self, input_path, frames_path, num_workers):
        """Create FFmpeg command for extracting frames from video"""
        return [
            'ffmpeg', '-i', input_path, '-qscale:v', '2', 
            '-threads', str(num_workers),
            frames_path
        ]

    def _create_ffmpeg_encoding_command(self, frames_path, output_path, fps, width, height, num_workers, crf=28):
        """Create FFmpeg command for encoding frames into video"""
        return [
            'ffmpeg', '-r', str(fps),
            '-i', frames_path,
            '-c:v', 'libx265',  # Use HEVC codec instead of H.264
            '-preset', 'medium',  # Use medium preset (balance between speed and compression)
            '-crf', str(crf),  # Higher CRF value (23-28 range for HEVC is visually lossless but smaller)
            '-pix_fmt', 'yuv420p',
            '-threads', str(num_workers),
            '-vf', f'scale={width}:{height}',
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

    def restore_color(self, image):
        """
        Restore natural colors in underwater image
        
        Args:
            image: Input BGR image (numpy array)
            
        Returns:
            Corrected BGR image (numpy array)
        """
        # Apply dehazing
        dehazed = self._apply_dehazing(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for contrast enhancement
        l = self.clahe.apply(l)
        
        # Enhance the a channel (green-red) to restore lost reds
        # Reduced from +15 to +10 for more natural look
        a = cv2.add(a, 10)  # Subtle shift toward red
        
        # Enhance the b channel (blue-yellow) to balance colors
        b = cv2.add(b, 5)   # Subtle shift toward yellow
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # White balance correction
        wb_corrected = self._white_balance(restored)
        
        # Convert to HSV for saturation boost
        hsv = cv2.cvtColor(wb_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation
        s = np.clip(s * self.saturation_boost, 0, 255).astype(np.uint8)
        
        # Merge and convert back to BGR
        hsv = cv2.merge((h, s, v))
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return final
    
    def _apply_dehazing(self, image):
        """Apply dehazing to reduce water turbidity effects"""
        if self.dehaze_strength == 0:
            return image
            
        # Dark channel prior dehazing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(gray, np.ones((15, 15), np.uint8))
        
        # Estimate atmospheric light
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape([-1, 3])
        indices = np.argsort(flat_dark)[-int(0.001 * dark_channel.size):]
        atmospheric = np.max(flat_image[indices], axis=0)
        
        # Calculate transmission map
        normalized = image.astype(np.float32) / atmospheric
        transmission = 1 - self.dehaze_strength * cv2.min(
            cv2.min(normalized[:, :, 0], normalized[:, :, 1]), 
            normalized[:, :, 2]
        )
        
        # Refine transmission map
        transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Recover scene radiance
        result = np.empty_like(image, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (image[:, :, i].astype(np.float32) - atmospheric[i]) / \
                              transmission + atmospheric[i]
                              
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _white_balance(self, image):
        """Apply white balance correction"""
        # Calculate color averages
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Calculate scales to balance colors
        avg_all = (avg_b + avg_g + avg_r) / 3
        scale_b = avg_all / max(avg_b, 1)  # Avoid division by zero
        scale_g = avg_all / max(avg_g, 1)
        scale_r = avg_all / max(avg_r, 1)
        
        # Apply user-controlled red boost
        scale_r *= self.red_boost
        
        # Apply correction
        balanced = np.clip(image.astype(np.float32) * [scale_b, scale_g, scale_r], 0, 255).astype(np.uint8)
        return balanced

    def process_image(self, input_path, output_path):
        """
        Process a single image file
        
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
            
        # Process image
        logger.info("Applying color restoration...")
        corrected = self.restore_color(image)
        
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

    def _process_frame(self, frame_path, output_frame_path):
        """Process a single frame"""
        # Check for termination request
        if terminate_requested:
            return False
            
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return False
            
        corrected = self.restore_color(frame)
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

    def process_video(self, input_path, output_path, fps=None, num_workers=None, crf=28):
        """
        Process a video file
        
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
            
            # Process frames in parallel
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_")])
            logger.info(f"Processing {len(frame_files)} frames with {num_workers} workers...")
            
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

    def process_batch(self, input_folder, output_folder, fps=None, num_workers=None, crf=28):
        """
        Process all supported files in a folder
        
        Args:
            input_folder: Path to input folder containing images and/or videos
            output_folder: Path to save processed files
            fps: Frames per second for videos (if None, detect from source)
            num_workers: Number of parallel workers (defaults to CPU count)
            crf: Constant Rate Factor for video quality (lower = better quality, higher = smaller size)
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
                
                # Create necessary subdirectories
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
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
        processor = UnderwaterImageProcessor()
        
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
    parser = argparse.ArgumentParser(description="Enhance underwater images and videos by restoring natural colors.")
    parser.add_argument("input", help="Path to the input image, video file, or folder containing images/videos.")
    parser.add_argument("--output", "-o", help="Path to save the corrected output file or folder. If not provided, defaults to '<original_file_name>_converted.<original_filetype>' or '<original_folder_name>_converted'.")
    parser.add_argument("--dehaze", "-d", type=float, default=0.5, help="Strength of dehazing effect (0.0-1.0)")
    parser.add_argument("--clahe", "-c", type=float, default=3.0, help="CLAHE clip limit for contrast enhancement")
    parser.add_argument("--saturation", "-s", type=float, default=1.5, help="Saturation boost factor")
    parser.add_argument("--red", "-r", type=float, default=1.05, help="Red channel boost factor (1.0 = neutral, higher = more red)")
    parser.add_argument("--fps", type=float, help="Output frames per second (default: same as input)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker threads for parallel processing (default: CPU count)")
    parser.add_argument("--fast", "-f", action="store_true", help="Use faster processing with slightly lower quality")
    parser.add_argument("--batch", "-b", action="store_true", help="Process input as a folder containing multiple files")
    parser.add_argument("--test-location", "-t", action="store_true", help="Run test for location metadata preservation")
    parser.add_argument("--quality", "-q", type=int, default=28, help="Video quality (CRF value: 18-28 range, lower = higher quality and larger size)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('UnderwaterConverter').setLevel(logging.DEBUG)
    
    logger.info("Underwater Footage Converter v2.0")
    
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
            args.output = str(input_path.with_name(f"{input_path.stem}_converted{input_path.suffix}"))
    
    logger.info(f"Output: {args.output}")
    
    # Adjust parameters for fast mode
    if args.fast:
        logger.info("Using fast mode with optimized parameters")
        # In fast mode, reduce dehaze strength for faster processing
        args.dehaze = min(args.dehaze, 0.3)
    
    # Initialize processor
    processor = UnderwaterImageProcessor(
        dehaze_strength=args.dehaze,
        clahe_clip=args.clahe,
        saturation_boost=args.saturation,
        red_boost=args.red
    )
    
    logger.info(f"Processing parameters: dehaze={args.dehaze}, clahe={args.clahe}, saturation={args.saturation}, red_boost={args.red}, video quality={args.quality}")
    
    # Process based on input type
    if os.path.isdir(args.input) or args.batch:
        # Process as batch
        logger.info(f"Processing folder: {args.input}")
        processor.process_batch(args.input, args.output, fps=args.fps, num_workers=args.workers, crf=args.quality)
    else:
        # Process single file
        if is_image_file(args.input):
            processor.process_image(args.input, args.output)
        elif is_video_file(args.input):
            processor.process_video(args.input, args.output, fps=args.fps, num_workers=args.workers, crf=args.quality)
        else:
            logger.error(f"Unsupported file format: {args.input}")
            return 1
        
    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    main() 