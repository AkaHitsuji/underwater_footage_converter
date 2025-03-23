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
                logger.info("Ensuring location metadata is preserved...")
                self._copy_location_metadata_with_exiftool(input_path, output_path)
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

    def process_video(self, input_path, output_path, fps=None, num_workers=None):
        """
        Process a video file
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            fps: Frames per second (if None, detect from source)
            num_workers: Number of parallel workers (defaults to CPU count)
        """
        start_time = time.time()
        logger.info(f"Starting video processing for: {input_path}")
        logger.info("Press Ctrl+C at any time to gracefully cancel processing")
        
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        
        logger.info(f"Using {num_workers} parallel workers for frame processing")
        
        # Is this a MOV file? Special handling for Apple formats
        is_mov_file = input_path.lower().endswith('.mov')
        needs_metadata_preservation = is_mov_file  # We especially care about MOV formats
        
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
            
            # Extract frames using ffmpeg (more efficient than OpenCV for extraction)
            logger.info("Extracting frames with ffmpeg...")
            frames_path = os.path.join(temp_dir, "frame_%04d.jpg")
            
            # Use high-quality extraction for better results
            extraction_command = [
                'ffmpeg', '-i', input_path, '-qscale:v', '2', 
                # Use multiple threads for extraction
                '-threads', str(num_workers),
                frames_path
            ]
            
            # Use new progress monitoring method
            self._run_ffmpeg_with_progress(
                extraction_command, 
                frame_count, 
                "Extracting frames"
            )
            
            # Process frames in parallel
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_")])
            
            logger.info(f"Processing {len(frame_files)} frames with {num_workers} workers...")
            
            # Process frames in parallel using ThreadPoolExecutor
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
            
            if terminate_requested:
                logger.info(f"Processing cancelled. {completed_frames} frames were processed.")
                return
            
            # Create a temporary processed video file (for intermediate processing)
            temp_processed = os.path.join(temp_dir, "processed_video.mp4")
            
            # Create video from processed frames
            logger.info("Creating processed video...")
            encoding_command = [
                'ffmpeg', '-r', str(fps),
                '-i', os.path.join(temp_dir, "corrected_%04d.jpg"),
                # Use optimized encoding settings
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                # Use multiple threads for encoding
                '-threads', str(num_workers),
                # Add these options to maintain original dimensions
                '-vf', f'scale={width}:{height}',
                # Add movflags for better compatibility
                '-movflags', '+faststart',
                temp_processed
            ]
            
            # Monitor progress during encoding
            self._run_ffmpeg_with_progress(
                encoding_command,
                completed_frames,
                "Creating video"
            )
            
            # Special handling for Apple MOV files with GPS data
            if needs_metadata_preservation:
                logger.info("Using optimized approach for preserving metadata in MOV file")
                
                # For MOV files, we'll use the reddit approach but with specific flags
                # This approach prioritizes metadata preservation for Apple devices
                
                # Create a temporary file path for the output
                temp_output_path = os.path.join(temp_dir, "final_with_metadata.mov")
                
                # Command that takes the processed video and copies metadata from the original
                # This specific order is important for Apple devices
                metadata_command = [
                    'ffmpeg',
                    '-i', input_path,          # Original file with metadata (first input)
                    '-i', temp_processed,      # Processed video (second input)
                    '-map', '1:v',             # Use video from the processed file (second input)
                    '-map_metadata', '0',      # Use metadata from the original file (first input)
                    # Copy audio, subtitles, and other streams from original
                    '-map', '0:a?',            # Copy audio if present, skip if absent
                    '-map', '0:s?',            # Copy subtitles if present, skip if absent
                    '-map', '0:d?',            # Copy data if present, skip if absent
                    '-map', '0:t?',            # Copy timecodes if present, skip if absent
                    # Copy all streams without re-encoding
                    '-c', 'copy',
                    # Add specific flags for QuickTime/MOV metadata handling
                    '-movflags', 'use_metadata_tags+faststart',
                    temp_output_path
                ]
                
                # Run the command to copy metadata
                logger.info("Copying metadata from original to processed video...")
                self._run_ffmpeg_with_progress(
                    metadata_command,
                    1,  # Only one "frame" for this operation
                    "Copying metadata"
                )
                
                # For Apple MOV files, we need additional fixes
                logger.info("Applying exiftool fixes for Apple QuickTime metadata...")
                try:
                    # First copy all metadata from original file to the processed file
                    exiftool_copy_cmd = [
                        'exiftool',
                        '-m',                        # Ignore minor errors
                        '-overwrite_original',       # Overwrite the output file
                        '-api', 'LargeFileSupport=1',  # Support for large video files
                        '-tagsFromFile', input_path, # Copy tags from the input file
                        '-all:all',                  # Copy all tags
                        '-GPS:all',                  # Specifically ensure GPS tags are copied
                        '-Apple:all',                # Copy all Apple-specific tags
                        '-quicktime:all',            # Copy all QuickTime tags
                        temp_output_path             # The output file to modify
                    ]
                    
                    logger.info("Copying metadata from original to processed file...")
                    subprocess.run(exiftool_copy_cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 check=False)
                    
                    # Then apply the specialized fix for Apple QuickTime Keys metadata
                    exiftool_keys_cmd = [
                        'exiftool',
                        '-m',                   # Ignore minor errors
                        '-overwrite_original',  # Overwrite the original file
                        '-api', 'LargeFileSupport=1',  # Support for large video files
                        '-Keys:All=',           # First clear the Keys to avoid conflicts
                        '-tagsFromFile', '@',   # Copy tags from the same file (self-reference)
                        '-Keys:All',            # Then copy back all the Keys (including location)
                        temp_output_path        # Target file to modify
                    ]
                    
                    logger.info("Applying Apple QuickTime Keys metadata fix...")
                    subprocess.run(exiftool_keys_cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 check=False)
                    
                    # Now move the final version to the requested output path
                    shutil.copy2(temp_output_path, output_path)
                    logger.info("Successfully applied metadata fixes with exiftool")
                    
                except Exception as e:
                    logger.warning(f"Error applying exiftool fixes: {e}")
                    # If exiftool fails, just use the ffmpeg output
                    shutil.copy2(temp_output_path, output_path)
            else:
                # For non-MOV files, use the normal approach
                logger.info("Using standard metadata preservation for non-MOV file")
                
                # The standard approach uses -map_metadata 0 to copy metadata
                metadata_command = [
                    'ffmpeg',
                    '-i', input_path,          # Original file with metadata (first input)
                    '-i', temp_processed,      # Processed video (second input)
                    '-map', '1:v',             # Use video from the processed file (second input)
                    '-map_metadata', '0',      # Use metadata from the original file (first input)
                    # Copy audio and other streams from original
                    '-map', '0:a?',            # Copy audio if present, skip if absent
                    '-map', '0:s?',            # Copy subtitles if present, skip if absent
                    '-c', 'copy',              # Copy all streams (no re-encoding)
                    # Add movflags for better compatibility
                    '-movflags', 'use_metadata_tags+faststart',
                    output_path
                ]
                
                # Run the command to copy metadata
                self._run_ffmpeg_with_progress(
                    metadata_command,
                    1,  # Only one "frame" for this operation
                    "Copying metadata"
                )
            
            # Verify location data was preserved
            verify_location_metadata(input_path, output_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Saved corrected video: {output_path}")
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        finally:
            # Clean up temporary directory even if processing is cancelled
            if os.path.exists(temp_dir):
                logger.info("Cleaning up temporary files...")
                shutil.rmtree(temp_dir, ignore_errors=True)

    def process_batch(self, input_folder, output_folder, fps=None, num_workers=None):
        """
        Process all supported files in a folder
        
        Args:
            input_folder: Path to input folder containing images and/or videos
            output_folder: Path to save processed files
            fps: Frames per second for videos (if None, detect from source)
            num_workers: Number of parallel workers (defaults to CPU count)
        """
        start_time = time.time()
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all files in the input folder
        input_files = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Check if file is a supported image or video
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', 
                                       '.mp4', '.avi', '.mov', '.mkv')):
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
                
                file_lower = input_file.lower()
                if file_lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    logger.info(f"Processing image: {input_file}")
                    self.process_image(input_file, output_file)
                    successful += 1
                elif file_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    logger.info(f"Processing video: {input_file}")
                    self.process_video(input_file, output_file, fps=fps, num_workers=num_workers)
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

    def _copy_location_metadata_with_exiftool(self, source_file, target_file):
        """
        Copy location metadata from source to target file using exiftool
        exiftool has better support for GPS metadata than ffmpeg in some cases
        
        Args:
            source_file: Path to the source file with GPS data
            target_file: Path to the target file to copy GPS data to
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if exiftool is available
            try:
                subprocess.run(['exiftool', '-ver'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               check=True)
                has_exiftool = True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("exiftool not found, skipping additional location metadata copy")
                return False
                
            if has_exiftool:
                logger.info("Using exiftool to copy location metadata...")
                
                # First, check if source has GPS data
                gps_info_cmd = ['exiftool', '-s', '-G', '-Location', '-GPS*', source_file]
                result = subprocess.run(gps_info_cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
                
                if not result.stdout.strip():
                    logger.warning(f"No location data found in source file: {source_file}")
                    return False
                
                logger.info(f"Found location data in source: {result.stdout.strip()}")
                
                # For MOV files, we need specific handling for QuickTime metadata structure
                file_ext = os.path.splitext(target_file)[1].lower()
                
                if file_ext in ['.mov', '.mp4']:
                    logger.info("Using QuickTime-specific method for GPS metadata...")
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
                        return False
                    
                    try:
                        # Parse JSON result
                        gps_data = json.loads(extract_result.stdout)[0]
                        
                        # Check if we have GPS data
                        if not ('GPSLatitude' in gps_data and 'GPSLongitude' in gps_data):
                            logger.warning("GPS data extraction returned empty results")
                            return False
                            
                        # Apply GPS data directly to target in the QuickTime format
                        lat = gps_data.get('GPSLatitude', '')
                        lat_ref = gps_data.get('GPSLatitudeRef', 'N')
                        lon = gps_data.get('GPSLongitude', '')
                        lon_ref = gps_data.get('GPSLongitudeRef', 'E')
                        
                        logger.info(f"Found GPS: Lat: {lat} {lat_ref}, Lon: {lon} {lon_ref}")
                        
                        # Convert to standard format if in DMS format
                        lat_formatted = lat
                        lon_formatted = lon
                        
                        # Apply to the target file with various tag formats to ensure compatibility
                        apply_cmd = [
                            'exiftool',
                            '-GPSLatitude=%s' % lat_formatted,
                            '-GPSLatitudeRef=%s' % lat_ref,
                            '-GPSLongitude=%s' % lon_formatted,
                            '-GPSLongitudeRef=%s' % lon_ref,
                            # Add QuickTime specific location tags
                            '-XMP:GPSLatitude=%s' % lat_formatted,
                            '-XMP:GPSLongitude=%s' % lon_formatted,
                            # Add as a human-readable location string
                            f'-LocationCreated="{lat_formatted} {lat_ref}, {lon_formatted} {lon_ref}"',
                            '-LocationTaken=%s' % f"{lat_formatted} {lat_ref}, {lon_formatted} {lon_ref}",
                            '-Location=%s' % f"{lat_formatted} {lat_ref}, {lon_formatted} {lon_ref}",
                            # Update metadata from original
                            '-TagsFromFile', source_file,
                            '-xmp:all',
                            '-iptc:all',
                            '-P',  # Preserve existing tags
                            '-overwrite_original',
                            target_file
                        ]
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse GPS data")
                        return False
                    except Exception as e:
                        logger.warning(f"Error processing GPS data: {e}")
                        return False
                else:
                    # For non-MOV files, use the original approach
                    apply_cmd = [
                        'exiftool', '-TagsFromFile', source_file,
                        '-gps:all',
                        '-location:all',
                        '-coordinates:all',
                        '-xmp:geotag',
                        '-P',
                        '-overwrite_original',
                        target_file
                    ]
                
                # Run the command
                result = subprocess.run(apply_cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
                
                if result.returncode == 0:
                    # Verify location metadata was transferred
                    verify_cmd = ['exiftool', '-s', '-G', '-Location', '-GPS*', target_file]
                    verify_result = subprocess.run(verify_cmd, 
                                                stdout=subprocess.PIPE, 
                                                stderr=subprocess.PIPE,
                                                text=True)
                    
                    if verify_result.stdout.strip():
                        logger.info(f"Successfully verified location metadata: {verify_result.stdout.strip()}")
                        return True
                    else:
                        logger.warning("Location metadata transfer failed verification")
                        return False
                else:
                    logger.warning(f"exiftool error: {result.stderr}")
                    return False
        except Exception as e:
            logger.warning(f"Error using exiftool to copy metadata: {e}")
            return False
        
        return False

    def _add_quicktime_gps_metadata(self, input_path, output_path):
        """
        Special function to add GPS metadata to QuickTime MOV files in a format recognized by macOS
        
        Args:
            input_path: Path to original file with GPS data
            output_path: Path to processed file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if exiftool is available
            try:
                subprocess.run(['exiftool', '-ver'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("exiftool not found, cannot add GPS metadata")
                return False
            
            # Use the suggested command that properly preserves all Keys metadata (crucial for Apple devices)
            logger.info("Using specialized exiftool command for Apple QuickTime metadata...")
            cmd = [
                'exiftool',
                '-m',                   # Ignore minor errors
                '-overwrite_original',  # Overwrite the original file
                '-api', 'LargeFileSupport=1',  # Support for large video files
                '-Keys:All=',           # First clear the Keys to avoid conflicts
                '-tagsFromFile', '@',   # Copy tags from the same file (self-reference)
                '-Keys:All',            # Then copy back all the Keys (including location)
                output_path             # Target file to modify
            ]
            
            # Execute the command
            result = subprocess.run(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
            
            if result.returncode == 0:
                logger.info("Successfully applied Apple QuickTime metadata fix")
                # Return true even if there's a warning, as long as the command was successful
                return True
            else:
                logger.warning(f"exiftool error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in _add_quicktime_gps_metadata: {e}")
            return False

def verify_location_metadata(input_file, output_file):
    """
    Verify that location metadata was preserved by checking both files
    
    Args:
        input_file: Original input file path
        output_file: Processed output file path
        
    Returns:
        bool: True if location metadata was preserved, False otherwise
    """
    logger.info(f"Verifying location metadata preservation...")
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
        file_lower = input_file.lower()
        
        if file_lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            processor.process_image(input_file, output_file)
        elif file_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
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
        # In fast mode, reduce dehaze strength and use lower quality ffmpeg settings
        args.dehaze = min(args.dehaze, 0.3)  # Lower dehaze is faster
    
    # Initialize processor
    processor = UnderwaterImageProcessor(
        dehaze_strength=args.dehaze,
        clahe_clip=args.clahe,
        saturation_boost=args.saturation,
        red_boost=args.red
    )
    
    logger.info(f"Processing parameters: dehaze={args.dehaze}, clahe={args.clahe}, saturation={args.saturation}, red_boost={args.red}")
    
    # Process based on input type
    if os.path.isdir(args.input) or args.batch:
        # Process as batch
        logger.info(f"Processing folder: {args.input}")
        processor.process_batch(args.input, args.output, fps=args.fps, num_workers=args.workers)
    else:
        # Process single file
        input_lower = args.input.lower()
        if input_lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            processor.process_image(args.input, args.output)
        elif input_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            processor.process_video(args.input, args.output, fps=args.fps, num_workers=args.workers)
        else:
            logger.error(f"Unsupported file format: {args.input}")
            return 1
        
    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    main() 