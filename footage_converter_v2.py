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
        
        # Create temporary directory for frames
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
                
            # Rebuild video with ffmpeg, preserving metadata and orientation
            logger.info("Rebuilding video with ffmpeg...")
            temp_output = os.path.join(temp_dir, "temp_output.mp4")
            
            # First create the video with processed frames, maintaining original dimensions
            logger.info(f"Creating video at {width}x{height} resolution...")
            
            encoding_command = [
                'ffmpeg', '-r', str(fps), 
                '-i', os.path.join(temp_dir, "corrected_%04d.jpg"),
                # Use optimized encoding settings
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                # Use multiple threads for encoding
                '-threads', str(num_workers),
                # Add these options to maintain original orientation and size
                '-vf', f'scale={width}:{height}',
                temp_output
            ]
            
            # Use new progress monitoring method for encoding
            self._run_ffmpeg_with_progress(
                encoding_command, 
                completed_frames, 
                "Creating video"
            )
            
            # Then copy the metadata from original to new video
            logger.info("Applying original metadata to the processed video...")
            
            final_command = [
                'ffmpeg', '-i', temp_output, 
                '-i', input_path, 
                '-map', '0:v', '-map_metadata', '1',
                # Copy all streams from input except video
                '-map', '1:a?', '-map', '1:s?', '-map', '1:d?', '-map', '1:t?',
                # Copy streams without re-encoding
                '-c', 'copy',
                output_path
            ]
            
            # This step can also benefit from progress tracking
            self._run_ffmpeg_with_progress(
                final_command,
                1,  # Only one "frame" for metadata copying
                "Finalizing video"
            )
            
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
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('UnderwaterConverter').setLevel(logging.DEBUG)
    
    logger.info("Underwater Footage Converter v2.0")
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