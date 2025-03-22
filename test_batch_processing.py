import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

def run_batch_test():
    """
    Test the batch processing functionality by creating a test directory structure,
    copying sample files into it, and running the converter in batch mode.
    """
    # Setup test directories
    test_dir = "test_batch"
    test_output_dir = f"{test_dir}_converted"
    
    # Remove existing test directories if they exist
    for dir_path in [test_dir, test_output_dir]:
        if os.path.exists(dir_path):
            print(f"Cleaning up existing directory: {dir_path}")
            shutil.rmtree(dir_path)
    
    # Create test directory structure
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "subfolder"), exist_ok=True)
    
    # Check for sample files in test_images and test_videos directories
    found_samples = False
    
    # Look for sample underwater images
    image_samples = []
    if os.path.exists("test_images"):
        for file in os.listdir("test_images"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_samples.append(os.path.join("test_images", file))
    
    # Look for sample underwater videos
    video_samples = []
    if os.path.exists("test_videos"):
        for file in os.listdir("test_videos"):
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_samples.append(os.path.join("test_videos", file))
    
    # Copy sample files to test directory
    if image_samples:
        print(f"Found {len(image_samples)} sample images")
        for i, sample in enumerate(image_samples):
            # Copy some to main folder, some to subfolder
            dest = os.path.join(test_dir, os.path.basename(sample)) if i % 2 == 0 else \
                  os.path.join(test_dir, "subfolder", os.path.basename(sample))
            shutil.copy2(sample, dest)
            found_samples = True
    
    if video_samples:
        print(f"Found {len(video_samples)} sample videos")
        for i, sample in enumerate(video_samples):
            # Copy some to main folder, some to subfolder
            dest = os.path.join(test_dir, os.path.basename(sample)) if i % 2 == 0 else \
                  os.path.join(test_dir, "subfolder", os.path.basename(sample))
            shutil.copy2(sample, dest)
            found_samples = True
    
    if not found_samples:
        print("No sample images or videos found.")
        print("Please add sample files to test_images or test_videos directories.")
        print("  - test_images/sample_underwater.jpg")
        print("  - test_videos/sample_underwater.mp4")
        return
    
    # Run the converter in batch mode with the test directory
    print("\nRunning underwater footage converter in batch mode...")
    cmd = [sys.executable, "footage_converter_v2.py", test_dir, "--fast"]
    
    # Parse any additional arguments passed to this test script
    parser = argparse.ArgumentParser(description="Test batch processing functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args, unknown = parser.parse_known_args()
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run the converter
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running converter: {e}")
        return
    
    # Verify output directory was created
    if not os.path.exists(test_output_dir):
        print(f"Test failed: Output directory '{test_output_dir}' was not created")
        return
    
    # Count processed files
    processed_files = []
    for root, _, files in os.walk(test_output_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')):
                processed_files.append(os.path.join(root, file))
    
    # Compare with input files
    input_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')):
                input_files.append(os.path.join(root, file))
    
    print(f"\nInput files: {len(input_files)}")
    print(f"Processed files: {len(processed_files)}")
    
    # Print each file path for easier debugging
    print("\nInput files:")
    for f in input_files:
        print(f"  - {f}")
    
    print("\nProcessed files:")
    for f in processed_files:
        print(f"  - {f}")
    
    # Verify that the number of processed files matches input files
    if len(processed_files) == len(input_files):
        print("\nTest successful! All files were processed.")
        
        # Check directory structure was preserved
        structure_preserved = True
        for input_file in input_files:
            rel_path = os.path.relpath(input_file, test_dir)
            output_file = os.path.join(test_output_dir, rel_path)
            if not os.path.exists(output_file):
                print(f"Directory structure issue: {output_file} not found")
                structure_preserved = False
        
        if structure_preserved:
            print("Directory structure was correctly preserved in the output.")
        else:
            print("Test partially failed: Directory structure was not preserved correctly.")
    else:
        print(f"\nTest failed: Number of processed files ({len(processed_files)}) doesn't match input files ({len(input_files)})")
    
    print("\nTest completed. You can examine the test directories:")
    print(f"  - Input: {test_dir}")
    print(f"  - Output: {test_output_dir}")

if __name__ == "__main__":
    run_batch_test() 