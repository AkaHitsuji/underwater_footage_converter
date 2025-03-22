import os
import sys
from footage_converter_v1 import process_image

def run_test():
    # Create a test directory if it doesn't exist
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Check if test image exists, else tell user what to do
    sample_image = os.path.join(test_dir, "sample_underwater.jpg")
    output_image = os.path.join(test_dir, "sample_underwater_converted.jpg")
    
    if not os.path.exists(sample_image):
        print(f"Test image not found at {sample_image}")
        print("Please place a test underwater image at this location to run the test")
        return
    
    # Process the test image
    print(f"Processing test image: {sample_image}")
    process_image(sample_image, output_image)
    
    if os.path.exists(output_image):
        print(f"Test successful! Output image created at {output_image}")
    else:
        print("Test failed: Output image was not created")

if __name__ == "__main__":
    run_test() 