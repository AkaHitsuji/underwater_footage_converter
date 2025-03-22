import cv2
import numpy as np
import os
import subprocess
import argparse

def restore_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    avg_b = np.mean(restored[:, :, 0])
    avg_g = np.mean(restored[:, :, 1])
    avg_r = np.mean(restored[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_all / avg_b
    scale_g = avg_all / avg_g
    scale_r = avg_all / avg_r
    balanced = np.clip(restored * [scale_b, scale_g, scale_r], 0, 255).astype(np.uint8)
    return balanced

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error loading image: {input_path}")
        return
    corrected = restore_color(image)
    cv2.imwrite(output_path, corrected)
    print(f"Saved corrected image: {output_path}")

def process_video(input_path, output_path):
    temp_frames = "temp_frames"
    os.makedirs(temp_frames, exist_ok=True)
    
    # Use subprocess instead of os.system for better error handling
    subprocess.run(['ffmpeg', '-i', input_path, '-qscale:v', '2', f'{temp_frames}/frame_%04d.jpg'], check=True)
    
    for file in sorted(os.listdir(temp_frames)):
        if file.startswith("corrected_"):
            continue
        frame_path = os.path.join(temp_frames, file)
        output_frame_path = os.path.join(temp_frames, "corrected_" + file)
        process_image(frame_path, output_frame_path)
    
    # Use subprocess for ffmpeg
    subprocess.run([
        'ffmpeg', '-r', '30', 
        '-i', f'{temp_frames}/corrected_frame_%04d.jpg', 
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', 
        output_path
    ], check=True)
    
    # Clean up temp files
    for file in os.listdir(temp_frames):
        os.remove(os.path.join(temp_frames, file))
    os.rmdir(temp_frames)
    print(f"Saved corrected video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Restore underwater images and videos by correcting color balance.")
    parser.add_argument("input", help="Path to the input image or video file.")
    parser.add_argument("output", nargs='?', help="Path to save the corrected output file. If not provided, defaults to '<original_file_name>_converted.<original_filetype>'.")
    args = parser.parse_args()
    
    if not args.output:
        file_name, file_ext = os.path.splitext(args.input)
        args.output = f"{file_name}_converted{file_ext}"
    
    if args.input.endswith(('.jpg', '.png', '.jpeg')):
        process_image(args.input, args.output)
    elif args.input.endswith(('.mp4', '.avi', '.mov', '.MOV')):
        process_video(args.input, args.output)
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    main()