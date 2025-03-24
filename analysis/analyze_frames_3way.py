import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

def analyze_frame_triplet(original_path, script_path, reference_path, frame_num):
    """Analyze a triplet of images: original, script output, and reference output"""
    # Read images
    original = cv2.imread(original_path)  # Original
    script = cv2.imread(script_path)      # Script output (v2)
    reference = cv2.imread(reference_path)  # Reference (target)
    
    # Convert BGR to RGB for visualization
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    script_rgb = cv2.cvtColor(script, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for color analysis
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    script_hsv = cv2.cvtColor(script, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    
    # Calculate average color values (BGR)
    original_avg = [np.mean(original[:,:,0]), np.mean(original[:,:,1]), np.mean(original[:,:,2])]
    script_avg = [np.mean(script[:,:,0]), np.mean(script[:,:,1]), np.mean(script[:,:,2])]
    reference_avg = [np.mean(reference[:,:,0]), np.mean(reference[:,:,1]), np.mean(reference[:,:,2])]
    
    # Calculate histogram differences
    h1_hist = cv2.calcHist([original_hsv], [0], None, [180], [0, 180])
    s1_hist = cv2.calcHist([original_hsv], [1], None, [256], [0, 256])
    v1_hist = cv2.calcHist([original_hsv], [2], None, [256], [0, 256])
    
    h2_hist = cv2.calcHist([script_hsv], [0], None, [180], [0, 180])
    s2_hist = cv2.calcHist([script_hsv], [1], None, [256], [0, 256])
    v2_hist = cv2.calcHist([script_hsv], [2], None, [256], [0, 256])
    
    h3_hist = cv2.calcHist([reference_hsv], [0], None, [180], [0, 180])
    s3_hist = cv2.calcHist([reference_hsv], [1], None, [256], [0, 256])
    v3_hist = cv2.calcHist([reference_hsv], [2], None, [256], [0, 256])
    
    # Calculate color difference factors between script and reference
    # These are the changes needed for our script to match the reference
    blue_diff = reference_avg[0] / max(script_avg[0], 1)
    green_diff = reference_avg[1] / max(script_avg[1], 1)
    red_diff = reference_avg[2] / max(script_avg[2], 1)
    
    # Calculate reference-to-original ratio
    # These show how the reference is improving the original
    ref_blue_factor = reference_avg[0] / max(original_avg[0], 1)
    ref_green_factor = reference_avg[1] / max(original_avg[1], 1)
    ref_red_factor = reference_avg[2] / max(original_avg[2], 1)
    
    # Calculate script-to-original ratio
    # These show how our script is currently improving the original
    script_blue_factor = script_avg[0] / max(original_avg[0], 1)
    script_green_factor = script_avg[1] / max(original_avg[1], 1)
    script_red_factor = script_avg[2] / max(original_avg[2], 1)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f'Frame {frame_num} Comparison', fontsize=16)
    
    # Display images side by side
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(original_rgb)
    ax1.set_title('Original (IMG_7671)')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(script_rgb)
    ax2.set_title('Script Output (IMG_7671_converted)')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(reference_rgb)
    ax3.set_title('Reference (dp_color_video)')
    ax3.axis('off')
    
    # Display color histograms
    # Hue
    ax4 = plt.subplot(3, 4, 5)
    ax4.plot(h1_hist, color='b', label='Original')
    ax4.plot(h2_hist, color='r', label='Script')
    ax4.plot(h3_hist, color='g', label='Reference')
    ax4.set_title('Hue Histograms')
    ax4.legend()
    
    # Saturation
    ax5 = plt.subplot(3, 4, 6)
    ax5.plot(s1_hist, color='b', label='Original')
    ax5.plot(s2_hist, color='r', label='Script')
    ax5.plot(s3_hist, color='g', label='Reference')
    ax5.set_title('Saturation Histograms')
    ax5.legend()
    
    # Value
    ax6 = plt.subplot(3, 4, 7)
    ax6.plot(v1_hist, color='b', label='Original')
    ax6.plot(v2_hist, color='r', label='Script')
    ax6.plot(v3_hist, color='g', label='Reference')
    ax6.set_title('Value Histograms')
    ax6.legend()
    
    # Display average BGR values
    ax7 = plt.subplot(3, 4, 9)
    ax7.bar(['Blue', 'Green', 'Red'], original_avg, alpha=0.5, label='Original')
    ax7.bar(['Blue', 'Green', 'Red'], script_avg, alpha=0.5, label='Script')
    ax7.bar(['Blue', 'Green', 'Red'], reference_avg, alpha=0.5, label='Reference')
    ax7.set_title('Average BGR Values')
    ax7.legend()
    
    # Display reference vs script ratios
    ax8 = plt.subplot(3, 4, 10)
    ax8.bar(['Blue Ratio', 'Green Ratio', 'Red Ratio'], [blue_diff, green_diff, red_diff])
    ax8.axhline(y=1.0, color='r', linestyle='--')
    ax8.set_title('Color Ratios (Reference/Script)')
    
    # Display improvement factors - Reference vs Original
    ax9 = plt.subplot(3, 4, 11)
    ax9.bar(['Blue', 'Green', 'Red'], [ref_blue_factor, ref_green_factor, ref_red_factor])
    ax9.axhline(y=1.0, color='r', linestyle='--')
    ax9.set_title('Reference Enhancement (Ref/Orig)')
    
    # Display improvement factors - Script vs Original
    ax10 = plt.subplot(3, 4, 12)
    ax10.bar(['Blue', 'Green', 'Red'], [script_blue_factor, script_green_factor, script_red_factor])
    ax10.axhline(y=1.0, color='r', linestyle='--')
    ax10.set_title('Script Enhancement (Script/Orig)')
    
    # Display statistics and conclusions
    stats_text = f"""
    Average BGR Values:
    Original: {[round(v, 2) for v in original_avg]}
    Script:   {[round(v, 2) for v in script_avg]}
    Reference: {[round(v, 2) for v in reference_avg]}
    
    Script Adjustment Needed (Reference/Script):
    Blue: {blue_diff:.2f}
    Green: {green_diff:.2f}
    Red: {red_diff:.2f}
    
    Reference Enhancement (Reference/Original):
    Blue: {ref_blue_factor:.2f}
    Green: {ref_green_factor:.2f}
    Red: {ref_red_factor:.2f}
    
    Current Script Enhancement (Script/Original):
    Blue: {script_blue_factor:.2f}
    Green: {script_green_factor:.2f}
    Red: {script_red_factor:.2f}
    """
    
    plt.figtext(0.02, 0.3, stats_text, wrap=True, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Calculate the overall difference magnitude
    overall_diff = np.sqrt(
        (blue_diff - 1.0)**2 + 
        (green_diff - 1.0)**2 + 
        (red_diff - 1.0)**2
    )
    
    return fig, {
        'blue_diff': blue_diff,
        'green_diff': green_diff,
        'red_diff': red_diff,
        'ref_blue_factor': ref_blue_factor,
        'ref_green_factor': ref_green_factor,
        'ref_red_factor': ref_red_factor,
        'script_blue_factor': script_blue_factor,
        'script_green_factor': script_green_factor,
        'script_red_factor': script_red_factor,
        'original_avg': original_avg,
        'script_avg': script_avg,
        'reference_avg': reference_avg,
        'overall_diff': overall_diff
    }

# Set the path to the comparison_videos directory
frames_dir = 'comparison_videos'

# Analyze multiple frames
frames_to_analyze = [1, 5, 10, 15, 20, 25, 30, 35]
results = []

with PdfPages('frame_analysis_3way.pdf') as pdf:
    for frame_num in frames_to_analyze:
        original_path = os.path.join(frames_dir, f'original_{frame_num:04d}.jpg')
        script_path = os.path.join(frames_dir, f'img1_{frame_num:04d}.jpg')
        reference_path = os.path.join(frames_dir, f'img2_{frame_num:04d}.jpg')
        
        if os.path.exists(original_path) and os.path.exists(script_path) and os.path.exists(reference_path):
            print(f'Analyzing frame {frame_num}...')
            fig, metrics = analyze_frame_triplet(original_path, script_path, reference_path, frame_num)
            results.append(metrics)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print(f'Frame {frame_num} not found, skipping')

# Calculate average adjustments across all frames
if results:
    avg_blue_diff = np.mean([result['blue_diff'] for result in results])
    avg_green_diff = np.mean([result['green_diff'] for result in results])
    avg_red_diff = np.mean([result['red_diff'] for result in results])
    
    avg_ref_blue_factor = np.mean([result['ref_blue_factor'] for result in results])
    avg_ref_green_factor = np.mean([result['ref_green_factor'] for result in results])
    avg_ref_red_factor = np.mean([result['ref_red_factor'] for result in results])
    
    avg_script_blue_factor = np.mean([result['script_blue_factor'] for result in results])
    avg_script_green_factor = np.mean([result['script_green_factor'] for result in results])
    avg_script_red_factor = np.mean([result['script_red_factor'] for result in results])
    
    avg_overall_diff = np.mean([result['overall_diff'] for result in results])
    
    # Calculate ideal absolute factors to apply to original
    ideal_blue_boost = avg_ref_blue_factor
    ideal_green_boost = avg_ref_green_factor
    ideal_red_boost = avg_ref_red_factor
    
    # Get average BGR levels
    avg_orig_bgr = np.mean([result['original_avg'] for result in results], axis=0)
    avg_script_bgr = np.mean([result['script_avg'] for result in results], axis=0)
    avg_ref_bgr = np.mean([result['reference_avg'] for result in results], axis=0)
    
    # Report summary
    summary = f"""
    Frame Analysis Summary (3-way):
    
    Average BGR Values:
    Original: {[round(v, 2) for v in avg_orig_bgr]}
    Script:   {[round(v, 2) for v in avg_script_bgr]}
    Reference: {[round(v, 2) for v in avg_ref_bgr]}
    
    Average Direct Adjustment Needed (Reference/Script):
    Blue: {avg_blue_diff:.2f}
    Green: {avg_green_diff:.2f}
    Red: {avg_red_diff:.2f}
    Overall difference magnitude: {avg_overall_diff:.2f}
    
    Reference Enhancement Factors (Reference/Original):
    Blue: {avg_ref_blue_factor:.2f}
    Green: {avg_ref_green_factor:.2f}
    Red: {avg_ref_red_factor:.2f}
    
    Current Script Enhancement Factors (Script/Original):
    Blue: {avg_script_blue_factor:.2f}
    Green: {avg_script_green_factor:.2f}
    Red: {avg_script_red_factor:.2f}
    
    Suggested Absolute Script Improvements:
    1. Set blue_boost parameter to: {ideal_blue_boost:.2f}
    2. Set green_boost parameter to: {ideal_green_boost:.2f}
    3. Set red_boost parameter to: {ideal_red_boost:.2f}
    
    Notes: 
    - The rocks appear too red in the script output compared to the reference.
    - The reference has gentler, more balanced colors.
    - Ideal parameters should be closer to the Reference/Original ratios.
    """
    
    print(summary)
    
    # Save summary to file
    with open('analysis_summary_3way.txt', 'w') as f:
        f.write(summary)
    
    print('Analysis completed. Check frame_analysis_3way.pdf for detailed visualizations.')
else:
    print("No frames were successfully analyzed. Please check the file paths.") 