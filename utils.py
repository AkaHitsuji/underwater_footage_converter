import os
import subprocess
import json
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('UnderwaterConverter')

# Constants for file types
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
QUICKTIME_EXTENSIONS = ('.mov', '.mp4')

def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image format"""
    return file_path.lower().endswith(IMAGE_EXTENSIONS)

def is_video_file(file_path: str) -> bool:
    """Check if a file is a supported video format"""
    return file_path.lower().endswith(VIDEO_EXTENSIONS)

def is_quicktime_format(file_path: str) -> bool:
    """Check if a file is in a QuickTime format (MOV/MP4)"""
    return file_path.lower().endswith(QUICKTIME_EXTENSIONS)

def check_exiftool_available() -> bool:
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

def verify_location_metadata(input_file: str, output_file: str) -> bool:
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

def get_exiftool_cmd_for_metadata_copy(source_file: str, target_file: str) -> Optional[List[str]]:
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

def fix_quicktime_metadata(file_path: str) -> bool:
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

def copy_all_metadata_with_exiftool(source_file: str, target_file: str) -> bool:
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

def test_location_preservation(input_file: str) -> bool:
    """
    Run test for location metadata preservation
    
    Args:
        input_file: Optional specific file to test. If not provided, a sample file will be used.
                  
    Returns:
        bool: True if test passes, False otherwise
    """
    import tempfile
    
    if not input_file:
        logger.error("No test file provided.")
        return False
        
    logger.info(f"Running location metadata preservation test with: {input_file}")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_file)[1], delete=False) as temp_file:
        output_file = temp_file.name
        
    try:
        # Process the file
        from footage_converter import AutoColorUnderwaterImageProcessor
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