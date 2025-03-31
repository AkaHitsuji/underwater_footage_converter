# Underwater Footage Converter Versions

This directory contains different versions of the underwater footage converter, each with its own improvements and features. Below is a detailed breakdown of each version and their differences.

## Version Overview

### v1 (Initial Version)
- Basic implementation with simple color correction
- Limited to image processing only
- Basic command-line interface
- No metadata preservation
- Simple RGB channel adjustments

### v2 (First Major Update)
- Added video processing support
- Introduced batch processing
- Added CLAHE for contrast enhancement
- Basic dehazing support
- Simple metadata preservation
- Command-line improvements with more options

### v3 (Enhanced Processing)
- Improved color correction algorithm
- Better handling of different underwater conditions
- Enhanced metadata preservation
- Added support for QuickTime formats
- Improved batch processing with progress tracking
- Better error handling and logging

### v4 (Advanced Features)
- Added automatic color parameter optimization
- Enhanced white balance correction
- Improved handling of red-tinted scenes
- Better support for deep blue water scenes
- Enhanced metadata preservation for GPS data
- Added support for high-quality video output
- Improved parallel processing

### v5 (Latest Version)
- Complete rewrite with advanced color analysis
- Scene-specific color optimization
- Enhanced rock detection and correction
- Improved deep blue water handling
- Better green tinge correction
- Advanced metadata preservation
- Enhanced parallel processing
- Support for output format conversion
- Improved progress tracking and error handling

## Changelog

### v1 → v2
- Added video processing capabilities
- Introduced batch processing
- Added CLAHE for contrast enhancement
- Basic metadata preservation
- Enhanced command-line interface

### v2 → v3
- Improved color correction algorithm
- Better underwater condition handling
- Enhanced metadata preservation
- QuickTime format support
- Better progress tracking
- Improved error handling

### v3 → v4
- Added automatic color parameter optimization
- Enhanced white balance correction
- Improved scene-specific corrections
- Better GPS metadata preservation
- High-quality video output support
- Enhanced parallel processing

### v4 → v5
- Complete rewrite with advanced color analysis
- Scene-specific color optimization
- Enhanced rock detection
- Improved deep blue water handling
- Better green tinge correction
- Advanced metadata preservation
- Output format conversion support
- Enhanced progress tracking

## Algorithm Approach

### v1
- Simple RGB channel adjustments
- Basic color correction
- No scene analysis

### v2
- CLAHE for contrast enhancement
- Basic dehazing
- Simple RGB channel adjustments
- Basic metadata copying

### v3
- Enhanced color correction in LAB space
- Improved contrast enhancement
- Better white balance
- Basic scene analysis
- Enhanced metadata preservation

### v4
- Automatic color parameter optimization
- Scene-specific color correction
- Enhanced white balance
- Improved LAB space processing
- Better metadata preservation
- Parallel processing optimization

### v5
- Advanced color analysis using K-means clustering
- Scene-specific color optimization
- Enhanced rock detection using HSV analysis
- Improved deep blue water detection
- Better green tinge correction
- Advanced metadata preservation with exiftool
- Enhanced parallel processing
- Progress tracking with FFmpeg

## Features by Version

### v1
- Image processing only
- Basic color correction
- Simple command-line interface

### v2
- Image and video processing
- Batch processing
- CLAHE contrast enhancement
- Basic dehazing
- Simple metadata preservation
- Enhanced command-line options

### v3
- Enhanced color correction
- Better underwater condition handling
- QuickTime format support
- Improved metadata preservation
- Better progress tracking
- Enhanced error handling

### v4
- Automatic color parameter optimization
- Scene-specific corrections
- Enhanced white balance
- High-quality video output
- Better GPS metadata preservation
- Improved parallel processing

### v5
- Advanced color analysis
- Scene-specific optimization
- Enhanced rock detection
- Improved deep blue water handling
- Better green tinge correction
- Advanced metadata preservation
- Output format conversion
- Enhanced progress tracking
- Improved error handling
- Better parallel processing

## Usage

Each version can be used with the following basic command:

```bash
python footage_converter_vX.py input_file -o output_file
```

For batch processing:
```bash
python footage_converter_vX.py input_folder -o output_folder --batch
```

For more options and features, refer to the help message:
```bash
python footage_converter_vX.py --help
```

Note: v5 is the latest version and includes all features from previous versions plus significant improvements in color correction and processing quality. 