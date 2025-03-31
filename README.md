# Underwater Footage Converter

A Python tool for enhancing underwater video and image footage while preserving important metadata including GPS location data.

## TODO
- [ ] Convert to rust project
- [ ] Package in dmg/exe
- [ ] Create desktop app UI

## Features

- Automatic color parameter optimization for underwater scenes
- Intelligent scene analysis for optimal color restoration
- Enhanced white balance correction
- Adaptive contrast enhancement using CLAHE algorithm
- Smart saturation and vibrance adjustment
- Red tint detection and correction
- Deep blue water scene optimization
- Sharpness enhancement option
- Multi-threaded processing for faster conversion
- Support for both images (JPG, PNG, TIFF) and videos (MP4, MOV, AVI, MKV)
- Comprehensive metadata preservation including GPS location data
- Progress monitoring with FFmpeg integration
- Graceful termination handling

## Metadata Preservation

The converter implements multiple approaches to preserve metadata from the original files, with special attention to GPS location data in Apple QuickTime MOV files:

1. **FFmpeg Metadata Mapping**: Uses FFmpeg's `-map_metadata` feature to copy metadata from the original file to the processed file without re-encoding the streams.

2. **Apple QuickTime Keys Preservation**: For MOV files from iOS devices, we use a specialized exiftool command that preserves the critical 'Keys' metadata atom, which is required for macOS to properly recognize location data:
   ```
   exiftool -m -overwrite_original -api LargeFileSupport=1 -Keys:All= -tagsFromFile @ -Keys:All output.mov
   ```

3. **Format-Specific Handling**: MOV files get special treatment to maximize compatibility with macOS and iOS.

### Location Data Working Properly

Unlike many other conversion tools, this converter successfully preserves location metadata in a way that:

- macOS Spotlight (`mdls`) correctly shows latitude and longitude
- Location is visible in macOS Finder's "Get Info" panel
- Photos and other media applications can read the geographic data
- Original creation dates and other metadata is retained

This is particularly important for underwater footage from iPhones and other iOS devices, where location data helps organize and map dive locations.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- FFmpeg
- Exiftool
- PIL/Pillow
- tqdm
- scikit-learn (for color analysis)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/underwater-footage-converter.git
   cd underwater-footage-converter
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have FFmpeg and Exiftool installed on your system:
   - **macOS**: `brew install ffmpeg exiftool`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg libimage-exiftool-perl`
   - **Windows**: Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and Exiftool from [exiftool.org](https://exiftool.org/)

## Usage

Basic usage:

```bash
python footage_converter.py input_file.mov -o output_file.mov
```

Process a folder of files:

```bash
python footage_converter.py input_folder/ --output output_folder/ --batch
```

Adjust processing parameters:

```bash
python footage_converter.py input.mov -o output.mov --contrast 2.5 --saturation 0.8 --brightness 1.15 --white-balance 0.8 --auto-tune 0.9 --sharpness 0.3
```

## Options

- `--output`, `-o`: Path to save the processed file or folder
- `--contrast`, `-c`: CLAHE clip limit for contrast enhancement (default: 2.0)
- `--saturation`, `-s`: Saturation factor (0-1 range, default: 0.9)
- `--brightness`, `-B`: Brightness boost factor (default: 1.12)
- `--white-balance`, `-wb`: White balance strength (0-1 range, default: 0.7)
- `--auto-tune`, `-a`: Auto-tuning strength (0-1 range, default: 0.9)
- `--sharpness`, `-sh`: Sharpness enhancement factor (0-1 range, default: 0.3)
- `--fps`: Output frames per second (default: same as input)
- `--workers`, `-w`: Number of worker threads (default: CPU count)
- `--batch`, `-b`: Process input as a folder containing multiple files
- `--verbose`, `-v`: Enable verbose logging
- `--test-location`, `-t`: Run test for location metadata preservation
- `--quality`, `-q`: Video quality (CRF value: 18-28 range, lower = better quality, higher = smaller size, default: 24)
- `--high-quality`, `-hq`: Use high quality preset (CRF 20, slow preset) for best quality output
- `--output-format`, `-f`: Convert output videos to the specified format (currently only supports mp4)

## Features in Detail

### Automatic Color Optimization

The converter uses advanced image analysis to automatically determine optimal color parameters for each scene:

- Detects red tinting in underwater scenes
- Identifies deep blue water conditions
- Recognizes red-tinted rocks and neutral objects
- Adjusts color balance based on scene type
- Optimizes contrast and saturation automatically

### Metadata Preservation

The tool implements a comprehensive metadata preservation strategy:

1. **FFmpeg-based Preservation**: Uses FFmpeg's metadata mapping to preserve all original metadata
2. **ExifTool Integration**: For detailed metadata handling, especially GPS data
3. **QuickTime-Specific Handling**: Special handling for MOV/MP4 files to ensure compatibility
4. **Location Data Verification**: Built-in verification of GPS data preservation

### Performance Features

- Multi-threaded frame processing for faster video conversion
- Progress monitoring with FFmpeg integration
- Graceful termination handling (Ctrl+C support)
- Memory-efficient processing of large files
- Support for high-quality video output

## Acknowledgments

This project uses several open-source libraries and tools:
- FFmpeg for video processing
- ExifTool for metadata handling
- OpenCV for image processing
- scikit-learn for color analysis
- Python and its ecosystem 