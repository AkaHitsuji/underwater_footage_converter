# Underwater Footage Converter

A Python tool for enhancing underwater video and image footage while preserving important metadata including GPS location data.

## TODO
- [ ] Convert to generic script that can do multiple other things like compression
- [ ] Consider chrome extension or webpage

## Features

- Color correction optimized for underwater footage
- Dehaze effect to reduce water turbidity
- Contrast enhancement using CLAHE algorithm 
- Red channel boosting to restore colors lost underwater
- Saturation enhancement for vivid results
- Preserves original metadata including GPS location data
- Multi-threaded processing for faster conversion
- Support for both images (JPG, PNG, TIFF) and videos (MP4, MOV, AVI, MKV)

## Metadata Preservation

The converter implements multiple approaches to preserve metadata from the original files, with special attention to GPS location data in Apple QuickTime MOV files:

1. **FFmpeg Metadata Mapping**: Uses FFmpeg's `-map_metadata` feature to copy metadata from the original file to the processed file without re-encoding the streams.

2. **Apple QuickTime Keys Preservation**: For MOV files from iOS devices, we use a specialized exiftool command that preserves the critical 'Keys' metadata atom, which is required for macOS to properly recognize location data:
   ```
   exiftool -m -overwrite_original -api LargeFileSupport=1 -Keys:All= -tagsFromFile @ -Keys:All output.mov
   ```
   This specific approach successfully preserves location metadata in a way that macOS Finder, Photos, and Spotlight can recognize.

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
python footage_converter_v2.py input_file.mov -o output_file.mov
```

Process a folder of files:

```bash
python footage_converter_v2.py input_folder/ --output output_folder/ --batch
```

Adjust processing parameters:

```bash
python footage_converter_v2.py input.mov -o output.mov --dehaze 0.7 --clahe 2.5 --saturation 1.8 --red 1.1
```

## Options

- `--output`, `-o`: Path to save the processed file or folder
- `--dehaze`, `-d`: Strength of dehazing effect (0.0-1.0)
- `--clahe`, `-c`: CLAHE clip limit for contrast enhancement
- `--saturation`, `-s`: Saturation boost factor
- `--red`, `-r`: Red channel boost factor (1.0 = neutral)
- `--fps`: Output frames per second (default: same as input)
- `--workers`, `-w`: Number of worker threads (default: CPU count)
- `--fast`, `-f`: Use faster processing with slightly lower quality
- `--batch`, `-b`: Process input as a folder containing multiple files
- `--verbose`, `-v`: Enable verbose logging

## Acknowledgments

This project uses several open-source libraries and tools:
- FFmpeg for video processing
- ExifTool for metadata handling
- OpenCV for image processing
- Python and its ecosystem 