# Underwater Footage Converter

A tool for restoring natural colors in underwater images and videos by compensating for spectral absorption and blue-green color shift.

## Features

- Processes both images and videos
- Restores natural colors by compensating for underwater light absorption
- Applies dehazing to reduce water turbidity effects
- Enhances contrast and clarity
- Boosts lost red tones in underwater footage
- Preserves original metadata (EXIF, creation date, etc.)
- Simple command-line interface

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- ffmpeg (installed on your system)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/underwater_footage_converter.git
   cd underwater_footage_converter
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have ffmpeg installed on your system:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Usage

### Basic Usage

Process an image or video:

```bash
python footage_converter_v2.py input_file.jpg
```

The processed file will be saved as `input_file_converted.jpg` in the same directory.

To specify an output file:

```bash
python footage_converter_v2.py input_file.mp4 --output corrected_file.mp4
```

### Advanced Options

The v2 script provides additional processing options:

```bash
python footage_converter_v2.py input_file.mp4 \
  --dehaze 0.7 \    # Strength of dehazing (0.0-1.0)
  --clahe 2.5 \     # CLAHE clip limit for contrast
  --saturation 1.8  # Saturation boost factor
```

For full help:

```bash
python footage_converter_v2.py --help
```

## Version Comparison

### v1 (Basic)
- Simple color restoration using CLAHE and white balancing
- Basic video processing

### v2 (Advanced)
- Improved color restoration algorithms
- Dehazing for reducing turbidity
- Metadata preservation
- Progress bar for video processing
- More customization options
- OOP design for better extensibility

## Examples

Before and after processing:

[Insert before/after images]

## License

[Your license information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 