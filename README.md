# People Icon Remover

A web application that automatically detects and removes people icons from images, particularly useful for processing mobile app screenshots and interface mockups.

## Features

- **Batch Upload**: Upload multiple images at once
- **Automatic Detection**: Uses computer vision to detect people icons
- **Smart Removal**: Intelligently removes icons while preserving background
- **Batch Download**: Download all processed images as a ZIP file
- **Web Interface**: User-friendly drag-and-drop interface
- **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, BMP, TIFF

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Build a Windows Executable

You can package the application into a standalone Windows executable using PyInstaller:

1. Ensure the dependencies (including PyInstaller) are installed:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. On Windows, run the build script from this repository:
   ```cmd
   build_windows.bat
   ```

3. The packaged executable will be created at `dist\ShufenTools.exe`. Launching it will start the Flask server bundled inside the binary.

## How It Works

The application uses two main detection methods:

1. **Circle Detection**: Uses HoughCircles to detect circular/rounded people icons
2. **Contour Analysis**: Analyzes shapes and sizes to identify icon-like patterns

The detection focuses on the right side of images where people icons are typically located in mobile interfaces.

## Usage

1. **Upload Images**:
   - Click the upload area or drag and drop images
   - Select multiple files for batch processing

2. **Process**:
   - Click "Process Images" to start automatic icon removal
   - Wait for processing to complete

3. **Download**:
   - Download individual processed images
   - Or download all as a ZIP file

4. **Clear**:
   - Use "Clear All" to remove all uploaded and processed files

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and process images
- `GET /download/<filename>` - Download individual processed image
- `GET /download_all` - Download all processed images as ZIP
- `GET /clear` - Clear all files

## Configuration

- Maximum file size: 50MB total
- Supported image formats: PNG, JPG, JPEG, GIF, BMP, TIFF
- Processing folders: `uploads/` and `processed/`

## Technical Details

The image processing algorithm:

1. Converts images to grayscale for analysis
2. Focuses on the rightmost 20% of the image
3. Uses HoughCircles to detect circular shapes
4. Analyzes contours for icon-like patterns
5. Samples background colors for seamless removal
6. Applies smart filling to remove detected icons
# ShufenTools
