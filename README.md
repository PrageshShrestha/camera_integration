# Camera Tracking with 3D Reconstruction

A web-based photogrammetry application that captures photos and generates 3D models using Meshroom.

## Features

- **Web-based Camera Capture**: Real-time camera interface using WebRTC
- **File Upload Support**: Upload existing photos from your device
- **Image Processing**: Advanced preprocessing with denoising and contrast enhancement
- **3D Reconstruction**: Meshroom pipeline for high-quality mesh generation
- **Model Visualization**: Three.js-based interactive 3D model viewer
- **Project Management**: Meshroom-compatible project structure and cache management

## Architecture

### Backend (FastAPI)
- `main.py` - Main application with Meshroom integration
- Uses subprocess to execute Meshroom pipeline
- OpenCV for advanced image preprocessing
- Comprehensive error handling and validation

### Frontend (HTML/JavaScript)
- `templates/index.html` - Camera capture and file upload interface
- `templates/view_3d_model.html` - Interactive 3D model viewer
- WebRTC for camera access
- Three.js for 3D visualization

### Project Structure
```
camera_tracking/
├── main.py              # FastAPI application
├── requirements.txt       # Python dependencies
├── templates/           # HTML templates
│   ├── index.html         # Camera capture interface
│   └── view_3d_model.html # 3D model viewer
├── static/                # Static files for web serving
├── meshroom_project/      # Meshroom project directory (auto-created)
│   ├── images/           # Input images
│   ├── output/           # Generated 3D models
│   ├── cache/            # Meshroom cache
│   └── preprocessed/     # Preprocessed images
└── tracking/             # Virtual environment
```

## API Endpoints

- `GET /` - Main camera capture interface
- `POST /upload` - Upload photos (camera capture or file upload)
- `POST /generate-3d-model` - Process images and generate 3D model
- `GET /view-3d-model` - View generated 3D model
- `GET /project-status` - Get current project status
- `POST /reset-project` - Reset entire project

## Meshroom Pipeline

The application implements a complete Meshroom pipeline:

1. **Image Preprocessing**: Denoising and contrast enhancement
2. **Feature Detection**: Automatic feature extraction (Meshroom handles)
3. **Structure from Motion**: Camera pose estimation
4. **Dense Reconstruction**: Point cloud generation
5. **Mesh Generation**: Surface reconstruction and texturing

## Key Improvements

- **Dual Input Methods**: Camera capture and file upload options
- **Enhanced Image Preprocessing**: Denoising and contrast enhancement for better reconstruction
- **Robust Error Handling**: Comprehensive validation and error recovery
- **Performance Monitoring**: Progress tracking and timing information
- **Meshroom Integration**: High-quality 3D mesh generation
- **Better User Experience**: Loading states and detailed error messages

## Requirements

### System Requirements
- Python 3.8+
- Meshroom (installed separately)
- Camera access (for capture mode)

### Python Dependencies
See `requirements.txt` for the complete list of dependencies.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd camera_tracking
   ```

2. **Install Meshroom**:
   ```bash
   # macOS
   brew install meshroom
   
   # Or download from https://github.com/alicevision/meshroom
   ```

3. **Set up Python environment**:
   ```bash
   python -m venv tracking
   source tracking/bin/activate  # On Windows: tracking\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application**:
   Open http://127.0.0.1:8000 in your browser

## Usage

1. **Choose Input Method**: Select camera capture or file upload
2. **Capture/Upload Photos**: Take photos with camera or upload existing images
3. **Generate 3D Model**: Click "Generate 3D Model" button
4. **View Results**: Interactive 3D model viewer with rotation/zoom controls

## Photo Guidelines for Best Results

- **Overlap**: 60-80% overlap between consecutive photos
- **Features**: Capture distinctive textures and objects
- **Lighting**: Consistent, good lighting conditions
- **Quantity**: Minimum 2 photos, 8-12 recommended for quality results
- **Angles**: Multiple viewpoints around the subject

## License

This project uses Meshroom under its respective license and follows Meshroom best practices for 3D reconstruction.
