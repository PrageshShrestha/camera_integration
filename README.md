# Camera Tracking with 3D Reconstruction

A web-based photogrammetry application that captures photos with device orientation data and generates 3D models using COLMAP.

## Features

- **Web-based Camera Capture**: Real-time camera interface using WebRTC
- **Device Orientation Tracking**: Captures gyroscope data with each photo
- **Image Processing**: Advanced preprocessing with denoising and contrast enhancement
- **3D Reconstruction**: COLMAP pipeline with geometric verification and bundle adjustment
- **Model Visualization**: Three.js-based 3D model viewer
- **Project Management**: COLMAP-compliant project structure and database management

## Architecture

### Backend (FastAPI)
- `main.py` - Main application with enhanced COLMAP integration
- Uses PyCOLMAP for structure-from-motion reconstruction
- OpenCV for advanced image preprocessing
- Comprehensive error handling and validation

### Frontend
- `templates/index.html` - Camera capture interface
- `templates/view_3d_model.html` - Three.js 3D model viewer
- Real-time feedback and loading states

## Project Structure

```
camera_tracking/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── index.html         # Camera capture interface
│   └── view_3d_model.html # 3D model viewer
├── static/                # Static files for web serving
├── colmap_project/        # COLMAP project directory (auto-created)
│   ├── images/           # Input images
│   ├── database.db       # COLMAP database
│   ├── sparse/           # Sparse reconstruction results
│   └── preprocessed/     # Preprocessed images
└── tracking/             # Virtual environment
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv tracking
source tracking/bin/activate  # On Windows: tracking\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Open `http://localhost:8000` in your browser
2. Allow camera and device orientation permissions
3. Capture multiple photos by clicking "Capture Photo"
4. Click "Render" to generate 3D model
5. View the generated 3D model

## API Endpoints

- `GET /` - Main camera capture interface
- `POST /upload` - Upload photos with orientation data
- `POST /generate-3d-model` - Generate 3D model from uploaded photos
- `GET /view-3d-model` - 3D model viewer
- `GET /project-status` - Get current project status
- `POST /reset-project` - Reset the entire project

## COLMAP Pipeline

The application implements a complete COLMAP pipeline:

1. **Feature Extraction**: SIFT feature detection with configurable parameters
2. **Feature Matching**: Exhaustive matching with geometric verification
3. **Sparse Reconstruction**: Incremental mapping with bundle adjustment
4. **Model Export**: OBJ format generation for web visualization

## Key Improvements

- **Enhanced Image Preprocessing**: Denoising and contrast enhancement for better reconstruction
- **Robust Error Handling**: Comprehensive validation and error recovery
- **Performance Monitoring**: Progress tracking and timing information
- **COLMAP Best Practices**: Proper project structure and parameter optimization
- **Better User Experience**: Loading states and detailed error messages

## Requirements

- Python 3.8+
- Modern browser with camera and gyroscope support
- Sufficient RAM for large image sets (recommended: 8GB+)

## Troubleshooting

- **Reconstruction Failed**: Ensure images have sufficient overlap and good lighting
- **Camera Access**: Check browser permissions and HTTPS requirements
- **Memory Issues**: Reduce image size or number of photos

## License

This project uses COLMAP under its respective license and follows COLMAP best practices for 3D reconstruction.
