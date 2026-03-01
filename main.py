#route_homepage.py

from fastapi import FastAPI, APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pydantic import BaseModel
import shutil
import json
import base64
import os
import cv2
import traceback
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
app = FastAPI()
router = APIRouter()

# Meshroom project structure
PROJECT_ROOT = Path("meshroom_project")
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
MESHROOM_CACHE = PROJECT_ROOT / "cache"

class PhotoData(BaseModel):
    photo: str

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def create_custom_sensor_data(image_dir: Path) -> Path:
    """Create custom sensor database with data for uploaded images."""
    sensor_db_path = image_dir.parent / "custom_cameraSensors.db"
    
    # Get image files
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    with open(sensor_db_path, 'w') as f:
        f.write("# Custom sensor database for uploaded images\n")
        f.write("# Format: Make;Model;SensorWidth(mm);Source\n\n")
        
        for img_path in image_files:
            try:
                # Read image to get dimensions and EXIF data
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width = img.shape[:2]
                    
                    # Try to extract EXIF data for more accurate sensor info
                    make = "Custom"
                    model = img_path.stem.replace("photo_", "Camera_")
                    sensor_width = 6.0  # Default fallback
                    
                    try:
                        # Use PIL to extract EXIF data
                        from PIL import Image
                        from PIL.ExifTags import TAGS, GPSTAGS
                        
                        with Image.open(img_path) as pil_img:
                            exif_data = pil_img._getexif()
                            if exif_data:
                                for tag_id, value in exif_data.items():
                                    tag = TAGS.get(tag_id, tag_id)
                                    if tag == "Make":
                                        make = str(value)
                                    elif tag == "Model":
                                        model = str(value).replace(" ", "_")
                                    elif tag == "ExifImageWidth":
                                        width = value
                                    elif tag == "ExifImageHeight":
                                        height = value
                                
                        # Estimate sensor width based on camera make/model and resolution
                        if "iPhone" in make or "Apple" in make:
                            sensor_width = 5.5  # Typical iPhone sensor
                        elif "Samsung" in make:
                            sensor_width = 5.6  # Typical Samsung sensor
                        elif "Google" in make:
                            sensor_width = 5.5  # Typical Pixel sensor
                        elif "Canon" in make or "Nikon" in make or "Sony" in make:
                            if max(width, height) > 4000:
                                sensor_width = 36.0  # Full frame
                            else:
                                sensor_width = 23.5  # APS-C
                        elif max(width, height) > 3000:  # High resolution
                            sensor_width = 35.0  # Assume full-frame
                        elif max(width, height) > 2000:  # Medium resolution
                            sensor_width = 5.5  # Assume smartphone
                        else:  # Low resolution
                            sensor_width = 6.0  # Assume action camera
                            
                    except Exception as exif_error:
                        logger.debug(f"Could not extract EXIF from {img_path}: {exif_error}")
                        # Fallback to resolution-based estimation
                        if max(width, height) > 3000:
                            sensor_width = 35.0
                        elif max(width, height) > 2000:
                            sensor_width = 5.5
                        else:
                            sensor_width = 6.0
                    
                    f.write(f"{make};{model};{sensor_width};EXIF_Custom\n")
                    
            except Exception as e:
                logger.warning(f"Could not process {img_path}: {e}")
                # Add fallback entry
                f.write(f"Custom;{img_path.stem};6.0;Fallback\n")
        
        # Add some common fallbacks
        f.write("\n# Fallback entries\n")
        f.write("Generic;Unknown_Camera;6.0;Default\n")
        f.write("Generic;Smartphone;5.5;Default\n")
        f.write("Generic;DSLR;35.0;Default\n")
    
    return sensor_db_path

@router.post("/upload")
async def upload_photos(request: Request):
    try:
        # Ensure COLMAP project structure exists
        PROJECT_ROOT.mkdir(exist_ok=True)
        IMAGES_DIR.mkdir(exist_ok=True)
        
        form = await request.form()
        photos = []

        # Extract only files from the form
        for key in form:
            if key.startswith("photo_"):
                photos.append(form[key])

        if not photos:
            return JSONResponse(content={"error": "No photos provided"}, status_code=400)

        logger.info(f"Processing {len(photos)} photos")
        
        for index, photo in enumerate(photos):
            # Save the uploaded photo to Meshroom images directory
            photo_filename = IMAGES_DIR / f"photo_{index + 1:03d}.jpg"
            with open(photo_filename, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

        return JSONResponse(content={
            "message": f"Successfully uploaded {len(photos)} photos!",
            "project_path": str(PROJECT_ROOT)
        })
    except Exception as e:
        logger.error(f"Error uploading photos: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

def validate_images(image_dir: Path) -> bool:
    """Validate that images meet photogrammetry requirements."""
    logger.info(f"Validating images in directory: {image_dir}")
    
    if not image_dir.exists() or not image_dir.is_dir():
        logger.error(f"Directory does not exist: {image_dir}")
        return False
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} image files")
    
    if len(image_files) < 2:
        logger.error("At least 2 images are required for reconstruction")
        return False
    
    # Check image sizes and formats
    for img_path in image_files:
        logger.info(f"Validating image: {img_path}")
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Cannot read image: {img_path}")
                return False
            
            height, width = img.shape[:2]
            logger.info(f"Image {img_path} size: {width}x{height}")
            if width < 100 or height < 100:
                logger.error(f"Image too small: {img_path} ({width}x{height})")
                return False
                
        except Exception as e:
            logger.error(f"Error validating image {img_path}: {e}")
            return False
    
    logger.info(f"Validated {len(image_files)} images")
    return True

# Helper function to preprocess images and remove background noise
def preprocess_image(input_path: Path, output_path: Path) -> bool:
    """Preprocess image for better 3D reconstruction."""
    try:
        image = cv2.imread(str(input_path))
        if image is None:
            return False
            
        # Basic preprocessing: denoise and enhance contrast
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(str(output_path), result)
        return True
    except Exception as e:
        logger.error(f"Error preprocessing image {input_path}: {e}")
        return False

def run_meshroom_pipeline(image_dir: Path, output_dir: Path, cache_dir: Path) -> Optional[Path]:
    """Run complete Meshroom pipeline."""
    try:
        start_time = time.time()
        logger.info("Starting Meshroom pipeline...")
        
        # Ensure directories exist
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create custom sensor database for uploaded images (use original images for EXIF data)
        original_images_dir = image_dir.parent / "images"
        if original_images_dir.exists():
            custom_sensor_db = create_custom_sensor_data(original_images_dir)
        else:
            custom_sensor_db = create_custom_sensor_data(image_dir)
        logger.info(f"Created custom sensor database: {custom_sensor_db}")
        
        # Check if Meshroom is available
        meshroom_path = Path(__file__).parent / "Meshroom-2023.3.0" / "meshroom_batch"
        try:
            subprocess.run([str(meshroom_path), "--help"], capture_output=True, check=True)
            logger.info("Meshroom found and accessible")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Meshroom not found. Please install Meshroom first.")
            return None
        
        # Run Meshroom pipeline
        logger.info("Running Meshroom photogrammetry pipeline...")
        
        # Set environment variables for Meshroom
        env = os.environ.copy()
        env['ALICEVISION_ROOT'] = str(Path(__file__).parent / "Meshroom-2023.3.0" / "aliceVision")
        env['LD_LIBRARY_PATH'] = f"{env['ALICEVISION_ROOT']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        # Use our custom sensor database
        env['ALICEVISION_SENSOR_DB'] = str(custom_sensor_db)
        env['ALICEVISION_LENS_PROFILE_INFO'] = str(custom_sensor_db)
        
        cmd = [
            str(meshroom_path),
            "-i", str(image_dir.absolute()),
            "-p", "photogrammetry",
            "-o", str(output_dir.absolute()),
            "--cache", str(cache_dir.absolute()),
            # Override CameraInit parameters with valid ones only
            "--paramOverrides", "CameraInit.viewIdMethod=filename",
            "--paramOverrides", "CameraInit.defaultFieldOfView=45.0",
            # Force CPU processing to avoid GPU memory issues
            "--paramOverrides", "DepthMap.nbGPUs=0",
            "--compute", "yes"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run Meshroom with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Monitor progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"Meshroom: {output.strip()}")
        
        # Check if Meshroom completed successfully
        if process.returncode == 0:
            logger.info("Meshroom pipeline completed successfully")
            
            # Find the generated mesh file
            mesh_files = list(output_dir.rglob("*.obj"))
            if mesh_files:
                mesh_path = mesh_files[0].parent
                logger.info(f"Generated mesh found at: {mesh_path}")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Meshroom pipeline completed in {elapsed_time:.2f} seconds")
                
                return mesh_path
            else:
                logger.error("No mesh files found in output")
                return None
        else:
            logger.error(f"Meshroom failed with return code: {process.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"Error in Meshroom pipeline: {str(e)}")
        traceback.print_exc()
        return None

@router.post("/generate-3d-model")
async def generate_3d_model():
    try:
        logger.info("Starting 3D model generation...")
        
        # Validate input images
        if not validate_images(IMAGES_DIR):
            return JSONResponse(
                content={"error": "Image validation failed. Please upload at least 2 valid images."}, 
                status_code=400
            )
        
        # Clear previous preprocessed photos
        preprocessed_dir = PROJECT_ROOT / "preprocessed"
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
        preprocessed_dir.mkdir(exist_ok=True)
        
        # Preprocess images
        logger.info("Preprocessing images...")
        image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
        logger.info(f"Found {len(image_files)} images in {IMAGES_DIR}")
        preprocessed_images = []
        
        for img_path in image_files:
            logger.info(f"Processing image: {img_path}")
            preprocessed_path = preprocessed_dir / img_path.name
            if preprocess_image(img_path, preprocessed_path):
                preprocessed_images.append(preprocessed_path)
                logger.info(f"Successfully preprocessed: {preprocessed_path}")
            else:
                logger.error(f"Failed to preprocess: {img_path}")
        
        logger.info(f"Preprocessed {len(preprocessed_images)} images successfully")
        
        if not preprocessed_images:
            return JSONResponse(
                content={"error": "No valid images after preprocessing"}, 
                status_code=400
            )
        
        # Run Meshroom pipeline
        logger.info("Running Meshroom reconstruction...")
        model_path = run_meshroom_pipeline(
            image_dir=preprocessed_dir,
            output_dir=OUTPUT_DIR,
            cache_dir=MESHROOM_CACHE
        )
        
        if model_path is None:
            return JSONResponse(
                content={
                    "error": "3D reconstruction failed. This usually happens when images don't have enough overlapping features. Try taking photos with more overlap and better lighting.",
                    "suggestion": "Capture photos with 60-80% overlap and distinctive features"
                }, 
                status_code=500
            )
        
        # Find the OBJ file in the output
        obj_files = list(model_path.rglob("*.obj"))
        if not obj_files:
            return JSONResponse(
                content={"error": "No OBJ file found in Meshroom output"}, 
                status_code=500
            )
        
        obj_path = obj_files[0]
        
        # Copy to static directory for web serving
        static_models_dir = Path("static/models")
        static_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy new model
        shutil.copy2(obj_path, static_models_dir / "model.obj")
        
        return JSONResponse(content={
            "message": "3D model generated successfully!", 
            "model_path": "/static/models/model.obj",
            "stats": {
                "images_processed": len(image_files),
                "model_location": str(model_path),
                "is_meshroom": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/view-3d-model")
async def view_3d_model(request: Request):
    return templates.TemplateResponse("view_3d_model.html", {"request": request})

def create_obj_file(model_path: Path, obj_path: Path):
    """Create OBJ file from 3D model for web viewer."""
    try:
        # For Meshroom, just copy the existing OBJ file
        obj_files = list(model_path.rglob("*.obj"))
        if obj_files:
            shutil.copy2(obj_files[0], obj_path)
            logger.info(f"Copied OBJ file from {obj_files[0]} to {obj_path}")
            return True
        else:
            # Create a simple placeholder OBJ if no mesh found
            with open(obj_path, 'w') as obj_file:
                obj_file.write("# 3D Model generated by Meshroom\n")
                obj_file.write("# Camera Tracking Project\n\n")
                obj_file.write("# Placeholder OBJ file\n")
                obj_file.write("v 0 0 0\n")
                obj_file.write("v 1 0 0\n")
                obj_file.write("v 0 1 0\n")
            return True
    except Exception as e:
        logger.error(f"Error creating OBJ file: {e}")
        return False

@router.get("/project-status")
async def get_project_status():
    """Get current project status and statistics."""
    try:
        status = {
            "project_path": str(PROJECT_ROOT),
            "images_count": len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0,
            "output_exists": OUTPUT_DIR.exists(),
            "cache_exists": MESHROOM_CACHE.exists()
        }
        
        # Add mesh info if available
        if OUTPUT_DIR.exists():
            try:
                obj_files = list(OUTPUT_DIR.rglob("*.obj"))
                if obj_files:
                    status["mesh_info"] = {
                        "obj_file": str(obj_files[0]),
                        "file_size": obj_files[0].stat().st_size if obj_files[0].exists() else 0
                    }
            except Exception as e:
                logger.warning(f"Could not load mesh info: {e}")
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting project status: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/reset-project")
async def reset_project():
    """Reset the entire Meshroom project."""
    try:
        if PROJECT_ROOT.exists():
            shutil.rmtree(PROJECT_ROOT)
            logger.info("Project reset successfully")
        
        return JSONResponse(content={"message": "Project reset successfully"})
        
    except Exception as e:
        logger.error(f"Error resetting project: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Include the router in the FastAPI app
app.include_router(router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
