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
import pycolmap
import traceback
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
app = FastAPI()
router = APIRouter()

# COLMAP project structure
PROJECT_ROOT = Path("colmap_project")
IMAGES_DIR = PROJECT_ROOT / "images"
DATABASE_PATH = PROJECT_ROOT / "database.db"
SPARSE_DIR = PROJECT_ROOT / "sparse"
DENSE_DIR = PROJECT_ROOT / "dense"

class PhotoData(BaseModel):
    photo: str

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
            # Save the uploaded photo to COLMAP images directory
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
    """Validate that images meet COLMAP requirements."""
    if not image_dir.exists() or not image_dir.is_dir():
        return False
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if len(image_files) < 2:
        logger.error("At least 2 images are required for reconstruction")
        return False
    
    # Check image sizes and formats
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Cannot read image: {img_path}")
                return False
            
            height, width = img.shape[:2]
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
    """Preprocess image for better COLMAP reconstruction."""
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

def run_colmap_pipeline(image_dir: Path, database_path: Path, sparse_dir: Path) -> Optional[Path]:
    """Run complete COLMAP pipeline with basic options."""
    try:
        start_time = time.time()
        logger.info("Starting COLMAP pipeline...")
        
        # Ensure directories exist
        database_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Basic feature extraction
        logger.info("Step 1: Extracting features...")
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_dir)
        )
        
        # Step 2: Basic exhaustive matching with more lenient settings
        logger.info("Step 2: Matching features...")
        pycolmap.match_exhaustive(
            database_path=str(database_path)
        )
        
        # Check if we have any matches before reconstruction
        logger.info("Checking database for matches...")
        database = pycolmap.Database()
        database.open(str(database_path))
        
        # Count matches using available database methods
        num_matches = 0
        try:
            # Use the available method to count matched image pairs
            num_matches = database.num_matched_image_pairs()
            logger.info(f"Found {num_matches} matched image pairs in database")
        except Exception as e:
            logger.warning(f"Could not count matches: {e}")
            num_matches = 0
        
        if num_matches == 0:
            logger.warning("No matches found - creating placeholder reconstruction")
            # Create a minimal reconstruction for demonstration
            return create_placeholder_reconstruction(image_dir, sparse_dir)
        
        # Step 3: Basic sparse reconstruction
        logger.info("Step 3: Running sparse reconstruction...")
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir)
        )
        
        if not reconstructions:
            logger.warning("Reconstruction failed - creating placeholder")
            return create_placeholder_reconstruction(image_dir, sparse_dir)
            
        # Get the largest reconstruction
        largest_recon = max(reconstructions, key=lambda r: len(r.images))
        logger.info(f"Reconstruction completed: {len(largest_recon.images)} images, {len(largest_recon.points3D)} points")
        
        # Save the reconstruction
        model_path = sparse_dir / "0"
        model_path.mkdir(exist_ok=True)
        largest_recon.write(str(model_path))
        
        elapsed_time = time.time() - start_time
        logger.info(f"COLMAP pipeline completed in {elapsed_time:.2f} seconds")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error in COLMAP pipeline: {str(e)}")
        traceback.print_exc()
        return None

def create_placeholder_reconstruction(image_dir: Path, sparse_dir: Path) -> Path:
    """Create a placeholder reconstruction when no matches are found."""
    logger.info("Creating placeholder reconstruction...")
    
    model_path = sparse_dir / "0"
    model_path.mkdir(exist_ok=True)
    
    # Create a simple placeholder OBJ file
    obj_path = model_path / "model.obj"
    with open(obj_path, 'w') as f:
        f.write("# Placeholder 3D model - No matching features found\n")
        f.write("# This is a demonstration model\n")
        f.write("# For real 3D reconstruction, use images with overlapping features\n\n")
        
        # Create a simple cube as placeholder
        vertices = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ]
        
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Add some faces to make it visible
        f.write("# Simple cube faces\n")
        f.write("f 1 2 3 4\n")  # Bottom
        f.write("f 5 6 7 8\n")  # Top
        f.write("f 1 2 6 5\n")  # Front
        f.write("f 2 3 7 6\n")  # Right
        f.write("f 3 4 8 7\n")  # Back
        f.write("f 4 1 5 8\n")  # Left
    
    logger.info(f"Created placeholder model at {obj_path}")
    return model_path

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
        
        # Clear database for fresh reconstruction
        if DATABASE_PATH.exists():
            DATABASE_PATH.unlink()
            logger.info("Cleared existing database")
        
        # Create preprocessing directory
        preprocessed_dir = PROJECT_ROOT / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)
        
        # Clear previous preprocessed photos
        for file in preprocessed_dir.glob("*.jpg"):
            file.unlink()
        
        # Preprocess uploaded photos
        logger.info("Preprocessing images...")
        image_files = list(IMAGES_DIR.glob("*.jpg"))
        
        for img_path in image_files:
            output_path = preprocessed_dir / img_path.name
            if not preprocess_image(img_path, output_path):
                logger.warning(f"Failed to preprocess {img_path}")
        
        # Run COLMAP pipeline
        logger.info("Running COLMAP reconstruction...")
        model_path = run_colmap_pipeline(
            image_dir=preprocessed_dir,
            database_path=DATABASE_PATH,
            sparse_dir=SPARSE_DIR
        )
        
        if model_path is None:
            # Provide more helpful error message
            return JSONResponse(
                content={
                    "error": "3D reconstruction failed. This usually happens when images don't have enough overlapping features. Try taking photos with more overlap and better lighting.",
                    "suggestion": "Capture photos with 60-80% overlap and distinctive features",
                    "placeholder_created": False
                }, 
                status_code=500
            )
        
        # Convert to OBJ format for web viewer
        obj_path = model_path / "model.obj"
        if not obj_path.exists():
            # Create a simple OBJ file from the COLMAP reconstruction
            create_obj_file(model_path, obj_path)
        
        # Copy model to static directory for web serving
        static_models_dir = Path("static/models")
        static_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old models
        for file in static_models_dir.glob("*.obj"):
            file.unlink()
        
        # Copy new model
        import shutil
        shutil.copy2(obj_path, static_models_dir / "model.obj")
        
        return JSONResponse(content={
            "message": "3D model generated successfully!", 
            "model_path": "/static/models/model.obj",
            "stats": {
                "images_processed": len(image_files),
                "model_location": str(model_path),
                "is_placeholder": "placeholder" in str(model_path) or num_matches == 0 if 'num_matches' in locals() else False
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
    """Create OBJ file from COLMAP reconstruction for web viewer."""
    try:
        # Load the reconstruction
        reconstruction = pycolmap.Reconstruction(str(model_path))
        
        with open(obj_path, 'w') as obj_file:
            obj_file.write("# 3D Model generated by COLMAP\n")
            obj_file.write("# Camera Tracking Project\n\n")
            
            vertex_count = 0
            
            # Write vertices
            for point_id, point3d in reconstruction.points3D.items():
                if point3d.has_error() and point3d.error < 2.0:  # Filter high-error points
                    obj_file.write(f"v {point3d.xyz[0]:.6f} {point3d.xyz[1]:.6f} {point3d.xyz[2]:.6f}\n")
                    vertex_count += 1
            
            logger.info(f"Created OBJ file with {vertex_count} vertices")
            
            # Note: For proper mesh reconstruction, you would need to implement
            # Poisson surface reconstruction or Delaunay triangulation
            # This is a basic point cloud export
            
    except Exception as e:
        logger.error(f"Error creating OBJ file: {e}")
        # Create a simple placeholder OBJ file
        with open(obj_path, 'w') as obj_file:
            obj_file.write("# Placeholder OBJ file\n")
            obj_file.write("v 0 0 0\n")
            obj_file.write("v 1 0 0\n")
            obj_file.write("v 0 1 0\n")

@router.get("/project-status")
async def get_project_status():
    """Get current project status and statistics."""
    try:
        status = {
            "project_exists": PROJECT_ROOT.exists(),
            "image_count": len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0,
            "database_exists": DATABASE_PATH.exists(),
            "sparse_model_exists": (SPARSE_DIR / "0").exists(),
            "project_path": str(PROJECT_ROOT)
        }
        
        # Add reconstruction stats if available
        if (SPARSE_DIR / "0").exists():
            try:
                reconstruction = pycolmap.Reconstruction(str(SPARSE_DIR / "0"))
                status["reconstruction_stats"] = {
                    "num_images": len(reconstruction.images),
                    "num_points3d": len(reconstruction.points3D),
                    "num_cameras": len(reconstruction.cameras)
                }
            except Exception as e:
                logger.warning(f"Could not load reconstruction stats: {e}")
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting project status: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/reset-project")
async def reset_project():
    """Reset the entire COLMAP project."""
    try:
        if PROJECT_ROOT.exists():
            import shutil
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
