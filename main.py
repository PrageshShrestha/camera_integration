#route_homepage.py

from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List
from pydantic import BaseModel
import shutil
import json
import base64
import os
import cv2
import pycolmap

templates = Jinja2Templates(directory="templates")
app = APIRouter()

class PhotoData(BaseModel):
    photo: str
    orientation: dict

@app.get("/")
async def home(request: Request):
	return templates.TemplateResponse("index.html",{"request":request})

@app.post("/upload")
async def upload_photos(request: Request):
    try:
        form = await request.form()
        photos = []
        orientations = []

        # Extract files and orientations from the form
        for key in form:
            if key.startswith("photo_"):
                photos.append(form[key])
            elif key.startswith("orientation_"):
                orientations.append(json.loads(form[key]))

        for index, photo in enumerate(photos):
            # Save the uploaded photo to a file
            photo_filename = f"static/photo_{index + 1}.jpg"
            with open(photo_filename, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

            # Save orientation data to a JSON file
            orientation_data = {
                "photo": photo_filename,
                "orientation": orientations[index]
            }
            with open("static/gyroscope_data.json", "a") as json_file:
                json_file.write(json.dumps(orientation_data) + "\n")

        return JSONResponse(content={"message": "Photos and orientation data uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Helper function to preprocess images and remove background noise
def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, result)

# Helper function to run COLMAP and generate a 3D model
def run_colmap(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    reconstruction = pycolmap.Reconstruction()
    reconstruction.run(image_dir, output_dir)
    return os.path.join(output_dir, "model.obj")

@app.post("/generate-3d-model")
async def generate_3d_model():
    try:
        image_dir = "static/preprocessed_photos"
        output_dir = "static/models"
        os.makedirs(image_dir, exist_ok=True)

        # Preprocess uploaded photos
        for index, photo in enumerate(os.listdir("static")):
            if photo.startswith("photo_"):
                preprocess_image(f"static/{photo}", f"{image_dir}/{photo}")

        # Run COLMAP to generate the 3D model
        model_path = run_colmap(image_dir, output_dir)

        return JSONResponse(content={"message": "3D model generated successfully!", "model_path": model_path})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
