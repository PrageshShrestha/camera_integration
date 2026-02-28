#route_homepage.py

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List
from pydantic import BaseModel
import shutil
import json
import base64

templates = Jinja2Templates(directory="templates")
app = APIRouter()

class PhotoData(BaseModel):
    photo: str
    orientation: dict

@app.get("/")
async def home(request: Request):
	return templates.TemplateResponse("index.html",{"request":request})

@app.post("/upload")
async def upload_photos(photos: List[PhotoData]):
    try:
        for index, photo_data in enumerate(photos):
            # Decode the base64 photo data and save it as a file
            photo_bytes = base64.b64decode(photo_data.photo.split(",")[1])
            photo_filename = f"static/photo_{index + 1}.jpg"
            with open(photo_filename, "wb") as photo_file:
                photo_file.write(photo_bytes)

            # Save orientation data to a JSON file
            orientation_data = {
                "photo": photo_filename,
                "orientation": photo_data.orientation
            }
            with open("static/gyroscope_data.json", "a") as json_file:
                json_file.write(json.dumps(orientation_data) + "\n")

        return JSONResponse(content={"message": "Photos and orientation data uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
