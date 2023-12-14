from hashlib import sha256
import sqlite3
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image
import io
import numpy as np

app = FastAPI()

class ImagesData(BaseModel):
    image1: str
    image2: str

def decode_base64_image(image_base64: str) -> np.ndarray:
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_bytes))
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

def scale_range(val: float, max_: float, min_: float)->float:
    return (val - min_)/(max_ - min_)

def generate_api_key()->str:
    with sqlite3.connect('data/api_key_db.db') as conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS api_keys (api_key TEXT NOT NULL PRIMARY KEY)')
        
        while True:
            uuid = uuid4()
            hash_ = sha256(uuid.bytes).hexdigest()

            query = 'SELECT api_key FROM api_keys WHERE api_key = ?'
            if cur.execute(query, [hash_]).fetchone() is None:
                cur.execute('INSERT INTO api_keys (api_key) VALUES (?)', [hash_])
                break
    
    return uuid.hex

@app.get('/api-key')
async def api_key_to_user():
    return generate_api_key()

@app.get()


@app.post("/process-images")
async def process_images(images_data: ImagesData):
    # Decode and convert base64 images to NumPy arrays
    img_array1 = decode_base64_image(images_data.image1)
    img_array2 = decode_base64_image(images_data.image2)

    # Add your image processing logic here if needed
    # For now, we'll just return the shape of the NumPy arrays
    result = {"image1_shape": img_array1.shape, "image2_shape": img_array2.shape}

    return result
