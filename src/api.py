import numpy as np
import io
from PIL import Image
import base64
from pathlib import Path
import sqlite3
from uuid import uuid4
from hashlib import sha256
from itertools import product

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from face_recognition import (
    is_similar,
    similarity_calculator,
    RetinaFaceDetector,
    ArcFaceRecognizer
)

# class Image_(BaseModel):
#     img: Base64Bytes
#     img_format: str


app = FastAPI()


class ImagesData(BaseModel):
    image1: str
    image2: str


def decode(image_base64: str) -> np.ndarray:
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_bytes))
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Invalid base64 encoding") from e


@app.post("/process-images",)
async def process_images(images_data: ImagesData):
    # Decode and convert base64 images to NumPy arrays
    img_array1 = decode(images_data.image1)
    img_array2 = decode(images_data.image2)

    # Add your image processing logic here if needed
    # For now, we'll just return the shape of the NumPy arrays
    result = {"image1_shape": img_array1.shape,
              "image2_shape": img_array2.shape}

    return result


def scale_range_between_1_0(val: float, max_: float, min_: float) -> float:
    return (val - min_) / (max_ - min_)


def generate_api_key() -> str:
    '''Generates and stores(Hashed) api key

    Returns:
        str: api_key for user to save
    '''

    with sqlite3.connect('data/api_key_db.db') as conn:
        cur = conn.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS api_keys (api_key TEXT NOT NULL PRIMARY KEY)')

        while True:
            uuid = uuid4()
            hash_ = sha256(uuid.bytes).hexdigest()

            # check for api keys existance
            query = 'SELECT api_key FROM api_keys WHERE api_key = ?'
            if cur.execute(query, [hash_]).fetchone() is None:
                cur.execute(
                    'INSERT INTO api_keys (api_key) VALUES (?)', [hash_])
                break

    return uuid.hex


api_key_header = APIKeyHeader(name='api-key')


def validate_api_key(api_key_header: str = Security(api_key_header)) -> str:
    with sqlite3.connect('data/api_key_db.db') as conn:
        cur = conn.cursor()
        if cur.execute('SELECT api_key FROM api_keys WHERE api_key = ?',
                       [sha256(bytes.fromhex(api_key_header)).hexdigest()]).fetchone() is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='invalid or missing api key'
            )
    return api_key_header


app = FastAPI()


@app.get('/')
async def hello_world():
    return 'Hello World'


@app.get('/api-key')
async def api_key_to_user():
    return generate_api_key()


@app.get('/test')
async def test_auth(api_key: str = Security(validate_api_key)) -> str:
    return 'valid api key'


@app.post('/face-similarity/', )
def face_similarity(
    images: ImagesData,
    api_key: str = Security(validate_api_key)
):
    image_1, image_2 = images.image1, images.image2
    if image_1 == image_2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Image strings can not be same.'
        )

    # print(f'{image_1[:50]=}')
    # print(f'{image_2[:50]=}')
    # img1 = base64_decoder(image_1.img, image_1.shape)
    # image_1.img.
    # img2 = base64_decoder(image_2.img, image_2.shape)
    img1 = decode(image_1)
    img2 = decode(image_2)

    print(type(img1))
    print(type(img2))

    detector = RetinaFaceDetector()

    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)

    print(faces1)
    print(faces2)

    if faces1 is None:
        return {
            'is_success': False,
            'detail': 'img1 doesn\'t contain a face'
        }
    if faces2 is None:
        return {
            'is_success': False,
            'detail': 'img2 doesn\'t contain a face'
        }

    recognizer = ArcFaceRecognizer()
    embeddings_1 = [recognizer.get_embedding(img1, face) for face in faces1]
    embeddings_2 = [recognizer.get_embedding(img2, face) for face in faces2]

    similarity_range = {'high': 1, 'low': -1}

    max_sim = None
    for emb1, emb2 in product(embeddings_1, embeddings_2):

        similarity = similarity_calculator(emb1, emb2)
        if max_sim is None:
            max_sim = similarity
        else:
            max_sim = max(similarity, max_sim)

        similar = is_similar(similarity)
        if similar:
            return {
                'is_success': True,
                'is_similar': bool(similar),
                'similarity_score': scale_range_between_1_0(float(similarity), similarity_range['high'], similarity_range['low'])
            }

    return {
        'is_success': True,
        'is_similar': False,
        'similarity_score': scale_range_between_1_0(float(max_sim), similarity_range['high'], similarity_range['low'])
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(f'{Path(__file__).stem}:app',
                host='127.0.0.1', port=8000, reload=True)
