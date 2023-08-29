import sqlite3
from uuid import uuid4
from hashlib import sha256
from itertools import product

from pydantic import BaseModel, Base64Bytes, validator, model_validator, Base64Str
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from decoder import base64_decoder
from face_recognition import (
    is_similar,
    similarity_calculator,
    RetinaFaceDetector,
    ArcFaceRecognizer
)

class Image(BaseModel):
    img: Base64Bytes
    shape: tuple[int, int, int]


def scale_range_between_1_0(val: float, max_:float, min_:float) -> float:
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
    image_1: Image,
    image_2: Image,
    api_key: str = Security(validate_api_key)
):
    if image_1.img == image_2.img:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Image strings can not be same.'
        )

    # print(f'{image_1.img=}')
    # print(f'{image_2.img=}')
    img1 = base64_decoder(image_1.img, image_1.shape)
    img2 = base64_decoder(image_2.img, image_2.shape)

    detector = RetinaFaceDetector()

    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)

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
