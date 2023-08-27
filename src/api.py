import sqlite3
from uuid import uuid4
from hashlib import sha256
from itertools import product

from pydantic import BaseModel, Base64Bytes, validator, model_validator
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from decoder import base64_decoder
from face_recognition import (
    is_similar,
    similarity_calculator,
    RetinaFaceDetector,
    ArcFaceRecognizer
)


class PostBody(BaseModel):
    img_1: Base64Bytes
    img_1_shape: tuple[int, int, int]
    img_2: Base64Bytes
    img_2_shape: tuple[int, int, int]

    # @validator('img_1', 'img_2', 'img_1_shape', 'img_2_shape')
    # @classmethod
    # def validate_not_empty(cls, value):
    #     if len(value) == 0:
    #         raise ValueError('image can\'t be empty.')

    # @model_validator(mode='before')
    # @classmethod
    # def validate_not_same(cls, values):
    #     img1 = values[0]
    #     img2 = values[2]

    #     if img1 == img2:
    #         raise ValueError('Images can not be same.')

    @validator('img_1_shape', 'img_2_shape')
    @classmethod
    def validate_shape(cls, value):
        if len(value) != 3:
            raise ValueError('Incorrect image size.')

        width, height, chanels = value[0], value[1], value[2]
        if not isinstance(width, int) or not isinstance(height, int) or not isinstance(chanels, int):
            raise TypeError(
                'width, height and chanels of image needs to be integer.')


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
        print(f'{api_key_header=}')
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


@app.post('/face-similarity/')
async def face_similarity(post: PostBody,api_key: str = Security(validate_api_key)):
    img1 = base64_decoder(
        post.img_1,
        post.img_1_shape
    )
    img2 = base64_decoder(
        post.img_2,
        post.img_2_shape
    )

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
                'is_similar': similar,
                'similarity_score': similarity
            }

    return {
        'is_success': True,
        'is_similar': False,
        'similarity_score': max_sim
    }
