import sqlite3
from uuid import uuid4
from hashlib import sha256

import numpy as np
from pydantic import BaseModel, Base64Bytes,  validator, model_validator
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from decoder import base64_decoder
from face_recognition import (detect_faces, insightface_detector_factory,
                              is_similar, similarity_calculator)


COLOR_SPACES_DIMANTION = {
    'RGB': 3,
    'BGR': 3,
    'GREY': 1
}


class PostBody(BaseModel):
    img_1: Base64Bytes
    img_1_metadata: tuple[tuple[int, int], str]
    img_2: Base64Bytes
    img_2_metadata: tuple[tuple[int, int], str]

    # @validator('img_1', 'img_2', 'img_1_metadata', 'img_2_metadata')
    # @classmethod
    # def validate_not_empty(cls, value):
    #     if len(value) == 0:
    #         raise ValueError('image can\'t be empty.')

    @model_validator(mode='before')
    @classmethod
    def validate_not_same(cls, values):
        img1 = values.get('img_1')
        img2 = values.get('img_2')

        if img1 == img2:
            raise ValueError('Images can not be same.')

    @validator('img_1_metadata', 'img_2_metadata')
    @classmethod
    def validate_metadata(cls, value):
        if len(value) != 2:
            raise ValueError('Incorrect argument metadata length.')

        size, type_ = value[0], value[1]
        if len(size) != 2:
            raise ValueError('Image width and height missing.')

        width, height = size[0], size[1]
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError('width and height of image needs to be integer.')

        if not isinstance(type_, str):
            raise TypeError('type of metadata needs to be str.')

        if type_.upper() not in COLOR_SPACES_DIMANTION:
            raise TypeError('incorrect img type')


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
                cur.execute('INSERT INTO api_keys (api_key) VALUES (?)', [hash_])
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


@app.post('/face-similarity')
async def face_similarity(post: PostBody):
    img1 = base64_decoder(
        post.img_1,
        (
            *post.img_1_metadata[0],
            COLOR_SPACES_DIMANTION[post.img_1_metadata[1]]
        )
    )
    img2 = base64_decoder(
        post.img_2,
        (
            *post.img_2_metadata[0],
            COLOR_SPACES_DIMANTION[post.img_2_metadata[1]]
        )
    )

    face_detector = insightface_detector_factory()

    faces1 = detect_faces(img1, face_detector)
    faces2 = detect_faces(img2, face_detector)

    if not faces1:
        return {
            'is_success': False,
            'detail': 'img1 doesn\'t contain a face'
        }
    if not faces2:
        return {
            'is_success': False,
            'detail': 'img2 doesn\'t contain a face'
        }

    for face1 in faces1:
        for face2 in faces2:

            similarity = similarity_calculator(
                face1['embedding'], face2['embedding'])
            similar = is_similar(similarity)
            if similar:
                return {
                    'is_success': True,
                    'is_similar': similar,
                    'similarity_score': similarity
                }
