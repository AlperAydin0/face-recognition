import pydantic
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
import sqlite3
from hashlib import sha256


class PostBody(pydantic.BaseModel):
    img_1: pydantic.Base64Str
    img_1_metadata: tuple[tuple[int, int], str]
    img_2: pydantic.Base64Str
    img_2_metadata: tuple[tuple[int, int], str]    

def generate_api_key() -> str:
    from uuid import uuid4
    with sqlite3.connect('data/api_key_db.db') as conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS api_keys (api_key TEXT NOT NULL PRIMARY KEY)')
        while True:
            uuid = uuid4()
            h = sha256(uuid.bytes).hexdigest()
            if cur.execute('SELECT api_key FROM api_keys WHERE api_key = ?', [h]).fetchone() is None:
                cur.execute('INSERT INTO api_keys (api_key) VALUES (?)', [h])
                print(f'inserting: {h}')
                break

    return uuid.hex

api_key_header = APIKeyHeader(name='api-key')
def validate_api_key(api_key_header: str = Security(api_key_header)) -> str:
    with sqlite3.connect('data/api_key_db.db') as conn:
        cur = conn.cursor()
        print(f'{api_key_header=}')
        if cur.execute('SELECT api_key FROM api_keys WHERE api_key = ?', [sha256(bytes.fromhex(api_key_header)).hexdigest()]).fetchone() is None:
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


@app.post('/face-recognition')
async def face_recognition(post: PostBody):
    raise NotImplementedError
