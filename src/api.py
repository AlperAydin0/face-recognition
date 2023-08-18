from fastapi import FastAPI
import pydantic

class PostBody(pydantic.BaseModel):
    img_1: pydantic.Base64Str
    img_1_metadata: tuple[tuple[int, int], str]
    img_2: pydantic.Base64Str
    img_2_metadata: tuple[tuple[int, int], str]
    api_key: str

app = FastAPI()

@app.get('/')
def hello_world():
    return 'Hello World'


@app.get('/api-key')
def get_api_key():
    raise NotImplementedError


@app.post('/face-recognition')
def face_recognition(post: PostBody):
    raise NotImplementedError