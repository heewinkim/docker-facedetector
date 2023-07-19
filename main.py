import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI,HTTPException
from face_detection import FaceDetection
from pydantic import BaseModel,Field
import traceback

app = FastAPI()
detector = FaceDetection()


class InputData(BaseModel):
    imageData: str = Field(description='One of base64jpeg string | url adress')


def image_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail="An error occurred while reading image.")

    return wrapper

@image_exception
def read_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

@image_exception
def read_from_b64(b64_str):
    pil_img = Image.open(BytesIO(base64.b64decode(b64_str)))
    return pil_img


@app.get("/")
def index():
    return {'statusCode':200,'message':'ok'}


@app.post("/face")
async def face_detection(input_data:InputData):

    if input_data.imageData.startswith('http'):
        pil_img = read_from_url(input_data.imageData)
    else:
        pil_img = read_from_b64(input_data.imageData)

    return detector.detect(np.array(pil_img))