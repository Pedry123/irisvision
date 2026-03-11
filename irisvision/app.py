import io
from http import HTTPStatus

import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image
from ultralytics import YOLO


def load_model():
    modelpath = r'models/best.pt'
    model = YOLO(modelpath)

    return model


model = load_model()

app = FastAPI()

@app.get('/health', status_code=HTTPStatus.OK)
def get_health():
    return {'Health': 'Ok', 'Status Code': HTTPStatus.OK}


@app.post('/predict', status_code=HTTPStatus.OK)
async def post_predictions(file: UploadFile):
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    result = model(image)
    names = result[0].names
    probability = result[0].probs.data.numpy()
    prediction = np.argmax(probability)
    response = {
        'Prediction': names[prediction],
        'Confidence': float(probability[prediction]),
    }

    return response


@app.get('/classifications', status_code=HTTPStatus.OK)
def get_classifications_list():
    ...

    
@app.get('statistics', status_code=HTTPStatus.OK)
def get_statistics():
    ...