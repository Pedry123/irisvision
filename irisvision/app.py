import io
from http import HTTPStatus

import numpy as np
from fastapi import Depends, FastAPI, UploadFile
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from ultralytics import YOLO

from irisvision.database import get_session
from irisvision.models import Classification
from irisvision.schemas import ClassificationList, ClassificationPublic


def load_model():
    modelpath = r'models/best.pt'
    model = YOLO(modelpath)

    return model


model = load_model()

app = FastAPI()


@app.get('/health', status_code=HTTPStatus.OK)
def get_health():
    return {'Health': 'Ok', 'Status Code': HTTPStatus.OK}


@app.post(
    '/predict', status_code=HTTPStatus.OK, response_model=ClassificationPublic
)
async def post_predictions(
    file: UploadFile, session: Session = Depends(get_session)
):

    image = await file.read()
    image = Image.open(io.BytesIO(image))
    result = model(image)
    names = result[0].names
    probability = result[0].probs.data.numpy()
    pred = np.argmax(probability)

    db_classif = Classification(
        prediction=names[pred], confidence=float(probability[pred])
    )

    session.add(db_classif)
    session.commit()
    session.refresh(db_classif)

    return db_classif


@app.get(
    '/classifications',
    status_code=HTTPStatus.OK,
    response_model=ClassificationList,
)
def get_classifications_list(
    limit: int = 10,
    skip: int = 0,
    session: Session = Depends(get_session),
):
    classif = session.scalars(select(Classification).limit(limit).offset(skip))
    return {'classifications': classif}


@app.get('statistics', status_code=HTTPStatus.OK)
def get_statistics(): ...
