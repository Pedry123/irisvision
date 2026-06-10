from http import HTTPStatus

from fastapi import Depends, FastAPI, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from irisvision.database import get_session
from irisvision.models import Classification
from irisvision.schemas import ClassificationList, ClassificationPublic
from irisvision.utils import predict_image

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
    image_bytes = await file.read()
    return await predict_image(image_bytes, session)


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
