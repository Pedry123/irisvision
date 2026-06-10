import io

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from ultralytics import YOLO

from irisvision.models import Classification


def load_model():
    modelpath = r'models/best.pt'
    model = YOLO(modelpath)

    return model


model = load_model()


async def predict_image(image_bytes: bytes, session: Session):
    image = Image.open(io.BytesIO(image_bytes))
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
