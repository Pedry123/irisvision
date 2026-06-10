from pydantic import BaseModel


class ClassificationSchema(BaseModel):
    prediction: str
    confidence: float


class ClassificationPublic(ClassificationSchema):
    id: int


class ClassificationList(BaseModel):
    classifications: list[ClassificationPublic]
