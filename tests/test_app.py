from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

from irisvision.app import app


def test_json_post_predictions():
    client = TestClient(app)

    image_path = Path(__file__).parent / 'images' / 'test1.jpg'
    image_bytes = image_path.read_bytes()
    files = {'file': ('test1.jpg', image_bytes, 'image/jpeg')}

    response = client.post('/predict', files=files)
    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    assert payload['Prediction']
    assert payload['Confidence']
