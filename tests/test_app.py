from http import HTTPStatus
from pathlib import Path


def test_health(client):
    response = client.get('/health')

    assert response.status_code == HTTPStatus.OK


def test_json_post_predictions(client):

    image_path = Path(__file__).parent / 'images' / 'test1.jpg'
    image_bytes = image_path.read_bytes()
    files = {'file': ('test1.jpg', image_bytes, 'image/jpeg')}

    response = client.post('/predict', files=files)
    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    assert payload['prediction']
    assert payload['confidence']


def test_get_classifications(client):

    response = client.get('/classifications')

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'classifications': []}
