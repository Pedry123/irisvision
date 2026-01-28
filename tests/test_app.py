from fastapi.testclient import TestClient

from irisvision.app import app


def test_root_deve_retornar_ola_mundo():
    client = TestClient(app)

    response = client.get('/helloworld')

    assert response.json() == {'message': 'Hello, World!'}
