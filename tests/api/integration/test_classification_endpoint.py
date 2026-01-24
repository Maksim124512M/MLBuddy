import os

from fastapi.testclient import TestClient

from api.main import app

def test_classification_endpoint():
    """
    Test the regression endpoint of the API.
    """

    client = TestClient(app)

    file_path = os.path.join('tests', 'data', 'Titanic-Dataset.csv')
    response = client.post('http://localhost:8000/v2/classification/train/', json={
        'df_path': file_path, 
        'target': 'Survived'})

    assert response.status_code == 200
    assert type(response.json()) is dict