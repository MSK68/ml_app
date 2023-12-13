from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_custom_img():
    response = client.post("/predict/",
                           json={"image_url": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg"})
    json_data = response.json()
    assert response.status_code == 200
    assert response.json() == {"result": "a herd of giraffes and zebras grazing in a field "}