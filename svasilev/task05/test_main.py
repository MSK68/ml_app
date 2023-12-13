from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main() -> None:
    """
    Test the main endpoint
    :return: None
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_custom_img() -> None:
    """
    Test the predict endpoint with a custom image
    :return: None
    """
    response = client.post("/predict/",
                           json={"image_url": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg"})
    json_data = response.json()
    assert response.status_code == 200
    assert response.json() == {"result": "a herd of giraffes and zebras grazing in a field "}


def test_response_length() -> None:
    """
    Test the length of the response
    :return: None
    """
    response = client.post("/predict/",
                           json={
                               "image_url": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg"})
    json_data = response.json()
    assert response.status_code == 200
    assert len(json_data["result"]) > 0