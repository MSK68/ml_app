"""Реализация сервиса для распознавания изображений с помощью модели ViT-GPT2"""

# Подключаем библиотеки

import uvicorn
from PIL import Image
import requests
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline

# Создаем экземпляр класса FastAPI
app = FastAPI()


# Создаем класс для хранения данных об изображении
class ImageData(BaseModel):
    """Класс для хранения данных об изображении"""
    image_url: str = None


# Создаем экземпляр класса pipeline для распознавания изображения
pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


# Создаем функцию для загрузки изображения
def load_image(image: ImageData) -> Image.Image:
    """
    Функция для загрузки изображения
    :param image: ссылка на изображение
    :return: изображение в формате PIL
    """
    # Получаем изображение
    image_data = image.image_url
    # Возвращаем изображение в формате PIL
    return Image.open(requests.get(image_data, stream=True).raw)


# Создаем функцию для распознавания изображения
def recognize_image(image: Image.Image) -> str:
    """
    Функция для распознавания изображения
    :param image: изображение в формате PIL
    :return: результаты распознавания
    """
    # Распознаем изображение
    result = pipe(image)
    # Возвращаем результаты распознавания
    return result[0]["generated_text"]


# Создаем функцию для обработки get запросов
@app.get("/")
async def root():  # Асинхронность для ускорения работы и возможности обработки нескольких запросов
    """
    Функция для обработки get запросов
    :return: приветственное сообщение
    """
    # Возвращаем приветственное сообщение
    return {"message": "Hello World"}


# Создаем функцию для обработки пост запросов
@app.post("/predict")
async def predict(image: ImageData):  # Асинхронность для ускорения работы и возможности обработки нескольких запросов
    """
    Функция для обработки пост запросов
    :param image: изображение в формате PIL
    :return: результаты распознавания
    """
    # Загружаем изображение
    img = load_image(image)
    # Распознаем изображение
    result = recognize_image(img)
    # Возвращаем результаты распознавания
    return {"result": result}


# Запускаем сервер
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
