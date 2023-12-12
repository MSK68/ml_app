import io
import os
import uvicorn
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline

# Создаем экземпляр класса FastAPI
app = FastAPI()

# Создаем класс для хранения данных о изображении
class ImageData(BaseModel):
    image: Optional[bytes] = None

# Создаем экземпляр класса pipeline для распознавания изображения
pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Создаем функцию для загрузки изображения
def load_image(image: ImageData) -> Image.Image:
    # Получаем изображение
    image_data = image.image
    # Возвращаем изображение в формате PIL
    return Image.open(io.BytesIO(image_data))

# Создаем функцию для распознавания изображения
def recognize_image(image: Image.Image) -> str:
    # Запускаем распознавание изображения средствами модели
    x = pipe(image)
    # Возвращаем результаты распознавания
    return x[0]['generated_text']

# Создаем функцию для обработки пост запросов
@app.post("/predict")
async def predict(image: ImageData):
    # Загружаем изображение
    img = load_image(image)
    # Распознаем изображение
    result = recognize_image(img)
    # Возвращаем результаты распознавания
    return {"result": result}

# Запускаем сервер
if __name__ == "__main__":
    uvicorn.run(app, host='mlapp.2118281-mb64895.twc1.net', port=8000, log_level='info')

# Запускаем сервер с помощью команды uvicorn main:app --reload
# Переходим по адресу http://mlapp.2118281-mb64895.twc1.net:8000/docs
# Переходим в раздел POST /predict
# Нажимаем кнопку Try it out
# Выбираем изображение для загрузки
# Нажимаем кнопку Execute
# Получаем результаты распознавания изображения
