import io
import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
from random import randint

# Переменные
box_width = 2
box_color = []
fill_color = "red"
text_color = "white"
font_size = 20

for i in range(20):
    box_color.append('#%06X' % randint(0, 0xFFFFFF))

def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None



def load_model():
    # Подключаем модель Обноружения предметов (Object Detection)
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    print("Load Model")
    return model

def load_processor():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    print("Load Processor")
    return processor



def process_images(processor, model, image):
    global box_color
    global font_size
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Берем вхождения > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Рисуем на изображении 
    draw= ImageDraw.Draw(image)
    # Выбираем шрифт
    font = ImageFont.truetype("arial.ttf", size=font_size)
    # перебираем все вхождения
    for score, label, box, colors in zip(results["scores"], results["labels"], results["boxes"], box_color):
        box = [round(i, 2) for i in box.tolist()]
        # Пишем лог что нашли
        # вычисляем квадрат с найденым объектом
        x, y, x_max, y_max = box
        # рисуем квадрат на найденом объекте
        draw.rectangle([x, y, x_max, y_max], outline=colors, width=box_width)

        # высчитываем размер текста
        left, top, right, bottom = font.getbbox(model.config.id2label[label.item()])
        text_width = right
        text_height =  bottom

        # рисуем подложку для текста
        draw.rectangle([x, y, x + text_width + 6, y + text_height], fill=fill_color)
        # рисуем текст
        draw.text((x, y), model.config.id2label[label.item()], fill=text_color, font=font)
        st.write(f"Нашли {model.config.id2label[label.item()]} с уверенностью {round(score.item(), 3)} в квадрате {box}")





st.title('Классификации изображений в облаке Streamlit')
image = load_image()
result = st.button('Распознать изображение')
if result:
    processor = load_processor()
    model = load_model()
    st.write('**Результаты распознавания:**')
    process_images(processor, model, image)
    st.image(image)