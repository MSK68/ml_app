import io
import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_model() -> pipeline:

    """Загрузка модели для распознавания изображения и генерации описания"""

    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def load_image() -> Image.Image:

    """Загрузка изображения для распознавания и генерации описания"""

    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(label='Выберите изображение для генерации описания.')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# Выводим заголовок страницы средствами Streamlit
st.title('Описание изображения')
# Вызываем функцию создания формы загрузки изображения
img = load_image()
# Вызываем функцию загрузки модели
pipe = load_model()

# Создаем кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')

# Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Запускаем распознавание изображения средствами модели
    x = pipe(img)
    # Выводим заголовок средствами Streamlit
    # используя форматирование Markdown
    st.write('**Результаты распознавания и генерации описания:**')
    # Выводим результаты распознавания
    st.write(x[0]['generated_text'])