import io
import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_model() -> pipeline:
	'''Загрузка модели распозноавние и описания изображений'''
	
	return pipeline("image-to-text", model="dumperize/movie-picture-captioning")


def load_image() -> Image.Image:

	"""Загрузка изображения для создания описания"""

	uploaded_file = st.file_uploader(label='Выберите изображение для создания описания в стиле киноафиши')
	if uploaded_file is not None:
		#Получение изображения
		image_data = uploaded_file.getvalue()
		#Демонстрация изображения на WEB-странице
		st.image(image_data)
		return Image.open(io.BytesIO(image_data))
	else:
		return None

#Заголовок страницы
st.title ('Генерация описания в стиле киноафиш')
#формы загрузки изображения
img = load_image()
#Загрузка модели
pipe = load_model()
#Кнопка запуска
result = st.button('Дать описание изображению')

#Если кнопка нажата, запускаем генерацию описания
if result:
	# Запускаем распознавание изображения средствами модели
	x = pipe(img)
	# Выводим заголовок
	st.write('**Результат**')
	#Выводим описание
	st.write(x[0]['generated_text'])

