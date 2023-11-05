from transformers import pipeline

pipe = pipeline("image-to-text", model="dumperize/movie-picture-captioning")

