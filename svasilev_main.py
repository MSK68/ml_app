from transformers import pipeline

with pipeline("image-to-text", model="dumperize/movie-picture-captioning") as pipe:
    print(pipe("http://images.cocodataset.org/val2017/000000039769.jpg"))