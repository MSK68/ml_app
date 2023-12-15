from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline(task="text-classification", model="papluca/xlm-roberta-base-language-detection", top_k=None)

@app.get("/")
def root():
    return {"message": "Модель анализа текста"}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text )[0]
