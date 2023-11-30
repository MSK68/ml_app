import gradio as gr
from transformers import pipeline
import re



text_summarizer_pipeline = pipeline('summarization', model="d0rj/rut5-base-summ")



def greetMe(articleInput, min_length_of_article, max_length_of_article):

    articleToBeSummarized = articleInput

    summarizedArticle = text_summarizer_pipeline(articleToBeSummarized, min_length=min_length_of_article, max_length=max_length_of_article, do_sample=False)

    return summarizedArticle[0]['summary_text']



demo = gr.Interface(fn=greetMe, 
                     title="Суммаризация текста",
                     inputs=[
                         gr.Textbox(lines=20, placeholder="Введите статью для того чтобы сократить ее", label="Ввод текста", interactive=True),
                         gr.Slider(10, 100, value=10, step=10, label="Минимальное колличество слов в сокращении", info="Выберите от 10 до 100"),
                         gr.Slider(120, 250, value=120, step=10, label="Максимальное колличество слов в сокращении", info="Выберите от 120 and 250"),
                     ],
                     outputs=gr.Textbox(lines=20, label="Выход", show_copy_button=True),
                     allow_flagging='never')


if __name__=="__main__":
    demo.launch(show_api=False)