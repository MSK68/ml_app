import streamlit as st
from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def main():
    st.title("Модель анализа текста")

    # Create an input text box
    input_text = st.text_input("Введите предложение на английском языке", "")

    # Create a button to trigger model inference
    if st.button("Анализировать"):
        # Perform inference using the loaded model
        result = classifier(input_text)
        st.write("Prediction:", result)

if __name__ == "__main__":
    main()