FROM python:3.9

# 
WORKDIR /code

# 
COPY /dolgovd/requirements.txt /code/requirements.txt

# 
RUN pip install -r requirements.txt

# 
COPY /dolgovd/main.py /code/main.py

EXPOSE 8501

# 
CMD ["streamlit", "run", "main.py"]