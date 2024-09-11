FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit","run" ]
CMD ["Week2.py"]
