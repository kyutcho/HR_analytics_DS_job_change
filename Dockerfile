FROM python:3.9.16-slim-buster

ENV MLFLOW_BACKEND_STORE_URI ""
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT ""
ENV MLFLOW_HOST 0.0.0.0
ENV MLFLOW_PORT 8000:8888

WORKDIR /

COPY . /

RUN pip install --upgrade pip & pip install -r requirements.txt

CMD python src/main.py