FROM python:3.10


WORKDIR /workspace

ADD requirements.txt app.py xgboost-model.pkl /workspace/

RUN pip install -r /workspace/requirements.txt
EXPOSE 7860
CMD ["python", "/workspace/app.py"]