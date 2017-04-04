FROM tensorflow/tensorflow:latest-py3

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN pip install virtualenv && virtualenv venv && pip install -r requirements.txt

CMD ["python", "worker.py"]
