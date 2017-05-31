FROM tensorflow/tensorflow:1.1.0-rc1-py3

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install virtualenv && virtualenv venv && pip install -r requirements.txt

CMD ["python", "worker.py"]
