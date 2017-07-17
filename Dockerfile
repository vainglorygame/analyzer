FROM python:3.6-alpine

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN pip install --pre --no-cache-dir -r requirements.txt

CMD ["python", "worker.py"]
