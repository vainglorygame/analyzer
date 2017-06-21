FROM python:3.6-alpine

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN apk add --no-cache git && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir git+git://github.com/zzzeek/sqlalchemy

CMD ["python", "worker.py"]
