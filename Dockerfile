FROM python:3.11

ENV N_WORKER=2

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install libgl1-mesa-glx -y

RUN  pip3 --no-cache-dir install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port 10010 --workers $N_WORKER