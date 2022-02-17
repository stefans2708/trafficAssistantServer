FROM bitnami/minideb:latest

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y  #for opencv
RUN pip3 install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 9990

ENTRYPOINT ["python3"]
CMD ["main.py"]