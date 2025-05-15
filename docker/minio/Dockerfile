FROM minio/minio:latest 

WORKDIR /minio

COPY ./docker/minio/requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt
