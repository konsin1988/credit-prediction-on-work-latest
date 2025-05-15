FROM postgres:13

WORKDIR /postgres

COPY ./docker/postgres/requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt
