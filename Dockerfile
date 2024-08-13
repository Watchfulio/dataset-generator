ARG BASE_IMAGE=python:3.11
FROM $BASE_IMAGE
ARG BASE_IMAGE=python:3.11

COPY ["bin/python-setup", "bin/pip-install", "/usr/local/bin/"]
RUN python-setup $BASE_IMAGE

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip-install $BASE_IMAGE

COPY . .
