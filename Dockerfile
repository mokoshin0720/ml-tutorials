FROM python:3

WORKDIR /development/ml-tutorials
ADD requirements.txt /development/ml-tutorials
RUN pip install -r requirements.txt