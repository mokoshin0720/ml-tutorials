FROM python:3

WORKDIR /development/ml-tutorials
ADD requirements.txt /development/ml-tutorials
RUN pip install -r requirements.txt
RUN pip3 install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html