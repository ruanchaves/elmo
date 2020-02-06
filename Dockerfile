FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .