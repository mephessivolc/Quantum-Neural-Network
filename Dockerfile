FROM python:latest

RUN apt-get update 
RUN apt-get install -y python3-dev python3-pip build-essential graphviz 

RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir -p /home/user
WORKDIR /home/user 