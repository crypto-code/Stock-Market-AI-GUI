# set base image (host OS)
FROM python:3.6.2

# copy the content of the local src directory to the working directory
COPY ./ .

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r docReq.txt

RUN apt update -y
RUN apt install -y vim

RUN python manage.py migrate 

# command to run on container start
CMD [ "python", "manage.py", "runserver" ]
