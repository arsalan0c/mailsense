FROM python:3.6
COPY . .
RUN pip install -r ./requirements.txt
ENTRYPOINT ["python", "./subscriber.py"]
