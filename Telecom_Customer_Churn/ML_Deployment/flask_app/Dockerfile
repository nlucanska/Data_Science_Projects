FROM python:3.10

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./api .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]