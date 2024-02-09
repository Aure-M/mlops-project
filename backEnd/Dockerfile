FROM python:3.8-slim

WORKDIR /app 

COPY . /app 

RUN pip install --no-cache-dir -r requirements.txt 

RUN python3 setup.py

RUN python3 model_preparation.py

ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["flask", "run"]