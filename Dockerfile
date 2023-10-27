FROM python:3.11-slim

COPY . /app
WORKDIR /app
RUN pip install -e .

CMD ["python", "main.py"]
