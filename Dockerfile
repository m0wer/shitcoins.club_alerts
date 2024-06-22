FROM python:3.11-slim

VOLUME /data

RUN pip install tenacity aiogram aiohttp beautifulsoup4 pydantic pandas plotly kaleido cloudscraper

COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
