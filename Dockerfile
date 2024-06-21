FROM python:3.11-slim

VOLUME /data

COPY . /app
WORKDIR /app
RUN pip install tenacity aiogram aiohttp beautifulsoup4 pydantic pandas plotly kaleido cloudscraper

CMD ["python", "main.py"]
