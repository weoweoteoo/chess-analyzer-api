FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN chmod +x engine/stockfish

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
