FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /engine/stockfish

EXPOSE 5000

CMD ["python", "app.py"]
