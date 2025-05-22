FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Use the PORT environment variable that Railway sets
EXPOSE $PORT
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT 'chess_ai_api:socketio'