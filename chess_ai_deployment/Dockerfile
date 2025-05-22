FROM python:3.10-slim AS builder

WORKDIR /app

# Install only what's needed for building
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Second stage - copy only what's needed for runtime
FROM python:3.10-slim

WORKDIR /app

# Copy only the necessary files
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

# Make sure gunicorn is installed in the final image
RUN pip install --no-cache-dir gunicorn==20.1.0 eventlet==0.33.0

# Clean up unnecessary files to reduce image size
RUN find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -delete

# Use the PORT environment variable that Railway sets
EXPOSE $PORT
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:$PORT", "chess_ai_api:socketio"]
