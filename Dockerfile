# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Give execute permission to Stockfish
RUN chmod +x engine/stockfish

# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
