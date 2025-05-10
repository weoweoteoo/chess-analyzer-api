# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask chess matplotlib

# Make port 5000 available
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
