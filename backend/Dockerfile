# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Create and set permissions for uploads directory
RUN mkdir -p /code/uploads && \
    chmod -R 777 /code/uploads

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory
COPY . /code/

# Run migrations and start the application
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 