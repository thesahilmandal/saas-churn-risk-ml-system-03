# Use an official lightweight Python base image
FROM python:3.12.1-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to optimize Python execution in Docker
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures logs are output immediately (useful for Docker logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# awscli is strictly required because S3Sync relies on shell-based AWS commands
RUN apt-get update && apt-get install -y \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies safely without storing cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the necessary project directories into the container
# We copy config/ because the DataValidator requires config/schema.yaml
COPY src/ ./src/
COPY data_schema/ ./data_schema/

# Create a non-root user for security best practices (Optional but highly recommended)
RUN useradd -m inference_user
RUN chown -R inference_user:inference_user /app
USER inference_user

# Expose the port that FastAPI will run on
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]