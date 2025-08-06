# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies for Cloud Run
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Networking tools for health checks
    curl \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
RUN mkdir /app
WORKDIR /app

# Copy Cloud Run requirements file first for caching
COPY requirements-cloudrun.txt .

# Install Python dependencies (Cloud Run compatible - no audio/video)
RUN pip install --no-cache-dir -r requirements-cloudrun.txt

# Copy the rest of the application code
COPY . .

# Create directories for output files
RUN mkdir -p /app/videos /app/audio /app/reports

# Expose port (Cloud Run will assign PORT dynamically)
EXPOSE 8000

# Command to run the application (use PORT environment variable for Cloud Run)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
