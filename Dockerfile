# RT Forecast - Production Dockerfile
# ====================================
# Bay Area Airspace Congestion Forecasting Service

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy/pandas compilation if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models output

# Expose port (Railway/Render will set PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:${PORT:-8000}/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# Run the server
# Note: $PORT is provided by Railway/Render at runtime
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
