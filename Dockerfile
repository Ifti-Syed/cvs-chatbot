FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
