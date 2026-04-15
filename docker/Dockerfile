FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN useradd -m -u 1000 chatbot

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Transfer ownership to non-root user
RUN chown -R chatbot:chatbot /app
USER chatbot

# Liveness probe — stdlib only, no extra dependencies required
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Single async worker is correct for FastAPI + asyncio.
# Cloud Run scales horizontally (more instances), not vertically (more workers).
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
