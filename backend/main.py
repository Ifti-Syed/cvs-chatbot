"""
CVS Chatbot - entry point.

Run directly to start the development server:
    python backend/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'backend.*' imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from backend.config import settings

if __name__ == "__main__":
    host = "0.0.0.0" if settings.is_production else "127.0.0.1"
    uvicorn.run(
        "backend.app:app",
        host=host,
        port=settings.port,
        reload=not settings.is_production,
        log_level="debug" if not settings.is_production else "info",
    )
