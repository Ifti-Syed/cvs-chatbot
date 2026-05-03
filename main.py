"""
CVS Chatbot — project entry point.

Run from the project root:
    python main.py                     # development (auto-reload)
    uvicorn backend.app:app            # production-style local
    uvicorn backend.app:app --host 0.0.0.0 --port 8000  # Docker / Cloud Run

backend/app.py   → FastAPI *application factory* (creates the `app` object)
main.py          → *entry point* (tells uvicorn to run that app)
"""

import sys
from pathlib import Path

# Guarantee the project root is on sys.path so all `backend.*` imports resolve
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

