"""
Configuration management for CVS Chatbot backend.
Uses pydantic-settings to load from environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's location (backend/), then check project root too
_HERE = Path(__file__).parent          # backend/
_PROJECT_ROOT = _HERE.parent           # CVS_Chatbot/
_ENV_FILE = _PROJECT_ROOT / ".env" if (_PROJECT_ROOT / ".env").exists() else _HERE / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Settings — default empty so server starts without a key;
    # actual API calls will fail with a clear error if the key is not set.
    openai_api_key: str = ""

    # Qdrant Vector Database Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "cvs_documents"

    # Google Cloud Storage
    gcs_bucket_name: Optional[str] = None
    # Sub-folder inside the bucket where sales PDFs live (e.g. "sales_pdfs")
    gcs_folder_path: str = "sales_pdfs"

    # Model Settings
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Chunking Settings
    # chunk_size: int = 512
    # chunk_overlap: int = 50
    # top_k_results: int = 5
    # max_tokens: int = 1024

    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k_results: int = 3
    max_tokens: int = 800

    # GCS polling — how often to check for new PDFs (seconds). 0 = disabled.
    gcs_poll_interval: int = 300  # 5 minutes default

    # Application Settings
    environment: str = "development"
    port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def qdrant_is_cloud(self) -> bool:
        return self.qdrant_api_key is not None and self.qdrant_api_key.strip() != ""


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
