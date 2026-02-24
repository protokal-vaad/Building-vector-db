"""Application settings — loaded from .env via Pydantic BaseSettings."""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment variables for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Google Cloud
    google_application_credentials: str = "SA.json"
    gcp_project_id: str
    gcp_location: str = "me-west1"
    firestore_database: str = "reyhan-db"

    # Google Cloud Storage
    bucket_name: str
    folder_path: str = ""

    # Gemini model
    model_name: str = "gemini-2.5-pro"

    # Firestore
    firestore_collection: str = "protocols"

    # Embeddings
    embedding_model: str = "models/embedding-001"

    # Logging
    log_level: str = "INFO"

    # Gemini API Key
    google_api_key: Optional[str] = None

