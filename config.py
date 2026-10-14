import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LiveKit Configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # LLM Configuration (Mistral)
    # NOTE: free-tier keys cannot use mistral-large / medium.
    # Use mistral-small-latest (free tier allowed).
    mistral_api_key: str
    mistral_model: str = "mistral-small-latest"

    # Document Ingestion
    docs_dir: str = "docs"
    do_ocr: bool = False

    # Embedding & Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "telecom_docs"
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval defaults
    default_top_k: int = 5
    score_threshold: float = 0.0
    max_context_chars: int = 6000

    # Voice Configuration
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "en"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Ashley"

    # Web API / UI
    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()

# Set environment variables for LiveKit + Mistral
os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("MISTRAL_API_KEY", settings.mistral_api_key)
