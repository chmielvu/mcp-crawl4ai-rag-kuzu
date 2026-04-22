"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    host: str = "0.0.0.0"
    port: int = 8051
    transport: str = "sse"

    mistral_api_key: str
    model_choice: str = "mistral-small-latest"
    embedding_model: str = "mistral-embed"
    embedding_dimensions: int = 1024

    kuzu_db_path: str = "./data/kuzu_db"

    use_contextual_embeddings: bool = False
    use_hybrid_search: bool = False
    use_reranking: bool = False
    use_agentic_rag: bool = False

    default_max_depth: int = 3
    default_max_concurrent: int = 10
    default_chunk_size: int = 5000
    default_overlap: int = 200

    default_num_results: int = 5
    default_semantic_threshold: float = 0.5
    default_rerank_threshold: float = 0.3

    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_cache_dir: str = "./data/flashrank_cache"
    reranker_max_length: int = 512

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate_required_fields(self) -> None:
        """Validate the required settings."""
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is required")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be greater than 0")


settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global settings
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]
        settings.validate_required_fields()
    return settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global settings
    settings = None
