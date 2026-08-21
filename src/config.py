"""Strict application settings for the remote Crawl4AI/FalkorDB stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_REMOVED_ENV_NAMES = frozenset(
    {
        "KUZU_DB_PATH",
        "CRAWL4AI_BASE_DIRECTORY",
        "PLAYWRIGHT_BROWSERS_PATH",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "RERANKER_MODEL",
        "RERANKER_CACHE_DIR",
        "RERANKER_MAX_LENGTH",
        "FLASHRANK_MODEL",
        "FLASHRANK_CACHE_DIR",
        "FLASHRANK_MAX_LENGTH",
    }
)


class Settings(BaseSettings):
    """Application settings with no local-provider compatibility aliases."""

    host: str = "0.0.0.0"
    port: int = 8051
    transport: str = "sse"

    mistral_api_key: str = ""
    model_choice: str = "mistral-small-latest"

    crawl4ai_base_url: str = "http://localhost:11235"
    crawl4ai_api_token: str = ""
    crawl4ai_timeout_seconds: float = 60.0
    crawl4ai_max_batch_size: int = 100

    falkordb_url: str = "falkor://localhost:6380"
    falkordb_graph: str = "crawl-graph"
    falkordb_query_timeout_ms: int = 1000
    falkordb_max_connections: int = 16

    unified_ml_base_url: str = "http://localhost:8000"
    unified_ml_embed_model: str = "intfloat/multilingual-e5-small"
    unified_ml_embedding_dimensions: int = 384
    unified_ml_timeout_seconds: float = 30.0
    unified_ml_batch_size: int = 32

    use_gliner_metadata: bool = True
    gliner_entity_labels: str = "product,technology,library,organization,person"
    gliner_relation_labels: str = "uses,depends_on,implements,stores"
    gliner_threshold: float = 0.5
    gliner_include_confidence: bool = True
    gliner_include_spans: bool = True

    use_langextract_metadata: bool = False
    langextract_model_id: str = "mistral-small-latest"
    langextract_base_url: str = "https://api.mistral.ai/v1"
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 4
    langextract_max_char_buffer: int = 2000

    use_contextual_embeddings: bool = False
    use_hybrid_search: bool = False
    use_reranking: bool = False
    use_agentic_rag: bool = True

    default_max_depth: int = 3
    default_max_concurrent: int = 10
    default_chunk_size: int = 5000
    default_overlap: int = 200
    default_num_results: int = 5
    default_semantic_threshold: float = 0.0

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    _removed_env_names: ClassVar[frozenset[str]] = _REMOVED_ENV_NAMES

    @field_validator("falkordb_graph")
    @classmethod
    def validate_graph_name(cls, value: str) -> str:
        """Keep the breaking release on its dedicated graph name."""

        if value != "crawl-graph":
            raise ValueError("FALKORDB_GRAPH must be exactly 'crawl-graph'")
        return value

    @field_validator("unified_ml_embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        """Require the deployed Unified-ML embedding width."""

        if value != 384:
            raise ValueError("UNIFIED_ML_EMBEDDING_DIMENSIONS must be exactly 384")
        return value

    def validate_required_fields(self) -> None:
        """Fail fast on missing credentials and removed legacy settings."""

        if not self.mistral_api_key.strip():
            raise ValueError("MISTRAL_API_KEY is required")
        if self.crawl4ai_timeout_seconds <= 0:
            raise ValueError("CRAWL4AI_TIMEOUT_SECONDS must be greater than 0")
        if self.crawl4ai_max_batch_size < 1 or self.crawl4ai_max_batch_size > 100:
            raise ValueError("CRAWL4AI_MAX_BATCH_SIZE must be between 1 and 100")
        if self.falkordb_query_timeout_ms <= 0:
            raise ValueError("FALKORDB_QUERY_TIMEOUT_MS must be greater than 0")
        if self.falkordb_max_connections <= 0:
            raise ValueError("FALKORDB_MAX_CONNECTIONS must be greater than 0")
        if self.unified_ml_timeout_seconds <= 0:
            raise ValueError("UNIFIED_ML_TIMEOUT_SECONDS must be greater than 0")
        if self.unified_ml_batch_size <= 0:
            raise ValueError("UNIFIED_ML_BATCH_SIZE must be greater than 0")
        if self.langextract_extraction_passes < 1:
            raise ValueError("LANGEXTRACT_EXTRACTION_PASSES must be at least 1")
        if self.langextract_max_workers < 1:
            raise ValueError("LANGEXTRACT_MAX_WORKERS must be at least 1")
        if self.langextract_max_char_buffer < 1:
            raise ValueError("LANGEXTRACT_MAX_CHAR_BUFFER must be at least 1")

        removed = sorted(_present_removed_env_names(self._removed_env_names))
        if removed:
            joined = ", ".join(removed)
            raise ValueError(f"Removed environment variables are not supported: {joined}")

    @property
    def gliner_entities(self) -> tuple[str, ...]:
        """Return configured GLiNER entity labels."""

        return _split_labels(self.gliner_entity_labels)

    @property
    def gliner_relations(self) -> tuple[str, ...]:
        """Return configured GLiNER relation labels."""

        return _split_labels(self.gliner_relation_labels)


settings: Settings | None = None
_runtime_root_cache: Path | None = None


def get_runtime_root() -> Path:
    """Resolve and cache the runtime project root."""

    global _runtime_root_cache
    if _runtime_root_cache is not None:
        return _runtime_root_cache

    configured = os.environ.get("MCP_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        cwd = Path.cwd().resolve()
        root = next(
            (parent for parent in (cwd, *cwd.parents) if (parent / "pyproject.toml").exists()),
            cwd,
        )
    os.environ.setdefault("MCP_PROJECT_ROOT", str(root))
    _runtime_root_cache = root
    return root


def _discover_env_files() -> tuple[Path, ...]:
    """Return the explicit env file or the normal runtime candidates."""

    candidates: list[Path] = []
    explicit = os.getenv("MCP_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        candidates.extend((get_runtime_root() / ".env", Path.cwd() / ".env"))

    if not explicit and not os.getenv("MCP_PROJECT_ROOT"):
        module_path = Path(__file__).resolve()
        for parent in (module_path, *module_path.parents):
            if (parent / "pyproject.toml").exists():
                candidates.append(parent / ".env")
                break

    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if resolved.exists() and key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def _present_removed_env_names(names: frozenset[str]) -> set[str]:
    """Find removed names in process env and discovered dotenv files."""

    present = {name for name in names if name in os.environ}
    for env_file in _discover_env_files():
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key = line.split("=", 1)[0].strip().removeprefix("export ")
                if key in names:
                    present.add(key)
        except OSError:
            continue
    return present


def _split_labels(value: str) -> tuple[str, ...]:
    """Normalize comma-separated label settings."""

    return tuple(label.strip() for label in value.split(",") if label.strip())


def get_settings() -> Settings:
    """Load, validate, and cache settings for the application lifespan."""

    global settings
    if settings is None:
        settings = Settings(_env_file=_discover_env_files())  # type: ignore[call-arg]
        settings.validate_required_fields()
    return settings


def reset_settings() -> None:
    """Reset settings and root caches for isolated tests."""

    global settings, _runtime_root_cache
    settings = None
    _runtime_root_cache = None
