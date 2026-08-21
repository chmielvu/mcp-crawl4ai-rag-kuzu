"""Tests for application settings, strict legacy rejection, and environment loading."""

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from crawl4ai_mcp.config import Settings, get_settings, reset_settings


def _make_runtime_root() -> Path:
    runtime_root = Path.cwd() / ".agent" / "test-runtime" / uuid4().hex
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def test_get_settings_loads_dotenv_from_runtime_root(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_settings()
    runtime_root = _make_runtime_root()
    env_file = runtime_root / ".env"
    env_file.write_text("MISTRAL_API_KEY=dotenv-key\n", encoding="utf-8")

    monkeypatch.chdir(runtime_root)
    monkeypatch.setenv("MCP_PROJECT_ROOT", str(runtime_root))
    monkeypatch.delenv("MCP_ENV_FILE", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    for env_name in Settings._removed_env_names:
        monkeypatch.delenv(env_name, raising=False)

    loaded = get_settings()
    assert loaded.mistral_api_key == "dotenv-key"
    assert loaded.falkordb_graph == "crawl-graph"
    assert loaded.unified_ml_embedding_dimensions == 384
    reset_settings()


def test_get_settings_respects_explicit_mcp_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_settings()
    runtime_root = _make_runtime_root()
    custom_env = runtime_root / "custom.env"
    custom_env.write_text(
        "MISTRAL_API_KEY=explicit-custom-key\nCRAWL4AI_TIMEOUT_SECONDS=45.0\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MCP_ENV_FILE", str(custom_env))
    monkeypatch.setenv("MCP_PROJECT_ROOT", str(runtime_root))
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    for env_name in Settings._removed_env_names:
        monkeypatch.delenv(env_name, raising=False)

    loaded = get_settings()
    assert loaded.mistral_api_key == "explicit-custom-key"
    assert loaded.crawl4ai_timeout_seconds == 45.0
    reset_settings()


def test_legacy_removed_env_vars_rejected_in_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in sorted(Settings._removed_env_names):
        reset_settings()
        monkeypatch.setenv("MISTRAL_API_KEY", "valid-key")
        monkeypatch.setenv(env_name, "deprecated_value")

        s = Settings()
        with pytest.raises(ValueError, match="Removed environment variables are not supported"):
            s.validate_required_fields()
        reset_settings()


def test_legacy_removed_env_vars_rejected_in_dotenv_file(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in sorted(Settings._removed_env_names):
        reset_settings()
        runtime_root = _make_runtime_root()
        env_file = runtime_root / ".env"
        env_file.write_text(
            f"MISTRAL_API_KEY=valid-key\n{env_name}=deprecated_value\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(runtime_root)
        monkeypatch.setenv("MCP_PROJECT_ROOT", str(runtime_root))
        monkeypatch.delenv("MCP_ENV_FILE", raising=False)

        with pytest.raises(ValueError, match="Removed environment variables are not supported"):
            get_settings()
        reset_settings()


def test_missing_mistral_api_key_raises_validation_error() -> None:
    s = Settings(mistral_api_key="")
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is required"):
        s.validate_required_fields()


def test_invalid_crawl4ai_timeout_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", crawl4ai_timeout_seconds=0.0)
    with pytest.raises(ValueError, match="CRAWL4AI_TIMEOUT_SECONDS must be greater than 0"):
        s.validate_required_fields()


def test_invalid_crawl4ai_max_batch_size_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", crawl4ai_max_batch_size=0)
    with pytest.raises(ValueError, match="CRAWL4AI_MAX_BATCH_SIZE must be between 1 and 100"):
        s.validate_required_fields()

    s2 = Settings(mistral_api_key="key", crawl4ai_max_batch_size=101)
    with pytest.raises(ValueError, match="CRAWL4AI_MAX_BATCH_SIZE must be between 1 and 100"):
        s2.validate_required_fields()


def test_invalid_falkordb_graph_name_rejected_by_field_validator() -> None:
    with pytest.raises(ValidationError, match="FALKORDB_GRAPH must be exactly 'crawl-graph'"):
        Settings(mistral_api_key="key", falkordb_graph="other-graph")


def test_invalid_falkordb_query_timeout_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", falkordb_query_timeout_ms=0)
    with pytest.raises(ValueError, match="FALKORDB_QUERY_TIMEOUT_MS must be greater than 0"):
        s.validate_required_fields()


def test_invalid_falkordb_connections_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", falkordb_max_connections=0)
    with pytest.raises(ValueError, match="FALKORDB_MAX_CONNECTIONS must be greater than 0"):
        s.validate_required_fields()


def test_invalid_unified_ml_embedding_dimensions_rejected_by_field_validator() -> None:
    with pytest.raises(ValidationError, match="UNIFIED_ML_EMBEDDING_DIMENSIONS must be exactly 384"):
        Settings(mistral_api_key="key", unified_ml_embedding_dimensions=512)


def test_invalid_unified_ml_timeout_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", unified_ml_timeout_seconds=0)
    with pytest.raises(ValueError, match="UNIFIED_ML_TIMEOUT_SECONDS must be greater than 0"):
        s.validate_required_fields()


def test_invalid_unified_ml_batch_size_raises_validation_error() -> None:
    s = Settings(mistral_api_key="key", unified_ml_batch_size=0)
    with pytest.raises(ValueError, match="UNIFIED_ML_BATCH_SIZE must be greater than 0"):
        s.validate_required_fields()


def test_invalid_langextract_settings_raise_validation_error() -> None:
    s1 = Settings(mistral_api_key="key", langextract_extraction_passes=0)
    with pytest.raises(ValueError, match="LANGEXTRACT_EXTRACTION_PASSES must be at least 1"):
        s1.validate_required_fields()

    s2 = Settings(mistral_api_key="key", langextract_max_workers=0)
    with pytest.raises(ValueError, match="LANGEXTRACT_MAX_WORKERS must be at least 1"):
        s2.validate_required_fields()

    s3 = Settings(mistral_api_key="key", langextract_max_char_buffer=0)
    with pytest.raises(ValueError, match="LANGEXTRACT_MAX_CHAR_BUFFER must be at least 1"):
        s3.validate_required_fields()


def test_gliner_label_properties() -> None:
    s = Settings(
        mistral_api_key="key",
        gliner_entity_labels="product, technology, library",
        gliner_relation_labels="uses, depends_on ",
    )
    assert s.gliner_entities == ("product", "technology", "library")
    assert s.gliner_relations == ("uses", "depends_on")
