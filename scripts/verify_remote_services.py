"""Verification script for remote Crawl4AI and Unified-ML services contract."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import httpx


async def check_crawl4ai(base_url: str, api_token: str = "", timeout: float = 10.0) -> bool:
    """Verify remote Crawl4AI REST service health endpoint."""
    print(f"\n[1/6] Checking Crawl4AI service at {base_url} ...")
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.get("/health", headers=headers)
            if response.status_code != 200:
                print(f"  ✗ Crawl4AI returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            if not isinstance(data, dict):
                print(f"  ✗ Crawl4AI returned non-object JSON: {type(data).__name__}")
                return False
            print(f"  ✓ Crawl4AI health check passed (HTTP 200): {data}")
            return True
    except Exception as exc:
        print(f"  ✗ Crawl4AI connection failed: {exc}")
        return False


async def check_unified_ml_health(
    base_url: str,
    required_models: tuple[str, ...] = (
        "intfloat/multilingual-e5-small",
        "ms-marco-MultiBERT-L-12",
        "fastino/gliner2-multi-v1",
    ),
    timeout: float = 10.0,
) -> bool:
    """Verify Unified-ML health endpoint and required model availability."""
    print(f"\n[2/6] Checking Unified-ML /health at {base_url} ...")
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.get("/health")
            if response.status_code != 200:
                print(
                    f"  ✗ Unified-ML /health returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False
            data = response.json()
            if not isinstance(data, dict):
                print(
                    f"  ✗ Unified-ML /health returned non-object JSON: "
                    f"{type(data).__name__}"
                )
                return False
            info_response = await client.get("/info")
            if info_response.status_code != 200:
                print(
                    f"  ✗ Unified-ML /info returned HTTP {info_response.status_code}: "
                    f"{info_response.text[:200]}"
                )
                return False
            info = info_response.json()
            if not isinstance(info, dict):
                print("  ✗ Unified-ML /info returned non-object JSON")
                return False
            loaded_keys = ("embed_loaded", "rerank_loaded", "ner_loaded")
            if any(data.get(key) is False for key in loaded_keys):
                print("  ✗ Unified-ML reported an unloaded required model")
                return False
            health_repr = f"{data} {info}".lower()
            missing_models = [m for m in required_models if m.lower() not in health_repr]
            if missing_models:
                print(f"  ✗ Unified-ML model metadata missing: {missing_models}")
                return False
            return True
    except Exception as exc:
        print(f"  ✗ Unified-ML /health connection failed: {exc}")
        return False


async def check_unified_ml_info(
    base_url: str,
    expected_model: str = "intfloat/multilingual-e5-small",
    expected_dim: int = 384,
    timeout: float = 10.0,
) -> bool:
    """Verify Unified-ML /info endpoint reports exact expected embedding model and dimensions."""
    print(f"\n[3/6] Checking Unified-ML /info at {base_url} ...")
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.get("/info")
            if response.status_code != 200:
                print(f"  ✗ Unified-ML /info returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            if not isinstance(data, dict):
                print(f"  ✗ Unified-ML /info returned non-object JSON: {type(data).__name__}")
                return False

            model = data.get("embedding_model") or data.get("embed_model") or data.get("model")
            if not model or model != expected_model:
                print(f"  ✗ Embedding model contract violation: expected {expected_model!r}, got {model!r}")
                return False

            raw_dim = (
                data.get("embedding_dimensions")
                or data.get("embedding_dimension")
                or data.get("dimensions")
            )
            if raw_dim is None:
                print(
                    "  ! Unified-ML /info omits dimensions; "
                    "the /embeddings check will enforce 384 dimensions"
                )
                return True
            try:
                actual_dim = int(raw_dim)
            except (TypeError, ValueError):
                print(f"  ✗ Invalid embedding dimensions value in /info: {raw_dim!r}")
                return False

            if actual_dim != expected_dim:
                print(f"  ✗ Dimension mismatch: expected {expected_dim}, got {actual_dim}")
                return False

            print(f"  ✓ Unified-ML info contract verified (model: {model}, dimensions: {actual_dim})")
            return True
    except Exception as exc:
        print(f"  ✗ Unified-ML /info connection failed: {exc}")
        return False


async def check_unified_ml_embeddings(
    base_url: str,
    model: str = "intfloat/multilingual-e5-small",
    expected_dim: int = 384,
    timeout: float = 15.0,
) -> bool:
    """Verify Unified-ML /embeddings endpoint generates 384-dimensional vectors."""
    print("\n[4/6] Checking Unified-ML /embeddings (384-dim) ...")
    payload: dict[str, Any] = {
        "texts": ["passage: Verification probe for remote services contract"],
    }
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.post("/embeddings", json=payload)
            if response.status_code != 200:
                print(f"  ✗ Unified-ML /embeddings returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            if not isinstance(data, dict):
                print("  ✗ Unified-ML /embeddings returned non-object JSON")
                return False

            if data.get("model") != model or int(data.get("dimensions", -1)) != expected_dim:
                print(
                    "  ✗ Embeddings metadata mismatch: "
                    f"expected model={model!r}, dimensions={expected_dim}; "
                    f"got model={data.get('model')!r}, dimensions={data.get('dimensions')!r}"
                )
                return False
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                print("  ✗ Embeddings response missing or empty 'embeddings' list")
                return False
            first_vector = embeddings[0]
            if not isinstance(first_vector, list) or len(first_vector) != expected_dim:
                actual_dim = len(first_vector) if isinstance(first_vector, list) else "non-list"
                print(f"  ✗ Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
                return False

            print(f"  ✓ Embeddings generated successfully ({len(embeddings)} vector, {expected_dim} dimensions)")
            return True
    except Exception as exc:
        print(f"  ✗ Unified-ML /embeddings connection failed: {exc}")
        return False


async def check_unified_ml_rerank(base_url: str, timeout: float = 15.0) -> bool:
    """Verify Unified-ML /rerank endpoint returns a validated results list."""
    print("\n[5/6] Checking Unified-ML /rerank ...")
    payload: dict[str, Any] = {
        "query": "web crawling framework",
        "texts": [
            "Crawl4AI is an open-source web crawler designed for AI workflows.",
            "PostgreSQL is a relational database management system.",
        ],
    }
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.post("/rerank", json=payload)
            if response.status_code != 200:
                print(f"  ✗ Unified-ML /rerank returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            if not isinstance(data, dict):
                print("  ✗ Unified-ML /rerank returned non-object JSON")
                return False

            results = data.get("results")
            if not isinstance(results, list) or len(results) == 0:
                print(f"  ✗ Rerank response missing non-empty 'results' list: {data}")
                return False

            print(f"  ✓ Rerank passed ({len(results)} candidate results scored)")
            return True
    except Exception as exc:
        print(f"  ✗ Unified-ML /rerank connection failed: {exc}")
        return False


async def check_unified_ml_extract(base_url: str, timeout: float = 15.0) -> bool:
    """Verify Unified-ML /extract endpoint sends entities/relations and validates relation_extraction."""
    print("\n[6/6] Checking Unified-ML /extract (GLiNER) ...")
    payload: dict[str, Any] = {
        "text": "FastMCP is a Python library developed by Anthropic for Model Context Protocol tools.",
        "entities": ["technology", "product", "library", "organization"],
        "relations": ["uses", "depends_on", "implements", "stores"],
        "threshold": 0.5,
    }
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            response = await client.post("/extract", json=payload)
            if response.status_code != 200:
                print(f"  ✗ Unified-ML /extract returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            if not isinstance(data, dict):
                print("  ✗ Unified-ML /extract returned non-object JSON")
                return False
            results = data.get("results")
            if not isinstance(results, dict):
                print("  ✗ Extraction response missing 'results' dictionary")
                return False
            entities = results.get("entities")
            relation_extraction = results.get("relation_extraction")
            if not isinstance(entities, (list, dict)) or not isinstance(
                relation_extraction, (list, dict)
            ):
                print(
                    "  ✗ Extraction contract violation: 'results' must contain "
                    "typed 'entities' and 'relation_extraction' collections"
                )
                return False
            entity_count = len(entities)
            relation_count = len(relation_extraction)

            print(
                f"  ✓ GLiNER extraction contract verified ({entity_count} entity groups, "
                f"{relation_count} relation groups)"
            )
            return True
    except Exception as exc:
        print(f"  ✗ Unified-ML /extract connection failed: {exc}")
        return False


async def main() -> int:
    """Run all remote service contract verification checks."""
    crawl4ai_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:11235")
    crawl4ai_token = os.getenv("CRAWL4AI_API_TOKEN", "")
    unified_ml_url = os.getenv("UNIFIED_ML_BASE_URL", "http://localhost:8000")
    embed_model = os.getenv("UNIFIED_ML_EMBED_MODEL", "intfloat/multilingual-e5-small")
    embed_dim = int(os.getenv("UNIFIED_ML_EMBEDDING_DIMENSIONS", "384"))

    print("=" * 60)
    print("Remote Services Contract Verification")
    print("=" * 60)
    print(f"Crawl4AI Base URL : {crawl4ai_url}")
    print(f"Unified-ML Base URL: {unified_ml_url}")
    print(f"Embedding Model    : {embed_model} ({embed_dim}-dim)")
    print("=" * 60)

    results: list[bool] = []
    results.append(await check_crawl4ai(crawl4ai_url, crawl4ai_token))
    results.append(await check_unified_ml_health(unified_ml_url))
    results.append(await check_unified_ml_info(unified_ml_url, expected_model=embed_model, expected_dim=embed_dim))
    results.append(await check_unified_ml_embeddings(unified_ml_url, model=embed_model, expected_dim=embed_dim))
    results.append(await check_unified_ml_rerank(unified_ml_url))
    results.append(await check_unified_ml_extract(unified_ml_url))

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Verification Summary: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("✓ All remote service contracts are satisfied.")
        return 0
    else:
        print("✗ One or more remote service contract checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
