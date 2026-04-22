<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# What if I would like to fix the reranker and use FlashRerank instead of sentence-transformer? Please research proper config/tuning and provide instruction on replacement.

Full picture. Here is the complete drop-in replacement guide.

***

## FlashRank vs. `sentence-transformers` CrossEncoder — key differences

Before the code, the architectural distinction matters:[^1]


|  | `sentence-transformers` CrossEncoder | FlashRank |
| :-- | :-- | :-- |
| Inference engine | PyTorch `.predict()` | ONNX Runtime (quantised INT8) |
| Output | Raw logits (unbounded) | Normalised scores `[0.0 – 1.0]` |
| API shape | `model.predict([[q, d], ...])` → `ndarray` | `Ranker.rerank(RerankRequest(...))` → `PassageResult` list |
| Thread safety | Not thread-safe (GIL-blocked) | ONNX session is thread-safe |
| Cold load time | ~1.2s (PyTorch model init) | ~0.3s (ONNX session) |
| Inference speed | ~18ms / 10 pairs | ~4ms / 10 pairs |
| Activation fix needed | ✅ Yes — logits require `torch.sigmoid` | ❌ No — scores already normalised |

The normalised `[0, 1]` output is the primary practical reason to switch: the existing `filter_by_threshold(threshold=0.3)` code becomes semantically correct immediately with no further changes.[^1]

***

## Step 1 — Dependencies

```bash
# Remove sentence-transformers if only used for reranking
pip uninstall sentence-transformers

# Install FlashRank
pip install flashrank

# requirements.txt change:
# - sentence-transformers>=2.2.2
# + flashrank>=0.2.10
```

FlashRank has no PyTorch dependency — only `onnxruntime`, `tokenizers`, and `huggingface_hub`.  This trims ~800MB from the Docker image if you were only using `sentence-transformers` for the CrossEncoder.[^1]

***

## Step 2 — Choose a model

| Model name | Size | Speed | Best for |
| :-- | :-- | :-- | :-- |
| `ms-marco-TinyBERT-L-2-v2` | ~4MB | ⚡⚡⚡ | Default nano — fastest, good enough |
| `ms-marco-MiniLM-L-12-v2` | ~34MB | ⚡⚡ | Best quality/speed balance — **recommended** |
| `rank-T5-flan` | ~110MB | ⚡ | Best zero-shot on out-of-domain content |
| `ms-marco-MultiBERT-L-12` | ~150MB | ⚡ | Multilingual (100+ langs) |

For a technical documentation RAG (code, API docs), `ms-marco-MiniLM-L-12-v2` is the right default. It was trained on the same MSMARCO passage corpus as the original `MiniLM-L-6-v2` but with a deeper 12-layer encoder — meaningfully more precise at similar latency.[^1]

***

## Step 3 — Replace `src/utilities/reranking.py`

Full file replacement — no partial edits needed:

```python
"""Reranking utilities — FlashRank ONNX backend."""

import logging
from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest

from crawl4ai_mcp.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker backed by FlashRank ONNX cross-encoder.
    
    Scores are normalised to [0, 1] — no sigmoid activation needed.
    Instantiate once at startup; the ONNX session is thread-safe.
    """

    def __init__(
        self,
        model: Optional[Ranker] = None,
        settings: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        self.model: Optional[Ranker] = model

        if self.model is None and self.settings.use_reranking:
            self.model = Ranker(
                model_name=self.settings.cross_encoder_model,
                cache_dir=self.settings.reranker_cache_dir,
                max_length=self.settings.reranker_max_length,
            )

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """Rerank results using FlashRank cross-encoder.

        Args:
            query: Search query string.
            results: List of result dicts.
            content_key: Dict key holding the passage text.

        Returns:
            Results sorted by rerank_score descending.
            Scores are in [0, 1] — no further normalisation needed.
        """
        if not self.model or not results:
            return results

        try:
            passages = [
                {"id": i, "text": r.get(content_key, "")}
                for i, r in enumerate(results)
            ]
            request = RerankRequest(query=query, passages=passages)
            reranked_passages = self.model.rerank(request)

            # Build id→score map
            score_map = {p.id: p.score for p in reranked_passages}

            for i, result in enumerate(results):
                result["rerank_score"] = float(score_map.get(i, 0.0))

            return sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        except Exception as e:
            logger.error(f"FlashRank reranking failed, returning original order: {e}")
            return results  # graceful degradation

    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: float,
        score_key: str = "rerank_score",
    ) -> List[Dict[str, Any]]:
        """Filter by minimum score. Threshold is semantically valid (0–1 scale)."""
        return [r for r in results if r.get(score_key, 0.0) >= threshold]
```


***

## Step 4 — Update `mcp_server.py` (startup warm-load)

```python
# src/crawl4ai_mcp/mcp_server.py — replace CrossEncoder import and init

# REMOVE:
# from sentence_transformers import CrossEncoder
# reranking_model = CrossEncoder(settings.cross_encoder_model)

# ADD:
from flashrank import Ranker

if settings.use_reranking:
    reranking_model = Ranker(
        model_name=settings.cross_encoder_model,
        cache_dir=settings.reranker_cache_dir,
        max_length=settings.reranker_max_length,
    )
else:
    reranking_model = None
```


***

## Step 5 — Update `perform_rag_query.py` (use pre-warmed model)

This simultaneously fixes the cold-load bug from the previous session:

```python
# src/tools/perform_rag_query.py
# REMOVE the inline model instantiation block entirely:
# from sentence_transformers import CrossEncoder
# model = CrossEncoder(settings.cross_encoder_model)
# reranker = Reranker(model=model, settings=settings)

# REPLACE with:
if settings.use_reranking and results and context.reranking_model:
    reranker = Reranker(model=context.reranking_model, settings=settings)
    results_dict = reranker.rerank_results(query, results_dict)
    results_dict = reranker.filter_by_threshold(
        results_dict, threshold=settings.default_rerank_threshold
    )
```


***

## Step 6 — Config additions

```python
# src/crawl4ai_mcp/config.py — add to Settings

cross_encoder_model: str = Field(
    default="ms-marco-MiniLM-L-12-v2",  # was: cross-encoder/ms-marco-MiniLM-L-6-v2
    env="CROSS_ENCODER_MODEL",
)
reranker_cache_dir: str = Field(
    default="/tmp/flashrank_cache",
    env="RERANKER_CACHE_DIR",
)
reranker_max_length: int = Field(
    default=512,
    env="RERANKER_MAX_LENGTH",
)
```

```bash
# .env
USE_RERANKING=true
CROSS_ENCODER_MODEL=ms-marco-MiniLM-L-12-v2
RERANKER_CACHE_DIR=/tmp/flashrank_cache   # or a volume-mounted path in Docker
RERANKER_MAX_LENGTH=512
DEFAULT_RERANK_THRESHOLD=0.3              # now semantically correct: 30% confidence floor
```

> ⚠️ **`max_length` and chunk size**: FlashRank truncates inputs at `max_length` tokens.  The repo defaults to `chunk_size=5000` characters which is ~1200 tokens — well above 512. Either reduce `chunk_size` to ~1800 chars (~450 tokens) or raise `max_length=768`. Truncation does not error out, but a passage truncated mid-sentence loses its tail silently. With documentation RAG (structured headers, code blocks) the head of a chunk is usually most informative, so 512 is acceptable — just be aware of it.[^1]
<span style="display:none">[^10][^11][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/PrithivirajDamodaran/FlashRank

[^2]: https://docs.langchain.com/oss/python/integrations/retrievers/flashrank-reranker

[^3]: https://pypi.org/project/rerankers/

[^4]: https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/

[^5]: https://rankify.readthedocs.io/en/latest/api/rerankings/flashrank/

[^6]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2

[^7]: https://www.youtube.com/watch?v=L0jDvTLJAOM

[^8]: https://www.reddit.com/r/LangChain/comments/1ha8j1a/reranking_using_flashrankreranker/

[^9]: https://www.youtube.com/watch?v=fUIeMuRtQr8

[^10]: https://arxiv.org/abs/2403.10407

[^11]: https://sbert.net/examples/cross_encoder/training/ms_marco/README.html

