"""Tests for reranking utilities."""

from unittest.mock import Mock, patch

import pytest

from crawl4ai_mcp.utilities.reranking import Reranker


@pytest.fixture
def mock_ranker():
    return Mock()


@pytest.fixture
def reranker(test_settings, mock_ranker):
    return Reranker(model=mock_ranker, settings=test_settings)


@pytest.fixture
def sample_results():
    return [
        {"content": "First result about machine learning", "url": "url1"},
        {"content": "Second result about deep learning", "url": "url2"},
        {"content": "Third result about neural networks", "url": "url3"},
    ]


def test_rerank_results_success(reranker, mock_ranker, sample_results) -> None:
    mock_ranker.rerank.return_value = [
        {"id": 1, "score": 0.9},
        {"id": 2, "score": 0.7},
        {"id": 0, "score": 0.5},
    ]
    reranked = reranker.rerank_results("test query", sample_results)
    assert len(reranked) == 3
    assert reranked[0]["content"] == "Second result about deep learning"
    assert reranked[0]["rerank_score"] == 0.9


def test_rerank_results_empty_results(reranker) -> None:
    assert reranker.rerank_results("query", []) == []


def test_rerank_results_no_model(test_settings, sample_results) -> None:
    reranker_no_model = Reranker(model=None, settings=test_settings)
    results = reranker_no_model.rerank_results("query", sample_results)
    assert results == sample_results


def test_rerank_results_custom_content_key(reranker, mock_ranker) -> None:
    mock_ranker.rerank.return_value = [{"id": 0, "score": 0.8}]
    results = [{"summary": "A summary"}]
    reranked = reranker.rerank_results("query", results, content_key="summary")
    assert reranked[0]["rerank_score"] == 0.8


def test_filter_by_threshold(reranker) -> None:
    results = [
        {"rerank_score": 0.9},
        {"rerank_score": 0.4},
        {"rerank_score": 0.2},
    ]
    filtered = reranker.filter_by_threshold(results, threshold=0.3)
    assert filtered == [{"rerank_score": 0.9}, {"rerank_score": 0.4}]


def test_rerank_results_error_handling(reranker, mock_ranker, sample_results) -> None:
    mock_ranker.rerank.side_effect = Exception("Reranking failed")
    assert reranker.rerank_results("query", sample_results) == sample_results


@patch("crawl4ai_mcp.utilities.reranking.Ranker")
def test_init_with_reranking_enabled(MockRanker, test_settings) -> None:
    test_settings.use_reranking = True
    test_settings.reranker_model = "test-model"
    test_settings.reranker_cache_dir = "./cache"
    test_settings.reranker_max_length = 256
    reranker = Reranker(model=None, settings=test_settings)
    assert reranker.model is MockRanker.return_value
    MockRanker.assert_called_once_with(
        model_name="test-model",
        cache_dir="./cache",
        max_length=256,
    )
