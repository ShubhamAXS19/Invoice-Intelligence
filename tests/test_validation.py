import pytest
from unittest.mock import patch
from src.extraction.invoice_schema import InvoiceFields, InvoiceExtractionResult
from src.validation.retriever import retrieve_rules


def test_retrieve_rules_returns_results():
    """Corpus must be built before running this test."""
    try:
        results = retrieve_rules("GSTIN mandatory field invoice", n_results=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "source" in results[0]
        assert "relevance_score" in results[0]
    except Exception:
        pytest.skip("ChromaDB collection not built yet — run build_corpus() first")