from pathlib import Path
import pytest
from src.ocr import extract_text, OCRResult

SAMPLE_IMAGE = Path("data/raw/train/img/X00016469612.jpg")


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="SROIE data not present")
def test_extract_text_returns_result():
    result = extract_text(SAMPLE_IMAGE)
    assert isinstance(result, OCRResult)
    assert len(result.raw_text) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.needs_fallback, bool)


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="SROIE data not present")
def test_confidence_threshold():
    result = extract_text(SAMPLE_IMAGE, confidence_threshold=0.99)
    assert result.needs_fallback is True

    result = extract_text(SAMPLE_IMAGE, confidence_threshold=0.01)
    assert result.needs_fallback is False