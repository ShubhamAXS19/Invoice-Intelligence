import pytest
from unittest.mock import patch, MagicMock
from src.ocr.paddle_extractor import OCRResult
from src.extraction import extract_fields
from src.extraction.invoice_schema import InvoiceFields


def _mock_ocr_result():
    return OCRResult(
        image_id="test_001",
        raw_text="ACME CORP\n25/12/2018\n123 Main St\nTotal: 100.00",
        confidence=0.92,
        needs_fallback=False,
    )


def test_invoice_fields_date_normalisation():
    fields = InvoiceFields(
        company="Test Co",
        date="25/12/20188:13PM",
        address="123 Main St",
        total="9.00",
    )
    assert fields.date == "25/12/2018"


def test_invoice_fields_total_normalisation():
    fields = InvoiceFields(
        company="Test Co",
        date="25/12/2018",
        address="123 Main St",
        total="RM 9.00 only",
    )
    assert fields.total == "9.00"