import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.ocr.paddle_extractor import OCRResult
from src.extraction.invoice_schema import InvoiceFields, InvoiceExtractionResult

load_dotenv()

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

_SYSTEM_PROMPT = """
You are an invoice parsing engine. Extract structured fields from raw OCR text.
Return ONLY valid JSON matching this schema exactly — no explanation, no markdown:
{
  "company": "string",
  "date": "DD/MM/YYYY",
  "address": "string",
  "total": "numeric string e.g. 9.00"
}
If a field is not found, use an empty string.
""".strip()


def extract_fields(ocr_result: OCRResult) -> InvoiceExtractionResult:
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"OCR TEXT:\n{ocr_result.raw_text}"),
    ]

    response = _llm.invoke(messages)

    raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    raw_json = json.loads(raw)
    extracted = InvoiceFields(**raw_json)

    return InvoiceExtractionResult(
        image_id=ocr_result.image_id,
        ocr_confidence=ocr_result.confidence,
        extracted=extracted,
        extraction_method="llm+fallback" if ocr_result.needs_fallback else "llm",
    )