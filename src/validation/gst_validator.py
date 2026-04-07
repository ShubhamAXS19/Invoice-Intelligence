import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from src.extraction.invoice_schema import InvoiceExtractionResult
from src.validation.retriever import retrieve_rules

load_dotenv()

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

_SYSTEM_PROMPT = """
You are a GST compliance auditor for Indian invoices.
You will be given extracted invoice fields and relevant GST rules.
Assess compliance and return ONLY valid JSON — no explanation, no markdown:
{
  "is_compliant": true or false,
  "verdict": "one sentence summary",
  "violations": ["list of violations found, empty if none"],
  "cited_rules": ["exact rule text that supports your verdict"],
  "confidence": 0.0 to 1.0
}
""".strip()


class ValidationResult(BaseModel):
    image_id: str
    is_compliant: bool
    verdict: str
    violations: List[str]
    cited_rules: List[str]
    confidence: float
    retrieved_context: List[Dict]


def validate_invoice(extraction: InvoiceExtractionResult) -> ValidationResult:
    """
    Run RAG-based GST validation on extracted invoice fields.
    """
    fields = extraction.extracted

    # build targeted queries for retrieval
    queries = [
        "mandatory fields required on GST invoice",
        "GSTIN format validation rules",
        "invoice date and serial number requirements",
    ]

    # deduplicate retrieved chunks across queries
    seen = set()
    all_context = []
    for query in queries:
        for chunk in retrieve_rules(query, n_results=2):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                all_context.append(chunk)

    context_text = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in all_context
    )

    invoice_summary = f"""
Invoice fields extracted:
- Company: {fields.company}
- Date: {fields.date}
- Address: {fields.address}
- Total: {fields.total}
- OCR Confidence: {extraction.ocr_confidence}
""".strip()

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"GST RULES:\n{context_text}\n\nINVOICE:\n{invoice_summary}"),
    ]

    response = _llm.invoke(messages)
    raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)

    return ValidationResult(
        image_id=extraction.image_id,
        is_compliant=result["is_compliant"],
        verdict=result["verdict"],
        violations=result.get("violations", []),
        cited_rules=result.get("cited_rules", []),
        confidence=result.get("confidence", 0.0),
        retrieved_context=all_context,
    )