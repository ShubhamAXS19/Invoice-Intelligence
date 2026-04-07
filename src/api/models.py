from typing import List, Optional
from pydantic import BaseModel


class InvoiceAnalysisRequest(BaseModel):
    image_path: str


class ExtractedFields(BaseModel):
    company: str
    date: str
    address: str
    total: str


class ValidationSummary(BaseModel):
    is_compliant: bool
    verdict: str
    violations: List[str]
    cited_rules: List[str]
    confidence: float


class InvoiceAnalysisResponse(BaseModel):
    image_id: str
    ocr_confidence: float
    extraction_method: str
    extracted_fields: ExtractedFields
    validation: ValidationSummary
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    corpus_loaded: bool