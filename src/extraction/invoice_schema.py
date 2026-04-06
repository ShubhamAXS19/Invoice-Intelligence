from pydantic import BaseModel, Field, field_validator
import re


class InvoiceFields(BaseModel):
    company: str = Field(description="Company or merchant name")
    date: str = Field(description="Invoice date in DD/MM/YYYY format")
    address: str = Field(description="Full address of the merchant")
    total: str = Field(description="Final total amount as a numeric string")

    @field_validator("date")
    @classmethod
    def normalise_date(cls, v: str) -> str:
        # strip any time component fused to date e.g. "25/12/20188:13:39PM"
        match = re.search(r"\d{2}/\d{2}/\d{4}", v)
        return match.group() if match else v

    @field_validator("total")
    @classmethod
    def normalise_total(cls, v: str) -> str:
        # keep only digits and decimal point
        match = re.search(r"\d+\.\d{2}", v)
        return match.group() if match else v


class InvoiceExtractionResult(BaseModel):
    image_id: str
    ocr_confidence: float
    extracted: InvoiceFields
    extraction_method: str  # "llm" or "llm+fallback"