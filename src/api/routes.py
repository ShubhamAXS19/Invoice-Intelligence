from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil
import tempfile

from src.api.models import InvoiceAnalysisResponse, HealthResponse
from src.ocr import extract_text
from src.extraction import extract_fields
from src.validation import validate_invoice

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        from src.validation.corpus_loader import get_collection
        get_collection()
        corpus_loaded = True
    except Exception:
        corpus_loaded = False

    return HealthResponse(status="ok", corpus_loaded=corpus_loaded)


@router.post("/analyze", response_model=InvoiceAnalysisResponse)
async def analyze_invoice(file: UploadFile = File(...)):
    """
    Accept an invoice image, run OCR + extraction + GST validation.
    Returns structured fields and compliance verdict.
    """
    allowed = {".jpg", ".jpeg", ".png"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {suffix} not supported. Use jpg or png.")

    # save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        ocr_result = extract_text(tmp_path)
        extraction = extract_fields(ocr_result)
        validation = validate_invoice(extraction)
    finally:
        tmp_path.unlink(missing_ok=True)

    return InvoiceAnalysisResponse(
        image_id=extraction.image_id,
        ocr_confidence=extraction.ocr_confidence,
        extraction_method=extraction.extraction_method,
        extracted_fields=extraction.extracted.model_dump(),
        validation=validation.model_dump(exclude={"image_id", "retrieved_context"}),
    )