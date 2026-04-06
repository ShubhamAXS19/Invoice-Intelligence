from pathlib import Path
from dataclasses import dataclass

from paddleocr import PaddleOCR
import cv2

# initialise once at module level — PaddleOCR loads heavy models on first call
_ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


@dataclass
class OCRResult:
    image_id: str
    raw_text: str
    confidence: float
    needs_fallback: bool


def extract_text(image_path: Path, confidence_threshold: float = 0.85) -> OCRResult:
    """
    Run PaddleOCR on a single invoice image.
    Returns raw text, mean confidence, and a flag if LLM fallback is needed.
    """
    image_id = image_path.stem
    result = _ocr_engine.ocr(str(image_path), cls=True)

    lines = []
    confidences = []

    # result is a nested list: result[0] is a list of (bbox, (text, confidence))
    if result and result[0]:
        for line in result[0]:
            text, conf = line[1]
            lines.append(text)
            confidences.append(conf)

    raw_text = "\n".join(lines)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    needs_fallback = mean_confidence < confidence_threshold

    return OCRResult(
        image_id=image_id,
        raw_text=raw_text,
        confidence=round(mean_confidence, 4),
        needs_fallback=needs_fallback,
    )


def load_ground_truth(entities_dir: Path, image_id: str) -> dict:
    """Load the ground truth JSON for a given image id."""
    import json
    gt_path = entities_dir / f"{image_id}.txt"
    return json.loads(gt_path.read_text())