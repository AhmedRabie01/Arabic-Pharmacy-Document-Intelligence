from pathlib import Path
import json

from app.core.config import (
    LOCAL_OCR_ENGINE,
    PROCESSED_OCR_JSON_DIR,
    PROCESSED_TEXT_DIR,
)
from app.schemas.document import (
    OCRDocumentResult,
    OCRLine,
    OCRPageResult,
)
from app.services.ocr_engines.paddle_engine import (
    calculate_average_confidence,
    run_paddle_ocr_on_image,
)
from app.services.ocr_engines.chandra_engine import run_chandra_ocr_on_image


def run_ocr_on_image(
    image_path: str,
    page_number: int,
) -> OCRPageResult:
    engine = LOCAL_OCR_ENGINE.lower().strip()

    if engine == "paddleocr":
        return run_paddle_ocr_on_image(
            image_path=image_path,
            page_number=page_number,
        )

    if engine == "chandra":
        return run_chandra_ocr_on_image(
            image_path=image_path,
            page_number=page_number,
        )

    raise ValueError(f"Unsupported OCR engine: {LOCAL_OCR_ENGINE}")


def get_output_paths(document_id: str) -> tuple[Path, Path]:
    ocr_json_path = PROCESSED_OCR_JSON_DIR / f"{document_id}.json"
    ocr_text_path = PROCESSED_TEXT_DIR / f"{document_id}.txt"
    return ocr_json_path, ocr_text_path


def calculate_document_average_confidence(
    pages: list[OCRPageResult],
) -> float | None:
    all_lines: list[OCRLine] = [
        line
        for page in pages
        for line in page.lines
        if line.confidence is not None
    ]

    if not all_lines:
        return None

    return calculate_average_confidence(all_lines)


def build_full_text(pages: list[OCRPageResult]) -> str:
    full_text_parts: list[str] = []

    for page in pages:
        full_text_parts.append(f"[PAGE {page.page_number}]")
        full_text_parts.append(page.page_text)

    return "\n\n".join(full_text_parts)


def save_ocr_outputs(result: OCRDocumentResult) -> None:
    PROCESSED_OCR_JSON_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if not result.ocr_json_path:
        raise ValueError("ocr_json_path is missing.")

    if not result.ocr_text_path:
        raise ValueError("ocr_text_path is missing.")

    with Path(result.ocr_json_path).open("w", encoding="utf-8") as file:
        json.dump(
            result.model_dump(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    with Path(result.ocr_text_path).open("w", encoding="utf-8") as file:
        file.write(result.full_text)


def run_ocr_on_document(
    document_id: str,
    image_paths: list[str],
    source: str = "preprocessed",
) -> OCRDocumentResult:
    pages: list[OCRPageResult] = []

    for page_number, image_path in enumerate(image_paths, start=1):
        page_result = run_ocr_on_image(
            image_path=image_path,
            page_number=page_number,
        )
        pages.append(page_result)

    full_text = build_full_text(pages)
    average_confidence = calculate_document_average_confidence(pages)

    ocr_json_path, ocr_text_path = get_output_paths(document_id)

    result = OCRDocumentResult(
        document_id=document_id,
        engine=LOCAL_OCR_ENGINE,
        source=source,
        page_count=len(pages),
        pages=pages,
        full_text=full_text,
        average_confidence=average_confidence,
        ocr_json_path=str(ocr_json_path),
        ocr_text_path=str(ocr_text_path),
        status="ocr_completed",
    )

    save_ocr_outputs(result)

    return result