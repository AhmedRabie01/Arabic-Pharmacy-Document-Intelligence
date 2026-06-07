from paddleocr import PaddleOCR

from app.core.config import OCR_LANGUAGE, OCR_USE_ANGLE_CLS
from app.schemas.document import OCRLine, OCRPageResult


_paddle_ocr_engine: PaddleOCR | None = None


def get_paddle_engine() -> PaddleOCR:
    global _paddle_ocr_engine

    if _paddle_ocr_engine is None:
        _paddle_ocr_engine = PaddleOCR(
            use_angle_cls=OCR_USE_ANGLE_CLS,
            lang=OCR_LANGUAGE,
        )

    return _paddle_ocr_engine


def calculate_average_confidence(lines: list[OCRLine]) -> float:
    confidence_values = [
        line.confidence
        for line in lines
        if line.confidence is not None
    ]

    if not confidence_values:
        return 0.0

    return round(sum(confidence_values) / len(confidence_values), 4)

def is_noise_text(text: str, confidence: float | None) -> bool:
    cleaned = text.strip()

    if not cleaned:
        return True

    known_noise_words = {
        "camscanner",
        "cam scanner",
        "redmi",
        "rid",
        "j,11",
    }

    if cleaned.lower() in known_noise_words:
        return True

    if confidence is not None and confidence < 0.30:
        return True

    if len(cleaned) == 1 and cleaned in {".", "•", "-", "_", "ـ"}:
        return True

    return False

def normalize_paddle_result(raw_result) -> list[OCRLine]:
    lines: list[OCRLine] = []

    if not raw_result:
        return lines

    if isinstance(raw_result, list):
        for result_item in raw_result:

            # PaddleOCR v3-style format
            if isinstance(result_item, dict):
                texts = result_item.get("rec_texts", [])
                scores = result_item.get("rec_scores", [])
                boxes = (
                    result_item.get("rec_polys")
                    or result_item.get("dt_polys")
                    or result_item.get("rec_boxes")
                    or []
                )

                for index, text in enumerate(texts):
                    confidence = None
                    box = None

                    if index < len(scores):
                        confidence = float(scores[index])

                    if is_noise_text(str(text), confidence):
                        continue

                    if index < len(boxes):
                        raw_box = boxes[index]

                        try:
                            box = raw_box.tolist()
                        except AttributeError:
                            box = raw_box

                    lines.append(
                        OCRLine(
                            text=str(text),
                            confidence=confidence,
                            box=box,
                        )
                    )

                continue

            # PaddleOCR v2-style format
            if isinstance(result_item, list):
                for item in result_item:
                    try:
                        box = item[0]
                        text = item[1][0]
                        confidence = float(item[1][1])

                        # وهنا برضه المكان الصحيح
                        if is_noise_text(str(text), confidence):
                            continue

                        lines.append(
                            OCRLine(
                                text=str(text),
                                confidence=confidence,
                                box=box,
                            )
                        )

                    except Exception:
                        continue

    return lines

def run_paddle_ocr_on_image(
    image_path: str,
    page_number: int,
) -> OCRPageResult:
    engine = get_paddle_engine()

    raw_result = engine.ocr(image_path)

    lines = normalize_paddle_result(raw_result)
    page_text = "\n".join(line.text for line in lines)
    average_confidence = calculate_average_confidence(lines)

    return OCRPageResult(
        page_number=page_number,
        image_path=image_path,
        lines=lines,
        page_text=page_text,
        average_confidence=average_confidence,
    )
