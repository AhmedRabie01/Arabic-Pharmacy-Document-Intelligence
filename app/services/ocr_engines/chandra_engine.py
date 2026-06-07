from pathlib import Path
import base64
import json
import requests

from app.core.config import (
    CHANDRA_BASE_URL,
    CHANDRA_MODEL,
    CHANDRA_OUTPUT_FORMAT,
    CHANDRA_TIMEOUT_SECONDS,
    CHANDRA_NUM_PREDICT,
    CHANDRA_TEMPERATURE,
    ENABLE_CHANDRA_CACHE,
    PROCESSED_CHANDRA_JSON_DIR,
)

from app.schemas.document import OCRLine, OCRPageResult


def extract_json_from_response_text(response_text: str) -> object | None:
    if not response_text:
        return None

    cleaned = response_text.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    first_array = cleaned.find("[")
    last_array = cleaned.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        try:
            return json.loads(cleaned[first_array:last_array + 1])
        except json.JSONDecodeError:
            pass

    first_object = cleaned.find("{")
    last_object = cleaned.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        try:
            return json.loads(cleaned[first_object:last_object + 1])
        except json.JSONDecodeError:
            pass

    return None


def is_compatible_layout_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False

    label = item.get("label")
    bbox = item.get("bbox")

    return isinstance(label, str) and isinstance(bbox, str)


def is_compatible_layout_payload(response_text: str) -> bool:
    parsed_payload = extract_json_from_response_text(response_text)

    if not isinstance(parsed_payload, list) or not parsed_payload:
        return False

    return all(is_compatible_layout_item(item) for item in parsed_payload)


def encode_image_to_base64(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_chandra_response(response_text: str) -> list[OCRLine]:
    """
    Chandra/Ollama output may be Markdown, JSON text, or structured text
    depending on model behavior and prompt.

    For now, we store the whole response as one OCRLine.
    Later we can improve parsing if the model returns stable JSON.
    """
    return [
        OCRLine(
            text=response_text.strip(),
            confidence=None,
            box=None,
        )
    ]

def get_chandra_cache_path(document_id: str, page_number: int) -> Path:
    return PROCESSED_CHANDRA_JSON_DIR / f"{document_id}_page_{page_number}.json"


def load_chandra_cache(document_id: str, page_number: int) -> dict | None:
    cache_path = get_chandra_cache_path(document_id, page_number)

    if not ENABLE_CHANDRA_CACHE:
        return None

    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as file:
        cached_payload = json.load(file)

    cached_page_text = str(cached_payload.get("page_text", ""))
    if not is_compatible_layout_payload(cached_page_text):
        return None

    return cached_payload


def save_chandra_cache(document_id: str, page_number: int, data: dict) -> None:
    PROCESSED_CHANDRA_JSON_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = get_chandra_cache_path(document_id, page_number)

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def run_chandra_ocr_on_image(
    image_path: str,
    page_number: int,
    document_id: str | None = None,
) -> OCRPageResult:
    # 1. Try cache first
    if document_id:
        cached = load_chandra_cache(document_id, page_number)
        if cached:
            return OCRPageResult(**cached)

    # 2. Encode image
    image_base64 = encode_image_to_base64(image_path)

    # 3. Prompt
    prompt = """
Analyze this Arabic pharmacy supplier document page for layout only.

Return valid JSON only as a flat array.

Required schema:
[
  {"label": "Text", "bbox": "x1 y1 x2 y2"},
  {"label": "Table", "bbox": "x1 y1 x2 y2"}
]

Rules:
- Start the response with `[` and end it with `]`.
- Coordinates must use a 0-1000 page-relative scale.
- bbox must be a string with exactly four numbers: x1 y1 x2 y2.
- Include every major visible region that matters for layout.
- If the page contains a line-items table, include one item with label exactly "Table".
- Do not return nested objects.
- Do not return markdown.
- Do not extract rows or cell values.
- Do not invent regions that are not visible.
"""

    # 4. Ollama endpoint
    url = f"{CHANDRA_BASE_URL}/api/generate"

    # 5. Request payload
    payload = {
        "model": CHANDRA_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": CHANDRA_TEMPERATURE,
            # Layout-only prompting should not need the full long-generation budget.
            "num_predict": min(CHANDRA_NUM_PREDICT, 350),
        },
    }

    # 6. Send request
    response = requests.post(
        url,
        json=payload,
        timeout=CHANDRA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    # 7. Parse response
    data = response.json()
    response_text = data.get("response", "")

    lines = parse_chandra_response(response_text)

    # 8. Build result
    result = OCRPageResult(
        page_number=page_number,
        image_path=image_path,
        lines=lines,
        page_text=response_text,
        average_confidence=None,
    )

    # 9. Save cache
    if document_id:
        save_chandra_cache(
            document_id=document_id,
            page_number=page_number,
            data=result.model_dump(),
        )

    return result
