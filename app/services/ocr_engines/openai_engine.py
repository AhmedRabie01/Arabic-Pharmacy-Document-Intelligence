from __future__ import annotations

from pathlib import Path
import base64
import json

import requests

from app.core.config import (
    CHANDRA_OUTPUT_FORMAT,
    ENABLE_OPENAI_CACHE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_IMAGE_DETAIL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    PROCESSED_OPENAI_JSON_DIR,
)
from app.schemas.document import OCRLine, OCRPageResult


def encode_image_to_data_url(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    return f"data:{media_type};base64,{encoded}"


def parse_openai_response(response_text: str) -> list[OCRLine]:
    return [
        OCRLine(
            text=response_text.strip(),
            confidence=None,
            box=None,
        )
    ]


def get_openai_cache_path(document_id: str, page_number: int) -> Path:
    return PROCESSED_OPENAI_JSON_DIR / f"{document_id}_page_{page_number}.json"


def load_openai_cache(document_id: str, page_number: int) -> dict | None:
    if not ENABLE_OPENAI_CACHE:
        return None

    cache_path = get_openai_cache_path(document_id, page_number)
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_openai_cache(document_id: str, page_number: int, data: dict) -> None:
    PROCESSED_OPENAI_JSON_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_openai_cache_path(document_id, page_number)

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _normalize_layout_payload(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("OpenAI returned an empty response.")

    parsed = json.loads(cleaned)
    regions = parsed.get("regions")
    if not isinstance(regions, list):
        raise ValueError("OpenAI JSON response is missing a 'regions' array.")

    normalized_regions = []
    for item in regions:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        bbox = str(item.get("bbox", "")).strip()
        if not label or not bbox:
            continue
        normalized_regions.append(
            {
                "label": label,
                "bbox": bbox,
            }
        )

    return json.dumps(normalized_regions, ensure_ascii=False)


def run_openai_ocr_on_image(
    image_path: str,
    page_number: int,
    document_id: str | None = None,
) -> OCRPageResult:
    if document_id:
        cached = load_openai_cache(document_id, page_number)
        if cached:
            return OCRPageResult(**cached)

    api_key = OPENAI_API_KEY.strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    model = OPENAI_MODEL.strip() or "gpt-4.1-mini"
    image_data_url = encode_image_to_data_url(image_path)

    prompt = """
Analyze this Arabic pharmacy supplier document page for layout only.

Return valid JSON only using this exact object shape:
{
  "regions": [
    {"label": "Text", "bbox": "x1 y1 x2 y2"},
    {"label": "Table", "bbox": "x1 y1 x2 y2"}
  ]
}

Rules:
- The response must be valid JSON.
- Coordinates must use a 0-1000 page-relative scale.
- bbox must be a string with exactly four numbers: x1 y1 x2 y2.
- Include every major visible region that matters for layout.
- If the page contains a line-items table, include one region with label exactly "Table".
- Do not return markdown.
- Do not extract rows or cell values.
- Do not invent regions that are not visible.
"""

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": OPENAI_IMAGE_DETAIL,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_object",
            }
        },
    }

    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    response_text = str(data.get("output_text", "")).strip()

    if not response_text:
        output_items = data.get("output") or []
        for item in output_items:
            content_items = item.get("content") or []
            for content in content_items:
                if content.get("type") == "output_text":
                    response_text = str(content.get("text", "")).strip()
                    if response_text:
                        break
            if response_text:
                break

    normalized_page_text = _normalize_layout_payload(response_text)
    lines = parse_openai_response(normalized_page_text)

    result = OCRPageResult(
        page_number=page_number,
        image_path=image_path,
        lines=lines,
        page_text=normalized_page_text,
        average_confidence=None,
    )

    if document_id:
        save_openai_cache(
            document_id=document_id,
            page_number=page_number,
            data=result.model_dump(),
        )

    return result
