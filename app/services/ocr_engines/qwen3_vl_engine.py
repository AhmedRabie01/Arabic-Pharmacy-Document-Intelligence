from pathlib import Path
import base64
import json
import os

import requests

from app.core.config import (
    CHANDRA_BASE_URL,
    CHANDRA_NUM_PREDICT,
    CHANDRA_OUTPUT_FORMAT,
    CHANDRA_TEMPERATURE,
    CHANDRA_TIMEOUT_SECONDS,
    FEW_SHOT_DIR,
    PROCESSED_CHANDRA_JSON_DIR,
)
from app.schemas.document import OCRLine, OCRPageResult


QWEN3_VL_MODEL = os.getenv("QWEN3_VL_MODEL", "qwen3-vl:4b")
QWEN3_VL_BASE_URL = os.getenv("QWEN3_VL_BASE_URL", CHANDRA_BASE_URL)
QWEN3_VL_TIMEOUT_SECONDS = int(
    os.getenv("QWEN3_VL_TIMEOUT_SECONDS", str(CHANDRA_TIMEOUT_SECONDS))
)
QWEN3_VL_NUM_PREDICT = int(
    os.getenv("QWEN3_VL_NUM_PREDICT", str(min(CHANDRA_NUM_PREDICT, 500)))
)
QWEN3_VL_TEMPERATURE = float(
    os.getenv("QWEN3_VL_TEMPERATURE", str(CHANDRA_TEMPERATURE))
)
QWEN3_VL_OUTPUT_FORMAT = os.getenv("QWEN3_VL_OUTPUT_FORMAT", CHANDRA_OUTPUT_FORMAT)
QWEN3_VL_CACHE_DIR = PROCESSED_CHANDRA_JSON_DIR.parent / "qwen3_vl_json"
FEW_SHOT_PACK_PATH = FEW_SHOT_DIR / "few_shot_pack.json"


def encode_image_to_base64(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_qwen_response(response_text: str) -> list[OCRLine]:
    return [
        OCRLine(
            text=response_text.strip(),
            confidence=None,
            box=None,
        )
    ]


def get_qwen_cache_path(document_id: str, page_number: int) -> Path:
    return QWEN3_VL_CACHE_DIR / f"{document_id}_page_{page_number}.json"


def load_qwen_cache(document_id: str, page_number: int) -> dict | None:
    cache_path = get_qwen_cache_path(document_id, page_number)
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_qwen_cache(document_id: str, page_number: int, data: dict) -> None:
    QWEN3_VL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_qwen_cache_path(document_id, page_number)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_qwen_few_shot_examples() -> tuple[str, str]:
    if not FEW_SHOT_PACK_PATH.exists():
        return "", ""

    with FEW_SHOT_PACK_PATH.open("r", encoding="utf-8") as file:
        pack = json.load(file)

    monthly_examples = pack.get("tasks", {}).get("monthly_statement_rows", {}).get(
        "examples", []
    )
    daily_examples = pack.get("tasks", {}).get("daily_invoice_rows", {}).get(
        "examples", []
    )

    monthly_example = ""
    daily_example = ""

    if monthly_examples:
        monthly_example = json.dumps(
            monthly_examples[0].get("output_sample", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if daily_examples:
        daily_example = json.dumps(
            daily_examples[0].get("output_sample", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return monthly_example, daily_example


def build_qwen_prompt() -> str:
    monthly_example, daily_example = load_qwen_few_shot_examples()

    prompt = """
Extract structured data from this Arabic pharmacy supplier document page.

First detect the page type:
- monthly_statement
- daily_invoice
- unknown

Return JSON only.

Allowed output shapes:

For monthly statement:
{
  "page_type": "monthly_statement",
  "header": {
    "customer_name": null,
    "period_from": null,
    "period_to": null
  },
  "rows": [
    {
      "date": null,
      "reference_number": null,
      "note": null,
      "debit": null,
      "credit": null,
      "balance": null
    }
  ],
  "footer_summary": {
    "current_balance": null
  }
}

For daily invoice:
{
  "page_type": "daily_invoice",
  "header": {
    "customer_name": null,
    "invoice_number": null,
    "invoice_date": null
  },
  "rows": [
    {
      "item_name_ar": null,
      "quantity": null,
      "price": null,
      "discount": null,
      "total": null
    }
  ],
  "footer_summary": {
    "current_balance": null
  }
}

For unknown:
{
  "page_type": "unknown",
  "header": {},
  "rows": [],
  "footer_summary": {}
}

Rules:
- JSON only.
- Do not add markdown.
- Do not add extra keys.
- Keep Arabic text exactly as seen when possible.
- If the page is a summary-only monthly page, return "rows": [].
- If a field is missing, use null.
- Do not invent rows.
""".strip()

    if monthly_example:
        prompt += f"\n\nMonthly example:\n{monthly_example}"

    if daily_example:
        prompt += f"\n\nDaily example:\n{daily_example}"

    return prompt


def run_qwen3_vl_on_image(
    image_path: str,
    page_number: int,
    document_id: str | None = None,
) -> OCRPageResult:
    if document_id:
        cached = load_qwen_cache(document_id, page_number)
        if cached:
            return OCRPageResult(**cached)

    image_base64 = encode_image_to_base64(image_path)
    prompt = build_qwen_prompt()
    url = f"{QWEN3_VL_BASE_URL}/api/generate"

    payload = {
        "model": QWEN3_VL_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "format": QWEN3_VL_OUTPUT_FORMAT,
        "options": {
            "temperature": QWEN3_VL_TEMPERATURE,
            "num_predict": QWEN3_VL_NUM_PREDICT,
        },
    }

    response = requests.post(
        url,
        json=payload,
        timeout=QWEN3_VL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    response_text = data.get("response", "")
    lines = parse_qwen_response(response_text)

    result = OCRPageResult(
        page_number=page_number,
        image_path=image_path,
        lines=lines,
        page_text=response_text,
        average_confidence=None,
    )

    if document_id:
        save_qwen_cache(
            document_id=document_id,
            page_number=page_number,
            data=result.model_dump(),
        )

    return result
