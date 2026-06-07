from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import base64
import json
import re

import requests

from app.core.config import (
    ENABLE_OPENAI_CACHE,
    FEW_SHOT_DIR,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_IMAGE_DETAIL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    PROCESSED_OPENAI_JSON_DIR,
)
from app.schemas.document import (
    DocumentTypeResult,
    MixedClassificationDetails,
    PageDocumentTypeResult,
)
from app.services.document_classifier import aggregate_document_type_from_pages


ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

KNOWN_DOCUMENT_TYPES = {"monthly_statement", "daily_invoice", "unknown"}
MODEL_PRICING_PER_1M_TOKENS = {
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached_input": 0.10,
        "output": 1.60,
    },
    "gpt-4.1": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
    },
}


def _encode_image_to_data_url(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    return f"data:{media_type};base64,{encoded}"


@lru_cache(maxsize=1)
def _load_few_shot_pack() -> dict:
    pack_path = FEW_SHOT_DIR / "few_shot_pack.json"
    with pack_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _build_prompt(page_number: int) -> str:
    few_shot_pack = _load_few_shot_pack()
    monthly_example = (
        few_shot_pack["tasks"]["monthly_statement_rows"]["examples"][0]["output_sample"]
    )
    daily_example = (
        few_shot_pack["tasks"]["daily_invoice_rows"]["examples"][0]["output_sample"]
    )

    monthly_example = {
        **monthly_example,
        "signature": {
            "signature_present": False,
            "signer_name": None,
            "signature_note": None,
        },
    }
    daily_example = {
        **daily_example,
        "signature": {
            "signature_present": True,
            "signer_name": "غير واضح",
            "signature_note": "ضع الاسم فقط إذا كان مقروءا بشكل معقول.",
        },
    }

    return (
        "Extract one Arabic pharmacy document page into JSON only.\n"
        f"Set page_number to {page_number}.\n"
        "Choose page_document_type from: monthly_statement, daily_invoice, unknown.\n"
        "The page may be a monthly statement page or a daily invoice page.\n"
        "Return exactly one JSON object using only the English keys from the schema.\n"
        "Rules:\n"
        "- Keep Arabic text exactly when readable.\n"
        "- Do not return markdown.\n"
        "- Do not explain.\n"
        "- Use these exact row keys only.\n"
        "- If the page is monthly_statement, each row must use: date, reference_number, note, debit, credit, balance.\n"
        "- If the page is daily_invoice, each row must use: item_name, item_name_ar, quantity, price, discount, total.\n"
        "- Also include item_name_ar for daily rows when possible.\n"
        "- If the page is a monthly summary-only page, rows may be [].\n"
        "- If a field is not visible, use null or omit it inside header/footer_summary.\n"
        "- invoice_date must be null unless an explicit invoice date label/value is visible in the actual invoice header. Do not use scan timestamps or watermark timestamps as invoice_date.\n"
        "- customer_name should be the supplier customer/account name, not company logo text.\n"
        "- Detect whether a handwritten signature exists. If visible, set signature_present=true and return signer_name only if reasonably readable in Arabic or mixed Arabic/English handwriting.\n"
        "- If the signer name is unclear, keep signer_name null and use signature_note.\n"
        "Monthly example:\n"
        f"{_compact_json(monthly_example)}\n"
        "Daily example:\n"
        f"{_compact_json(daily_example)}"
    )


def _extract_response_text(data: dict) -> str:
    output_text = str(data.get("output_text", "")).strip()
    if output_text:
        return output_text

    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                text = str(content.get("text", "")).strip()
                if text:
                    return text

    return ""


def _parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        possible_json = cleaned[first_brace:last_brace + 1]
        parsed = json.loads(possible_json)
        return parsed if isinstance(parsed, dict) else {}

    raise ValueError("OpenAI did not return a valid JSON object.")


def _resolve_model_pricing(model_name: str) -> dict | None:
    normalized = (model_name or "").strip()
    if not normalized:
        return None

    if normalized in MODEL_PRICING_PER_1M_TOKENS:
        return MODEL_PRICING_PER_1M_TOKENS[normalized]

    for known_model, pricing in MODEL_PRICING_PER_1M_TOKENS.items():
        if normalized.startswith(f"{known_model}-"):
            return pricing

    return None


def _build_usage_summary(raw_response: dict) -> dict | None:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    cached_input_tokens = int(input_details.get("cached_tokens") or 0)
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    billable_input_tokens = max(0, input_tokens - cached_input_tokens)

    model_name = str(raw_response.get("model") or OPENAI_MODEL or "").strip()
    pricing = _resolve_model_pricing(model_name)

    estimated_cost_usd = None
    pricing_basis = None
    if pricing is not None:
        estimated_cost_usd = round(
            (
                (billable_input_tokens / 1_000_000) * pricing["input"]
                + (cached_input_tokens / 1_000_000) * pricing["cached_input"]
                + (output_tokens / 1_000_000) * pricing["output"]
            ),
            8,
        )
        pricing_basis = {
            "input_per_1m_usd": pricing["input"],
            "cached_input_per_1m_usd": pricing["cached_input"],
            "output_per_1m_usd": pricing["output"],
        }

    return {
        "model": model_name or None,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing_basis": pricing_basis,
    }


def _print_usage_summary(scope: str, usage_summary: dict | None) -> None:
    if not usage_summary:
        return

    estimated_cost_usd = usage_summary.get("estimated_cost_usd")
    estimated_cost_text = (
        f"${estimated_cost_usd:.8f}" if isinstance(estimated_cost_usd, float) else "unknown"
    )
    print(
        "[OpenAI Usage] "
        f"{scope}: "
        f"model={usage_summary.get('model') or 'unknown'} "
        f"input_tokens={usage_summary.get('input_tokens', 0)} "
        f"cached_input_tokens={usage_summary.get('cached_input_tokens', 0)} "
        f"output_tokens={usage_summary.get('output_tokens', 0)} "
        f"total_tokens={usage_summary.get('total_tokens', 0)} "
        f"estimated_cost_usd={estimated_cost_text}"
    )


def _aggregate_usage_summaries(page_cache_payloads: list[dict]) -> dict | None:
    usage_summaries = [
        payload.get("usage_summary")
        for payload in page_cache_payloads
        if isinstance(payload.get("usage_summary"), dict)
    ]
    if not usage_summaries:
        return None

    document_model = next(
        (summary.get("model") for summary in usage_summaries if summary.get("model")),
        None,
    )
    pricing_basis = next(
        (
            summary.get("pricing_basis")
            for summary in usage_summaries
            if isinstance(summary.get("pricing_basis"), dict)
        ),
        None,
    )

    estimated_cost_values = [
        summary.get("estimated_cost_usd")
        for summary in usage_summaries
        if isinstance(summary.get("estimated_cost_usd"), float)
    ]

    return {
        "model": document_model,
        "input_tokens": sum(int(summary.get("input_tokens") or 0) for summary in usage_summaries),
        "cached_input_tokens": sum(
            int(summary.get("cached_input_tokens") or 0) for summary in usage_summaries
        ),
        "billable_input_tokens": sum(
            int(summary.get("billable_input_tokens") or 0) for summary in usage_summaries
        ),
        "output_tokens": sum(
            int(summary.get("output_tokens") or 0) for summary in usage_summaries
        ),
        "reasoning_tokens": sum(
            int(summary.get("reasoning_tokens") or 0) for summary in usage_summaries
        ),
        "total_tokens": sum(int(summary.get("total_tokens") or 0) for summary in usage_summaries),
        "estimated_cost_usd": (
            round(sum(estimated_cost_values), 8) if estimated_cost_values else None
        ),
        "pricing_basis": pricing_basis,
        "page_count": len(usage_summaries),
    }


def _normalize_digits(value: str) -> str:
    return value.translate(ARABIC_DIGIT_TRANSLATION)


def _normalize_document_type(value: object) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in KNOWN_DOCUMENT_TYPES else "unknown"


def _normalize_date(value: object) -> str | None:
    if value is None:
        return None

    text = _normalize_digits(str(value).strip())
    if not text:
        return None

    normalized = re.sub(r"\s+", "", text).replace(".", "/").replace("-", "/")
    for date_format in ("%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(normalized, date_format).date().isoformat()
        except ValueError:
            continue

    return text


def _normalize_text_field(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_header(header: object) -> dict:
    if not isinstance(header, dict):
        return {}

    normalized_header: dict[str, object] = {}
    for key, value in header.items():
        if key in {"period_from", "period_to", "invoice_date"}:
            normalized_header[str(key)] = _normalize_date(value)
        elif value is None:
            normalized_header[str(key)] = None
        else:
            normalized_header[str(key)] = str(value).strip()
    return normalized_header


def _normalize_numeric_string(value: object) -> str | None:
    if value is None:
        return None

    text = _normalize_digits(str(value).strip())
    if not text:
        return None

    text = text.replace("٬", "").replace(",", "")
    text = text.replace("٫", ".")
    text = re.sub(r"\s+", "", text)
    return text or None


def _normalize_monthly_row(row: dict, page_number: int) -> dict:
    return {
        "date": _normalize_date(row.get("date")),
        "reference_number": _normalize_text_field(row.get("reference_number")),
        "note": _normalize_text_field(row.get("note")),
        "debit": _normalize_numeric_string(row.get("debit")),
        "credit": _normalize_numeric_string(row.get("credit")),
        "balance": _normalize_numeric_string(row.get("balance")),
        "page_number": page_number,
    }


def _normalize_daily_row(row: dict, page_number: int) -> dict:
    item_name = _normalize_text_field(row.get("item_name")) or _normalize_text_field(
        row.get("item_name_ar")
    )
    return {
        "movement_date": _normalize_date(row.get("movement_date")),
        "notice_number": _normalize_text_field(row.get("notice_number")),
        "invoice_number": _normalize_text_field(row.get("invoice_number")),
        "account_name": _normalize_text_field(row.get("account_name")),
        "item_name": item_name,
        "item_name_ar": item_name,
        "quantity": _normalize_numeric_string(row.get("quantity")),
        "price": _normalize_numeric_string(row.get("price")),
        "discount": _normalize_numeric_string(row.get("discount")),
        "total": _normalize_numeric_string(row.get("total")),
        "page_number": page_number,
    }


def _normalize_signature(signature_payload: object) -> dict:
    if not isinstance(signature_payload, dict):
        signature_payload = {}

    signature_present = bool(signature_payload.get("signature_present"))
    signer_name = _normalize_text_field(signature_payload.get("signer_name"))
    signature_note = _normalize_text_field(signature_payload.get("signature_note"))

    if signer_name:
        signature_present = True

    return {
        "signature_present": signature_present,
        "signer_name": signer_name,
        "signature_note": signature_note,
    }


def _normalize_page_payload(payload: dict, page_number: int) -> dict:
    page_document_type = _normalize_document_type(payload.get("page_document_type"))
    header = _normalize_header(payload.get("header"))
    footer_summary = (
        payload.get("footer_summary") if isinstance(payload.get("footer_summary"), dict) else {}
    )
    raw_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    signature = _normalize_signature(payload.get("signature"))

    if page_document_type == "monthly_statement":
        rows = [
            _normalize_monthly_row(row, page_number)
            for row in raw_rows
            if isinstance(row, dict)
        ]
    elif page_document_type == "daily_invoice":
        rows = [
            _normalize_daily_row(row, page_number)
            for row in raw_rows
            if isinstance(row, dict)
        ]
    else:
        rows = []

    return {
        "page_number": page_number,
        "page_document_type": page_document_type,
        "header": header,
        "rows": rows,
        "footer_summary": footer_summary,
        "signature": signature,
    }


def _page_cache_path(document_id: str, page_number: int) -> Path:
    return PROCESSED_OPENAI_JSON_DIR / f"{document_id}_direct_page_{page_number}.json"


def _document_output_path(document_id: str) -> Path:
    return PROCESSED_OPENAI_JSON_DIR / f"{document_id}_direct.json"


def _load_page_cache(document_id: str, page_number: int) -> dict | None:
    if not ENABLE_OPENAI_CACHE:
        return None

    cache_path = _page_cache_path(document_id, page_number)
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload if isinstance(payload, dict) else None


def _save_page_cache(document_id: str, page_number: int, payload: dict) -> None:
    PROCESSED_OPENAI_JSON_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _page_cache_path(document_id, page_number)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _call_openai_for_page(image_path: str, page_number: int) -> dict:
    api_key = OPENAI_API_KEY.strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    image_data_url = _encode_image_to_data_url(image_path)
    payload = {
        "model": OPENAI_MODEL.strip() or "gpt-4.1-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_prompt(page_number=page_number),
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
                "type": "json_schema",
                "name": "pharmacy_document_page",
                "strict": False,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "page_number",
                        "page_document_type",
                        "header",
                        "rows",
                        "footer_summary",
                        "signature",
                    ],
                    "properties": {
                        "page_number": {
                            "type": "integer",
                        },
                        "page_document_type": {
                            "type": "string",
                            "enum": ["monthly_statement", "daily_invoice", "unknown"],
                        },
                        "header": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "date": {"type": ["string", "null"]},
                                    "reference_number": {"type": ["string", "null"]},
                                    "note": {"type": ["string", "null"]},
                                    "debit": {"type": ["string", "null"]},
                                    "credit": {"type": ["string", "null"]},
                                    "balance": {"type": ["string", "null"]},
                                    "item_name": {"type": ["string", "null"]},
                                    "item_name_ar": {"type": ["string", "null"]},
                                    "quantity": {"type": ["string", "null"]},
                                    "price": {"type": ["string", "null"]},
                                    "discount": {"type": ["string", "null"]},
                                    "total": {"type": ["string", "null"]},
                                },
                            },
                        },
                        "footer_summary": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "signature": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "signature_present",
                                "signer_name",
                                "signature_note",
                            ],
                            "properties": {
                                "signature_present": {"type": "boolean"},
                                "signer_name": {"type": ["string", "null"]},
                                "signature_note": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
        "max_output_tokens": 2600,
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
    return response.json()


def _get_page_payload(
    document_id: str,
    image_path: str,
    page_number: int,
) -> tuple[dict, dict]:
    cached = _load_page_cache(document_id=document_id, page_number=page_number)
    if cached and isinstance(cached.get("normalized_page"), dict):
        _print_usage_summary(
            scope=f"page {page_number} (cache)",
            usage_summary=cached.get("usage_summary"),
        )
        return cached["normalized_page"], cached

    raw_response = _call_openai_for_page(image_path=image_path, page_number=page_number)
    response_text = _extract_response_text(raw_response)
    if not response_text:
        raise ValueError(f"OpenAI returned an empty response for page {page_number}.")

    parsed_payload = _parse_json_text(response_text)
    normalized_page = _normalize_page_payload(parsed_payload, page_number=page_number)
    usage_summary = _build_usage_summary(raw_response)
    cache_payload = {
        "page_number": page_number,
        "image_path": image_path,
        "normalized_page": normalized_page,
        "usage_summary": usage_summary,
        "raw_response": raw_response,
    }
    _save_page_cache(document_id=document_id, page_number=page_number, payload=cache_payload)
    _print_usage_summary(scope=f"page {page_number}", usage_summary=usage_summary)
    return normalized_page, cache_payload


def _page_confidence(page_payload: dict) -> float:
    page_document_type = page_payload.get("page_document_type")
    rows = page_payload.get("rows") or []
    footer_summary = page_payload.get("footer_summary") or {}

    if page_document_type == "unknown":
        return 0.25
    if rows:
        return 0.88
    if page_document_type == "monthly_statement" and footer_summary:
        return 0.72
    return 0.6


def _build_page_classification_result(page_payload: dict) -> PageDocumentTypeResult:
    page_document_type = str(page_payload.get("page_document_type", "unknown"))
    rows = page_payload.get("rows") or []
    footer_summary = page_payload.get("footer_summary") or {}

    evidence = [
        f"OpenAI direct extraction classified page as {page_document_type}.",
    ]
    if rows:
        evidence.append(f"OpenAI extracted {len(rows)} structured rows.")
    if footer_summary:
        evidence.append("Footer summary fields were detected.")

    return PageDocumentTypeResult(
        page_number=int(page_payload["page_number"]),
        document_type=page_document_type,
        confidence=_page_confidence(page_payload),
        evidence=evidence,
    )


def _build_signature_payload(document_id: str, pages: list[dict], image_paths: list[str]) -> dict:
    page_results: list[dict] = []
    signature_pages: list[int] = []

    for page_payload, image_path in zip(pages, image_paths):
        signature = page_payload.get("signature") or {}
        signature_present = bool(signature.get("signature_present"))
        signer_name = signature.get("signer_name")
        signature_note = signature.get("signature_note")
        page_number = int(page_payload["page_number"])

        if signature_present:
            signature_pages.append(page_number)

        page_results.append(
            {
                "page_number": page_number,
                "image_path": image_path,
                "signature_present": signature_present,
                "signer_name": signer_name,
                "signature_bbox": None,
                "nearby_text": [signature_note] if signature_note else [],
                "confidence": 0.9 if signature_present else 0.0,
            }
        )

    best_page_result = None
    if page_results:
        best_page_result = max(page_results, key=lambda item: item["confidence"])

    return {
        "document_id": document_id,
        "page_count": len(page_results),
        "signature_pages": sorted(signature_pages),
        "best_page_result": best_page_result,
        "pages": page_results,
        "provider": "openai_direct",
    }


def _page_route_plan_item(page_payload: dict) -> dict:
    page_document_type = str(page_payload.get("page_document_type", "unknown"))
    extractor_name = {
        "monthly_statement": "openai_direct_monthly_statement",
        "daily_invoice": "openai_direct_daily_invoice",
    }.get(page_document_type, "openai_direct_unknown")
    return {
        "page_number": int(page_payload["page_number"]),
        "page_document_type": page_document_type,
        "target_extractor": extractor_name,
    }


def _build_mixed_route_result(
    document_id: str,
    classification_result: DocumentTypeResult,
    pages: list[dict],
) -> dict:
    page_extractions = []
    for page_payload in pages:
        page_document_type = str(page_payload.get("page_document_type", "unknown"))
        rows = page_payload.get("rows") or []
        page_extractions.append(
            {
                "page_number": int(page_payload["page_number"]),
                "page_document_type": page_document_type,
                "extraction_result": {
                    "document_id": document_id,
                    "document_type": page_document_type,
                    "status": (
                        "extraction_completed"
                        if rows or page_payload.get("footer_summary")
                        else "no_rows_found"
                    ),
                    "row_count": len(rows),
                    "page_row_counts": {int(page_payload["page_number"]): len(rows)},
                    "header": page_payload.get("header") or {},
                    "footer_summary": page_payload.get("footer_summary") or {},
                    "rows": rows,
                    "warnings": [],
                },
            }
        )

    return {
        "document_id": document_id,
        "document_type": "mixed",
        "target_extractor": "openai_direct_mixed",
        "status": "extraction_completed" if page_extractions else "no_rows_found",
        "review_required": False,
        "confidence": classification_result.confidence,
        "evidence": classification_result.evidence,
        "mixed_details": (
            classification_result.mixed_details.model_dump()
            if classification_result.mixed_details
            else None
        ),
        "page_route_plan": [_page_route_plan_item(page_payload) for page_payload in pages],
        "extraction_result": {
            "page_extractions": page_extractions,
        },
    }


def _build_single_type_route_result(
    document_id: str,
    classification_result: DocumentTypeResult,
    pages: list[dict],
    document_type: str,
) -> dict:
    matching_pages = [
        page_payload for page_payload in pages if page_payload.get("page_document_type") == document_type
    ]
    rows: list[dict] = []
    page_row_counts: dict[int, int] = {}
    page_headers: dict[str, dict] = {}
    page_footer_summaries: dict[str, dict] = {}

    for page_payload in matching_pages:
        page_number = int(page_payload["page_number"])
        page_rows = page_payload.get("rows") or []
        rows.extend(page_rows)
        page_row_counts[page_number] = len(page_rows)
        page_headers[str(page_number)] = page_payload.get("header") or {}
        page_footer_summaries[str(page_number)] = page_payload.get("footer_summary") or {}

    header = next(
        (
            page_headers[key]
            for key in sorted(page_headers, key=lambda value: int(value))
            if page_headers[key]
        ),
        {},
    )
    footer_summary = next(
        (
            page_footer_summaries[key]
            for key in sorted(page_footer_summaries, key=lambda value: int(value), reverse=True)
            if page_footer_summaries[key]
        ),
        {},
    )

    status = "extraction_completed"
    if not rows and footer_summary and document_type == "monthly_statement":
        status = "summary_only_extraction"
    elif not rows:
        status = "no_rows_found"

    return {
        "document_id": document_id,
        "document_type": document_type,
        "target_extractor": f"openai_direct_{document_type}",
        "status": status,
        "review_required": False,
        "confidence": classification_result.confidence,
        "evidence": classification_result.evidence,
        "page_route_plan": [_page_route_plan_item(page_payload) for page_payload in matching_pages],
        "extraction_result": {
            "document_id": document_id,
            "document_type": document_type,
            "status": status,
            "row_count": len(rows),
            "page_row_counts": page_row_counts,
            "header": header,
            "footer_summary": footer_summary,
            "page_headers": page_headers,
            "page_footer_summaries": page_footer_summaries,
            "rows": rows,
            "warnings": [],
        },
    }


def _build_route_result(
    document_id: str,
    classification_result: DocumentTypeResult,
    pages: list[dict],
) -> dict:
    document_type = classification_result.document_type
    if document_type == "mixed":
        return _build_mixed_route_result(
            document_id=document_id,
            classification_result=classification_result,
            pages=pages,
        )

    if document_type in {"monthly_statement", "daily_invoice"}:
        return _build_single_type_route_result(
            document_id=document_id,
            classification_result=classification_result,
            pages=pages,
            document_type=document_type,
        )

    return {
        "document_id": document_id,
        "document_type": "unknown",
        "target_extractor": "openai_direct_unknown",
        "status": "no_rows_found",
        "review_required": True,
        "confidence": classification_result.confidence,
        "evidence": classification_result.evidence,
        "page_route_plan": [_page_route_plan_item(page_payload) for page_payload in pages],
        "extraction_result": {
            "document_id": document_id,
            "document_type": "unknown",
            "status": "no_rows_found",
            "row_count": 0,
            "page_row_counts": {},
            "header": {},
            "footer_summary": {},
            "rows": [],
            "warnings": ["OpenAI could not classify the document into a supported type."],
        },
    }


def run_openai_direct_extraction(
    document_id: str,
    image_paths: list[str],
    source: str = "preprocessed",
) -> dict:
    pages: list[dict] = []
    page_cache_payloads: list[dict] = []
    warnings: list[str] = []

    for page_number, image_path in enumerate(image_paths, start=1):
        normalized_page, cache_payload = _get_page_payload(
            document_id=document_id,
            image_path=image_path,
            page_number=page_number,
        )
        pages.append(normalized_page)
        page_cache_payloads.append(cache_payload)

    page_results = [_build_page_classification_result(page_payload) for page_payload in pages]
    (
        final_document_type,
        final_confidence,
        evidence,
        type_counts,
        mixed_details,
    ) = aggregate_document_type_from_pages(page_results)

    classification_result = DocumentTypeResult(
        document_type=final_document_type,
        confidence=final_confidence,
        evidence=evidence,
        type_counts=type_counts,
        page_results=page_results,
        mixed_details=(
            MixedClassificationDetails(**mixed_details.model_dump())
            if mixed_details is not None
            else None
        ),
    )

    route_result = _build_route_result(
        document_id=document_id,
        classification_result=classification_result,
        pages=pages,
    )
    usage_summary = _aggregate_usage_summaries(page_cache_payloads)
    signature_payload = _build_signature_payload(
        document_id=document_id,
        pages=pages,
        image_paths=image_paths,
    )

    output_path = _document_output_path(document_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_payload = {
        "document_id": document_id,
        "provider": "openai",
        "source": source,
        "page_count": len(image_paths),
        "pages": pages,
        "classification_result": classification_result.model_dump(),
        "route_result": route_result,
        "signature_payload": signature_payload,
        "usage_summary": usage_summary,
        "warnings": warnings,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document_payload, file, ensure_ascii=False, indent=2)

    _print_usage_summary(scope=f"document {document_id}", usage_summary=usage_summary)

    return {
        "document_id": document_id,
        "provider": "openai",
        "source": source,
        "page_count": len(image_paths),
        "pages": pages,
        "raw_output": page_cache_payloads,
        "classification_result": classification_result,
        "route_result": route_result,
        "signature_payload": signature_payload,
        "usage_summary": usage_summary,
        "output_json_path": str(output_path),
        "status": "completed",
        "warnings": warnings,
    }
