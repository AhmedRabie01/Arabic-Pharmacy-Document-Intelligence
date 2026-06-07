import re

from app.schemas.document import DocumentTypeResult, SpatialMappingResponse


RETURN_KEYWORDS = [
    "\u0645\u0631\u062a\u062c\u0639",
    "\u0627\u0634\u0639\u0627\u0631 \u062e\u0635\u0645",
    "\u0631\u062f",
    "\u062a\u0633\u0648\u064a\u0647",
]
DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.lower()
    replacements = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064a",
    }

    for source_char, target_char in replacements.items():
        normalized = normalized.replace(source_char, target_char)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _contains_return_keyword(text: str) -> bool:
    normalized_text = _normalize_text(text)

    for keyword in RETURN_KEYWORDS:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            return True

    return False


def _to_row_dict(row: object) -> dict:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    return {}


def _extract_notice_number(row_text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(row_text)
    if not matches:
        return None
    return matches[0]


def _extract_date(row_text: str) -> str | None:
    match = DATE_PATTERN.search(row_text)
    if match:
        return match.group(0)
    return None


def _build_return_row_from_text(row_text: str, page_number: int) -> dict | None:
    if not row_text.strip():
        return None

    if not _contains_return_keyword(row_text):
        return None

    return {
        "page_number": page_number,
        "date": _extract_date(row_text),
        "reference_number": _extract_notice_number(row_text),
        "description": row_text.strip(),
        "raw_row_text": row_text,
    }


def extract_return_invoice(
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult | None = None,
) -> dict:
    rows: list[dict] = []
    page_row_counts: dict[int, int] = {}
    warnings: list[str] = []

    for table in spatial_result.tables:
        page_number = table.page_number
        page_row_counts.setdefault(page_number, 0)

        for mapped_row in table.grouped_rows:
            if mapped_row.row_type in {"header", "empty"}:
                continue

            return_row = _build_return_row_from_text(
                row_text=mapped_row.row_text,
                page_number=page_number,
            )
            if return_row is None:
                continue

            rows.append(return_row)
            page_row_counts[page_number] += 1

        # Fallback signal: structured rows can still be useful when keyword OCR is weak.
        if page_row_counts[page_number] == 0 and table.extracted_rows:
            for extracted_row in table.extracted_rows:
                extracted_row_dict = _to_row_dict(extracted_row)
                fallback_row = {
                    "page_number": page_number,
                    "date": extracted_row_dict.get("movement_date"),
                    "reference_number": extracted_row_dict.get("notice_number"),
                    "description": extracted_row_dict.get("item_name"),
                    "raw_row_text": None,
                }
                rows.append(fallback_row)
                page_row_counts[page_number] += 1

    if not rows:
        warnings.append("No return rows were detected.")

    if classification_result is not None:
        if classification_result.document_type not in {"return_invoice", "mixed"}:
            warnings.append(
                "Classifier result is not return_invoice/mixed; return extraction may be unreliable."
            )

    return {
        "document_id": spatial_result.document_id,
        "document_type": "return_invoice",
        "status": "extraction_completed" if rows else "no_rows_found",
        "row_count": len(rows),
        "page_row_counts": page_row_counts,
        "rows": rows,
        "warnings": warnings,
    }
