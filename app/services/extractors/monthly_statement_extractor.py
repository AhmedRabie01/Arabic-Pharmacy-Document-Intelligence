import re

from app.schemas.document import DocumentTypeResult, SpatialMappingResponse


DATE_PATTERN = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}/\d{1,2}/\d{1,2})\b")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
ARABIC_LETTER_PATTERN = re.compile(r"[\u0621-\u064A]")


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = value.replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return None


def _split_row_text_to_parts(row_text: str) -> list[str]:
    if not row_text:
        return []

    parts = [part.strip() for part in row_text.split("|")]
    return [part for part in parts if part]


def _extract_date(parts: list[str]) -> str | None:
    for part in parts:
        date_match = DATE_PATTERN.search(part)
        if date_match:
            return date_match.group(0)

    return None


def _extract_number_candidates(parts: list[str]) -> list[str]:
    candidates: list[str] = []

    for part in parts:
        normalized = part.replace(",", "").strip()
        if NUMBER_PATTERN.match(normalized):
            candidates.append(normalized)

    return candidates


def _normalize_note_text(note_text: str, debit: str | None, credit: str | None) -> str:
    cleaned = note_text.strip()
    normalized = cleaned.replace(" ", "")

    debit_value = _safe_float(debit)
    credit_value = _safe_float(credit)

    if "مبيع" in normalized or "مبي" in normalized:
        return "\u0645\u0628\u064a\u0639\u0627\u062a"

    if credit_value is not None and credit_value > 0 and (debit_value is None or debit_value == 0):
        return "\u0627\u0633\u062a\u0644\u0627\u0645 \u0646\u0642\u062f\u064a\u0647"

    arabic_letters = ARABIC_LETTER_PATTERN.findall(cleaned)
    if len(arabic_letters) <= 3 and debit_value is not None and debit_value > 0:
        return "\u0645\u0628\u064a\u0639\u0627\u062a"

    return cleaned or "-"


def _build_statement_row(row_text: str, page_number: int) -> dict | None:
    parts = _split_row_text_to_parts(row_text)
    if not parts:
        return None

    date_value = _extract_date(parts)
    numeric_candidates = _extract_number_candidates(parts)

    reference_number = None
    note_text = None
    if len(parts) >= 2:
        reference_number = parts[1]
    if len(parts) >= 3:
        note_text = parts[2]

    row_data = {
        "date": date_value,
        "reference_number": reference_number,
        "note": note_text,
        "description": row_text.strip(),
        "debit": None,
        "credit": None,
        "balance": None,
        "page_number": page_number,
    }

    if len(numeric_candidates) >= 1:
        row_data["balance"] = numeric_candidates[-1]

    if len(numeric_candidates) >= 2:
        row_data["credit"] = numeric_candidates[-2]

    if len(numeric_candidates) >= 3:
        row_data["debit"] = numeric_candidates[-3]

    row_data["note"] = _normalize_note_text(
        note_text=safe_string(row_data["note"]),
        debit=row_data["debit"],
        credit=row_data["credit"],
    )

    has_amount = any(
        _safe_float(row_data[field_name]) is not None
        for field_name in ["debit", "credit", "balance"]
    )
    has_date_or_amount = bool(date_value) or has_amount

    if not has_date_or_amount:
        return None

    return row_data


def safe_string(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def extract_monthly_statement(
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult | None = None,
) -> dict:
    rows: list[dict] = []
    page_row_counts: dict[int, int] = {}
    warnings: list[str] = []

    for table in spatial_result.tables:
        page_number = table.page_number
        page_row_counts.setdefault(page_number, 0)

        for row in table.grouped_rows:
            if row.row_type in {"header", "summary", "empty"}:
                continue

            statement_row = _build_statement_row(
                row_text=row.row_text,
                page_number=page_number,
            )
            if statement_row is None:
                continue

            rows.append(statement_row)
            page_row_counts[page_number] += 1

    if not rows:
        warnings.append("No monthly-statement rows were detected from grouped rows.")

    if classification_result is not None:
        if classification_result.document_type not in {"monthly_statement", "mixed"}:
            warnings.append(
                "Classifier result is not monthly_statement/mixed; monthly extraction may be unreliable."
            )

    return {
        "document_id": spatial_result.document_id,
        "document_type": "monthly_statement",
        "status": "extraction_completed" if rows else "no_rows_found",
        "row_count": len(rows),
        "page_row_counts": page_row_counts,
        "rows": rows,
        "warnings": warnings,
    }
