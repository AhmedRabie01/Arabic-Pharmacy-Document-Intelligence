from app.schemas.document import DocumentTypeResult, SpatialMappingResponse


def _to_row_dict(row: object) -> dict:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    return {}


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = value.replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return None


def _split_row_text(row_text: str) -> list[str]:
    return [
        part.strip()
        for part in row_text.split("|")
        if part.strip()
    ]


def _looks_like_header_or_summary(row_text: str) -> bool:
    normalized = row_text.replace(" ", "")

    if not normalized:
        return True

    blocked_tokens = [
        "الصنف",
        "الكمية",
        "السعر",
        "الخصم",
        "الاجمالى",
        "الاجمالي",
        "الاجمالي",
        "اسم",
        "الصلاحيه",
        "الصالحيه",
    ]

    return any(token in normalized for token in blocked_tokens)


def _extract_daily_row_from_grouped_text(
    row_text: str,
    page_number: int,
    table_bbox: list[float],
) -> dict | None:
    parts = _split_row_text(row_text)
    if len(parts) < 4:
        return None

    numeric_parts = [part for part in parts if _safe_float(part) is not None]
    text_parts = [part for part in parts if _safe_float(part) is None]

    if len(numeric_parts) < 3 or not text_parts:
        return None

    item_name = " ".join(text_parts).strip()
    if not item_name:
        return None

    quantity = None
    price = None
    discount = None
    total = None

    if len(numeric_parts) >= 4:
        quantity, price, discount, total = numeric_parts[-4:]
    else:
        price, discount, total = numeric_parts[-3:]

    return {
        "movement_date": None,
        "notice_number": None,
        "invoice_number": None,
        "account_name": None,
        "item_name": item_name,
        "quantity": quantity,
        "price": price,
        "discount": discount,
        "total": total,
        "page_number": page_number,
        "table_bbox": table_bbox,
        "raw_row_text": row_text,
    }


def extract_daily_invoice(
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult | None = None,
) -> dict:
    extracted_rows: list[dict] = []
    page_row_counts: dict[int, int] = {}
    warnings: list[str] = []

    for table in spatial_result.tables:
        page_number = table.page_number
        page_row_counts.setdefault(page_number, 0)

        for row in table.extracted_rows:
            row_dict = _to_row_dict(row)
            row_dict["page_number"] = page_number
            row_dict["table_bbox"] = table.table_bbox
            extracted_rows.append(row_dict)
            page_row_counts[page_number] += 1

        if page_row_counts[page_number] == 0:
            for grouped_row in table.grouped_rows:
                if grouped_row.row_type in {"header", "summary", "empty"}:
                    continue

                if _looks_like_header_or_summary(grouped_row.row_text):
                    continue

                fallback_row = _extract_daily_row_from_grouped_text(
                    row_text=grouped_row.row_text,
                    page_number=page_number,
                    table_bbox=table.table_bbox,
                )
                if fallback_row is None:
                    continue

                extracted_rows.append(fallback_row)
                page_row_counts[page_number] += 1

    numeric_totals = [
        total_value
        for total_value in (
            _safe_float(row.get("total"))
            for row in extracted_rows
        )
        if total_value is not None
    ]
    computed_total_sum = round(sum(numeric_totals), 3) if numeric_totals else None

    if not extracted_rows:
        warnings.append("No structured daily-invoice rows were found.")
    elif any("raw_row_text" in row for row in extracted_rows):
        warnings.append(
            "Some daily invoice rows were recovered from grouped row text because "
            "column-level extraction was incomplete."
        )

    if classification_result is not None:
        if classification_result.document_type not in {"daily_invoice", "mixed"}:
            warnings.append(
                "Classifier result is not daily_invoice/mixed; daily extraction may be unreliable."
            )

    return {
        "document_id": spatial_result.document_id,
        "document_type": "daily_invoice",
        "status": "extraction_completed" if extracted_rows else "no_rows_found",
        "row_count": len(extracted_rows),
        "page_row_counts": page_row_counts,
        "rows": extracted_rows,
        "computed_total_sum": computed_total_sum,
        "warnings": warnings,
    }
