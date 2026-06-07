from __future__ import annotations

from pathlib import Path
import json
import re
from collections import Counter, defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (
    APPROVED_DATASET_DIR,
    PROCESSED_CLASSIFICATION_JSON_DIR,
    PROCESSED_EXTRACTION_JSON_DIR,
)

APPROVED_DATASET_PATH = APPROVED_DATASET_DIR / "approved_real_v2.jsonl"
CLASSIFICATION_DIR = PROCESSED_CLASSIFICATION_JSON_DIR
EXTRACTION_DIR = PROCESSED_EXTRACTION_JSON_DIR
OUTPUT_PATH = APPROVED_DATASET_DIR / "evaluation_report.json"

ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).translate(ARABIC_DIGIT_TRANSLATION).strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_numeric(value: object) -> str:
    text = _normalize_text(value)
    text = text.replace(",", "")
    return text


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_approved_records() -> list[dict]:
    records: list[dict] = []

    with APPROVED_DATASET_PATH.open("r", encoding="utf-8-sig") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))

    return records


def get_page_classification(classification_payload: dict | None, page_number: int) -> str | None:
    if not classification_payload:
        return None

    for page_result in classification_payload.get("page_results") or []:
        if int(page_result.get("page_number", 0)) == page_number:
            return page_result.get("document_type")

    return None


def get_page_extraction(extraction_payload: dict | None, page_number: int) -> dict | None:
    if not extraction_payload:
        return None

    document_type = extraction_payload.get("document_type")
    extraction_result = extraction_payload.get("extraction_result") or {}

    if document_type == "mixed":
        for page_item in extraction_result.get("page_extractions") or []:
            if int(page_item.get("page_number", 0)) == page_number:
                page_extraction_result = page_item.get("extraction_result") or {}
                return {
                    "document_type": page_item.get("page_document_type"),
                    **page_extraction_result,
                }
        return None

    page_counts = extraction_result.get("page_row_counts") or {}
    if document_type and (
        str(page_number) in page_counts
        or page_number in page_counts
        or extraction_payload.get("document_type") == document_type
    ):
        return {
            "document_type": document_type,
            **extraction_result,
        }

    return None


def compare_rows(expected_rows: list[dict], actual_rows: list[dict], document_type: str) -> dict:
    if document_type == "monthly_statement":
        fields = ["date", "reference_number", "note", "debit", "credit", "balance"]
        normalizer = lambda field, value: _normalize_numeric(value) if field in {"debit", "credit", "balance"} else _normalize_text(value)
    else:
        fields = ["item_name_ar", "quantity", "price", "discount", "total"]

        def normalizer(field: str, value: object) -> str:
            if field == "item_name_ar":
                return _normalize_text(value)
            return _normalize_numeric(value)

    expected_count = len(expected_rows)
    actual_count = len(actual_rows)
    matched_fields = 0
    total_fields = expected_count * len(fields)
    row_mismatches: list[dict] = []

    for row_index, expected_row in enumerate(expected_rows):
        actual_row = actual_rows[row_index] if row_index < actual_count else {}
        field_mismatches: list[str] = []

        for field_name in fields:
            expected_value = normalizer(field_name, expected_row.get(field_name))
            actual_value = normalizer(field_name, actual_row.get(field_name))
            if expected_value == actual_value:
                matched_fields += 1
            else:
                field_mismatches.append(field_name)

        if field_mismatches:
            row_mismatches.append(
                {
                    "row_index": row_index,
                    "mismatched_fields": field_mismatches,
                }
            )

    accuracy = round(matched_fields / total_fields, 4) if total_fields else 1.0
    return {
        "expected_row_count": expected_count,
        "actual_row_count": actual_count,
        "row_count_match": expected_count == actual_count,
        "field_accuracy": accuracy,
        "row_mismatches": row_mismatches,
    }


def compare_footer_summary(expected_footer: dict, actual_footer: dict) -> dict:
    comparable_keys = sorted(set(expected_footer) & set(actual_footer))
    if not comparable_keys:
        return {
            "comparable_field_count": 0,
            "matched_field_count": 0,
            "field_accuracy": 1.0,
            "mismatches": [],
        }

    matched_field_count = 0
    mismatches: list[str] = []
    for key in comparable_keys:
        expected_value = _normalize_numeric(expected_footer.get(key))
        actual_value = _normalize_numeric(actual_footer.get(key))
        if expected_value == actual_value:
            matched_field_count += 1
        else:
            mismatches.append(key)

    return {
        "comparable_field_count": len(comparable_keys),
        "matched_field_count": matched_field_count,
        "field_accuracy": round(matched_field_count / len(comparable_keys), 4),
        "mismatches": mismatches,
    }


def build_document_report(document_id: str, records: list[dict]) -> dict:
    classification_payload = _load_json(CLASSIFICATION_DIR / f"{document_id}.json")
    extraction_payload = _load_json(EXTRACTION_DIR / f"{document_id}.json")
    page_reports: list[dict] = []

    for record in records:
        page_number = int(record["page_number"])
        expected_type = record["document_type"]
        actual_type = get_page_classification(classification_payload, page_number)
        actual_page_extraction = get_page_extraction(extraction_payload, page_number)
        actual_rows = (actual_page_extraction or {}).get("rows") or []
        actual_footer = (actual_page_extraction or {}).get("footer_summary") or {}

        row_comparison = compare_rows(
            expected_rows=record.get("rows") or [],
            actual_rows=actual_rows,
            document_type=expected_type,
        )
        footer_comparison = compare_footer_summary(
            expected_footer=record.get("footer_summary") or {},
            actual_footer=actual_footer,
        )

        page_reports.append(
            {
                "page_number": page_number,
                "expected_document_type": expected_type,
                "actual_document_type": actual_type,
                "document_type_match": expected_type == actual_type,
                "expected_content_profile": record.get("content_profile"),
                "row_comparison": row_comparison,
                "footer_comparison": footer_comparison,
            }
        )

    return {
        "document_id": document_id,
        "classification_found": classification_payload is not None,
        "extraction_found": extraction_payload is not None,
        "page_reports": page_reports,
    }


def summarize_reports(document_reports: list[dict]) -> dict:
    total_pages = 0
    evaluated_pages = 0
    document_type_matches = 0
    exact_row_count_matches = 0
    footer_accuracy_values: list[float] = []
    row_field_accuracy_values: list[float] = []
    missing_documents: list[str] = []
    page_type_counter = Counter()

    for document_report in document_reports:
        if not document_report["classification_found"] or not document_report["extraction_found"]:
            missing_documents.append(document_report["document_id"])

        for page_report in document_report["page_reports"]:
            total_pages += 1
            page_type_counter[page_report["expected_document_type"]] += 1
            is_evaluated_page = (
                document_report["classification_found"]
                and document_report["extraction_found"]
                and page_report["actual_document_type"] is not None
            )

            if not is_evaluated_page:
                continue

            evaluated_pages += 1

            if page_report["document_type_match"]:
                document_type_matches += 1

            if page_report["row_comparison"]["row_count_match"]:
                exact_row_count_matches += 1

            row_field_accuracy_values.append(page_report["row_comparison"]["field_accuracy"])
            footer_accuracy_values.append(page_report["footer_comparison"]["field_accuracy"])

    return {
        "approved_page_count": total_pages,
        "evaluated_page_count": evaluated_pages,
        "evaluation_coverage": round(evaluated_pages / total_pages, 4) if total_pages else 0.0,
        "approved_pages_by_type": dict(page_type_counter),
        "document_type_accuracy": round(document_type_matches / evaluated_pages, 4) if evaluated_pages else 0.0,
        "row_count_exact_match_rate": round(exact_row_count_matches / evaluated_pages, 4) if evaluated_pages else 0.0,
        "average_row_field_accuracy": round(sum(row_field_accuracy_values) / len(row_field_accuracy_values), 4) if row_field_accuracy_values else 0.0,
        "average_footer_field_accuracy": round(sum(footer_accuracy_values) / len(footer_accuracy_values), 4) if footer_accuracy_values else 0.0,
        "documents_missing_outputs": sorted(missing_documents),
    }


def main() -> int:
    approved_records = load_approved_records()
    records_by_document: dict[str, list[dict]] = defaultdict(list)
    for record in approved_records:
        records_by_document[record["document_id"]].append(record)

    document_reports = [
        build_document_report(document_id=document_id, records=records)
        for document_id, records in sorted(records_by_document.items())
    ]

    report = {
        "dataset_path": str(APPROVED_DATASET_PATH),
        "generated_at": "2026-06-05",
        "summary": summarize_reports(document_reports),
        "documents": document_reports,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(str(OUTPUT_PATH))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
