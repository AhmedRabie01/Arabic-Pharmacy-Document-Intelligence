import json
import re
from pathlib import Path

from app.core.config import PROCESSED_CLASSIFICATION_JSON_DIR
from app.schemas.document import (
    DocumentTypeResult,
    MixedClassificationDetails,
    PageDocumentTypeResult,
    SpatialMappingResponse,
)


MONTHLY_STATEMENT_KEYWORDS = [
    "\u0643\u0634\u0641 \u062d\u0633\u0627\u0628",
    "\u0631\u0635\u064a\u062f \u0633\u0627\u0628\u0642",
    "\u0645\u062f\u064a\u0646",
    "\u062f\u0627\u0626\u0646",
    "\u0645\u0646 \u062a\u0627\u0631\u064a\u062e",
    "\u0627\u0644\u0649 \u062a\u0627\u0631\u064a\u062e",
]

DAILY_INVOICE_KEYWORDS = [
    "\u0641\u0627\u062a\u0648\u0631\u0629 \u0628\u064a\u0639",
    "\u0627\u0644\u0635\u0646\u0641",
    "\u0627\u0644\u0643\u0645\u064a\u0629",
    "\u0627\u0644\u0633\u0639\u0631",
    "\u0627\u0644\u0627\u062c\u0645\u0627\u0644\u0649",
    "\u0627\u0644\u0627\u062c\u0645\u0627\u0644\u064a",
    "\u0627\u0633\u0645 \u0627\u0644\u0639\u0645\u064a\u0644",
]
STRONG_DAILY_INVOICE_KEYWORDS = {
    "\u0641\u0627\u062a\u0648\u0631\u0629 \u0628\u064a\u0639",
    "\u0627\u0644\u0635\u0646\u0641",
    "\u0627\u0644\u0643\u0645\u064a\u0629",
    "\u0627\u0644\u0633\u0639\u0631",
}

RETURN_KEYWORDS = [
    "\u0645\u0631\u062a\u062c\u0639",
    "\u0627\u0634\u0639\u0627\u0631 \u062e\u0635\u0645",
    "\u0631\u062f",
    "\u062a\u0633\u0648\u064a\u0647",
]
STRONG_RETURN_KEYWORDS = {
    "\u0645\u0631\u062a\u062c\u0639",
    "\u0627\u0634\u0639\u0627\u0631 \u062e\u0635\u0645",
    "\u062a\u0633\u0648\u064a\u0647",
}
WEAK_RETURN_KEYWORDS = {
    "\u0631\u062f",
}

PAGE_HEADER_PATTERN = re.compile(r"^\[PAGE\s+(\d+)\]\s*$", re.IGNORECASE)

CLASSIFICATION_TYPES = [
    "daily_invoice",
    "monthly_statement",
    "return_invoice",
    "mixed",
    "unknown",
]


def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.lower()
    replacements = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064a",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
    }

    for source_char, target_char in replacements.items():
        normalized = normalized.replace(source_char, target_char)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def find_keyword_matches(
    normalized_text: str,
    raw_keywords: list[str],
) -> list[str]:
    matches: list[str] = []

    for raw_keyword in raw_keywords:
        normalized_keyword = normalize_arabic_text(raw_keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            matches.append(raw_keyword)

    return matches


def contains_keyword_as_token(
    normalized_text: str,
    normalized_keyword: str,
) -> bool:
    escaped_keyword = re.escape(normalized_keyword)
    pattern = rf"(?<![\u0621-\u064A]){escaped_keyword}(?![\u0621-\u064A])"
    return re.search(pattern, normalized_text) is not None


def find_return_keyword_matches(normalized_text: str) -> list[str]:
    matches: list[str] = []

    for raw_keyword in RETURN_KEYWORDS:
        normalized_keyword = normalize_arabic_text(raw_keyword)
        if not normalized_keyword:
            continue

        if raw_keyword in WEAK_RETURN_KEYWORDS:
            if contains_keyword_as_token(normalized_text, normalized_keyword):
                matches.append(raw_keyword)
            continue

        if normalized_keyword in normalized_text:
            matches.append(raw_keyword)

    return matches


def split_extracted_text_by_page(extracted_text: str | None) -> dict[int, str]:
    if not extracted_text:
        return {}

    page_text_map: dict[int, str] = {}
    current_page_number: int | None = None
    current_lines: list[str] = []

    for raw_line in extracted_text.splitlines():
        line = raw_line.strip()
        page_header_match = PAGE_HEADER_PATTERN.match(line)

        if page_header_match:
            if current_page_number is not None:
                page_text_map[current_page_number] = "\n".join(current_lines).strip()

            current_page_number = int(page_header_match.group(1))
            current_lines = []
            continue

        if current_page_number is not None:
            current_lines.append(raw_line)

    if current_page_number is not None:
        page_text_map[current_page_number] = "\n".join(current_lines).strip()

    if not page_text_map:
        page_text_map[1] = extracted_text.strip()

    return page_text_map


def get_page_numbers_for_classification(
    spatial_result: SpatialMappingResponse,
    extracted_text_by_page: dict[int, str],
) -> list[int]:
    page_numbers: set[int] = set()

    for table in spatial_result.tables:
        page_numbers.add(table.page_number)

    for page_number in extracted_text_by_page.keys():
        page_numbers.add(page_number)

    if spatial_result.page_count > 0:
        for page_number in range(1, spatial_result.page_count + 1):
            page_numbers.add(page_number)

    if not page_numbers:
        return [1]

    return sorted(page_numbers)


def count_daily_invoice_rows_for_page(
    spatial_result: SpatialMappingResponse,
    page_number: int,
) -> int:
    complete_rows = 0

    for table in spatial_result.tables:
        if table.page_number != page_number:
            continue

        for extracted_row in table.extracted_rows:
            if (
                extracted_row.item_name
                and extracted_row.quantity
                and extracted_row.price
                and extracted_row.total
            ):
                complete_rows += 1

    return complete_rows


def collect_page_text(
    spatial_result: SpatialMappingResponse,
    page_number: int,
    extracted_text_by_page: dict[int, str],
) -> str:
    text_parts: list[str] = []

    page_extracted_text = extracted_text_by_page.get(page_number)
    if page_extracted_text:
        text_parts.append(page_extracted_text)

    for table in spatial_result.tables:
        if table.page_number != page_number:
            continue

        for row in table.grouped_rows:
            if row.row_text:
                text_parts.append(row.row_text)

    return "\n".join(text_parts)


def canonical_document_type(document_type: str) -> str:
    if document_type == "mixed_document":
        return "mixed"

    if document_type not in CLASSIFICATION_TYPES:
        return "unknown"

    return document_type


def has_strong_return_signal(return_matches: list[str]) -> bool:
    return any(match in STRONG_RETURN_KEYWORDS for match in return_matches)


def has_strong_daily_signal(daily_matches: list[str]) -> bool:
    strong_match_count = sum(
        1
        for match in daily_matches
        if match in STRONG_DAILY_INVOICE_KEYWORDS
    )
    return strong_match_count >= 2 or "\u0641\u0627\u062a\u0648\u0631\u0629 \u0628\u064a\u0639" in daily_matches


def classify_page_document_type(
    spatial_result: SpatialMappingResponse,
    page_number: int,
    extracted_text_by_page: dict[int, str],
) -> PageDocumentTypeResult:
    evidence: list[str] = []

    complete_daily_rows = count_daily_invoice_rows_for_page(
        spatial_result=spatial_result,
        page_number=page_number,
    )
    has_daily_invoice_signal = complete_daily_rows > 0

    if has_daily_invoice_signal:
        evidence.append(
            f"Detected {complete_daily_rows} structured rows with item_name, quantity, price, and total."
        )

    page_text = collect_page_text(
        spatial_result=spatial_result,
        page_number=page_number,
        extracted_text_by_page=extracted_text_by_page,
    )
    normalized_page_text = normalize_arabic_text(page_text)

    daily_matches = find_keyword_matches(
        normalized_text=normalized_page_text,
        raw_keywords=DAILY_INVOICE_KEYWORDS,
    )
    monthly_matches = find_keyword_matches(
        normalized_text=normalized_page_text,
        raw_keywords=MONTHLY_STATEMENT_KEYWORDS,
    )
    return_matches = find_return_keyword_matches(normalized_page_text)

    if daily_matches:
        evidence.append(
            "Found daily invoice keywords: " + ", ".join(daily_matches)
        )

    has_daily_keyword_signal = has_strong_daily_signal(daily_matches)

    if monthly_matches:
        evidence.append(
            "Found monthly statement keywords: " + ", ".join(monthly_matches)
        )

    if return_matches:
        evidence.append(
            "Found return keywords: " + ", ".join(return_matches)
        )

    signal_count = sum(
        [
            1 if (has_daily_invoice_signal or has_daily_keyword_signal) else 0,
            1 if monthly_matches else 0,
            1 if return_matches else 0,
        ]
    )

    if signal_count == 0:
        return PageDocumentTypeResult(
            page_number=page_number,
            document_type="unknown",
            confidence=0.2,
            evidence=["No strong document-type signals were detected."],
        )

    # Return invoices often use the same line-item table shape as daily invoices.
    # When we see a strong return keyword plus structured rows on the same page,
    # prefer return_invoice instead of escalating to mixed.
    if (
        (has_daily_invoice_signal or has_daily_keyword_signal)
        and return_matches
        and not monthly_matches
        and has_strong_return_signal(return_matches)
    ):
        evidence.append(
            "Strong return keywords override the generic daily-table signal on this page."
        )
        confidence = 0.9 if complete_daily_rows >= 2 else 0.8
        return PageDocumentTypeResult(
            page_number=page_number,
            document_type="return_invoice",
            confidence=confidence,
            evidence=evidence,
        )

    if signal_count > 1:
        evidence.append("Multiple type signals were detected on this page.")
        return PageDocumentTypeResult(
            page_number=page_number,
            document_type="mixed",
            confidence=0.65,
            evidence=evidence,
        )

    if has_daily_invoice_signal or has_daily_keyword_signal:
        confidence = 0.9 if complete_daily_rows >= 2 else 0.8
        return PageDocumentTypeResult(
            page_number=page_number,
            document_type="daily_invoice",
            confidence=confidence,
            evidence=evidence,
        )

    if monthly_matches:
        confidence = 0.85 if len(monthly_matches) >= 2 else 0.75
        return PageDocumentTypeResult(
            page_number=page_number,
            document_type="monthly_statement",
            confidence=confidence,
            evidence=evidence,
        )

    confidence = 0.85 if len(return_matches) >= 2 else 0.75
    return PageDocumentTypeResult(
        page_number=page_number,
        document_type="return_invoice",
        confidence=confidence,
        evidence=evidence,
    )


def build_type_counts(page_results: list[PageDocumentTypeResult]) -> dict[str, int]:
    type_counts = {document_type: 0 for document_type in CLASSIFICATION_TYPES}

    for page_result in page_results:
        normalized_type = canonical_document_type(page_result.document_type)
        type_counts[normalized_type] += 1

    return type_counts


def aggregate_document_type_from_pages(
    page_results: list[PageDocumentTypeResult],
) -> tuple[str, float, list[str], dict[str, int], MixedClassificationDetails | None]:
    if not page_results:
        return (
            "unknown",
            0.2,
            ["No pages were available for classification."],
            {document_type: 0 for document_type in CLASSIFICATION_TYPES},
            None,
        )

    type_counts = build_type_counts(page_results)
    primary_types = [
        document_type
        for document_type in ["daily_invoice", "monthly_statement", "return_invoice"]
        if type_counts[document_type] > 0
    ]

    has_mixed_pages = type_counts["mixed"] > 0

    if has_mixed_pages or len(primary_types) > 1:
        final_document_type = "mixed"
    elif len(primary_types) == 1:
        final_document_type = primary_types[0]
    else:
        final_document_type = "unknown"

    all_confidences = [page_result.confidence for page_result in page_results]
    average_confidence = sum(all_confidences) / len(all_confidences)

    if final_document_type == "mixed":
        final_confidence = round(max(0.65, min(0.9, average_confidence)), 2)
    elif final_document_type == "unknown":
        final_confidence = 0.2
    else:
        matched_confidences = [
            page_result.confidence
            for page_result in page_results
            if canonical_document_type(page_result.document_type) == final_document_type
        ]
        if matched_confidences:
            final_confidence = round(
                sum(matched_confidences) / len(matched_confidences), 2
            )
        else:
            final_confidence = round(average_confidence, 2)

    evidence: list[str] = [
        f"Page {page_result.page_number}: "
        f"{canonical_document_type(page_result.document_type)} "
        f"(confidence={page_result.confidence:.2f})"
        for page_result in page_results
    ]

    mixed_details: MixedClassificationDetails | None = None
    if final_document_type == "mixed":
        detected_types = [
            document_type
            for document_type in ["daily_invoice", "monthly_statement", "return_invoice", "mixed"]
            if type_counts[document_type] > 0
        ]

        notes: list[str] = []
        if has_mixed_pages:
            notes.append("At least one page contained multiple document-type signals.")
        if len(primary_types) > 1:
            notes.append("Different primary document types appeared across pages.")

        mixed_details = MixedClassificationDetails(
            detected_types=detected_types,
            recommended_flow="split_by_page_then_extract",
            notes=notes,
        )
        evidence.append(
            "Document classified as mixed because multiple document-type signals were detected."
        )

    return final_document_type, final_confidence, evidence, type_counts, mixed_details


def classify_document_type(
    spatial_result: SpatialMappingResponse,
    extracted_text: str | None = None,
) -> DocumentTypeResult:
    extracted_text_by_page = split_extracted_text_by_page(extracted_text)
    page_numbers = get_page_numbers_for_classification(
        spatial_result=spatial_result,
        extracted_text_by_page=extracted_text_by_page,
    )

    page_results = [
        classify_page_document_type(
            spatial_result=spatial_result,
            page_number=page_number,
            extracted_text_by_page=extracted_text_by_page,
        )
        for page_number in page_numbers
    ]

    (
        final_document_type,
        final_confidence,
        evidence,
        type_counts,
        mixed_details,
    ) = aggregate_document_type_from_pages(page_results)

    return DocumentTypeResult(
        document_type=final_document_type,
        confidence=final_confidence,
        evidence=evidence,
        type_counts=type_counts,
        page_results=page_results,
        mixed_details=mixed_details,
    )


def get_classification_output_path(document_id: str) -> Path:
    return PROCESSED_CLASSIFICATION_JSON_DIR / f"{document_id}.json"


def save_document_classification_result(
    document_id: str,
    classification_result: DocumentTypeResult,
) -> str:
    output_path = get_classification_output_path(document_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(classification_result, "model_dump"):
        classification_data = classification_result.model_dump()
    else:
        classification_data = classification_result.dict()

    payload = {
        "document_id": document_id,
        **classification_data,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def classify_and_save_document_type(
    document_id: str,
    spatial_result: SpatialMappingResponse,
    extracted_text: str | None = None,
) -> tuple[DocumentTypeResult, str]:
    classification_result = classify_document_type(
        spatial_result=spatial_result,
        extracted_text=extracted_text,
    )
    output_path = save_document_classification_result(
        document_id=document_id,
        classification_result=classification_result,
    )
    return classification_result, output_path
