from __future__ import annotations

from datetime import datetime
import re
from typing import Any


DATE_CANDIDATE_PATTERN = re.compile(
    r"^(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}/\d{1,2}/\d{1,2}|\d{4}-\d{1,2}-\d{1,2})$"
)
NUMBER_CLEANUP_PATTERN = re.compile(r"[^0-9.\-]")
ID_CLEANUP_PATTERN = re.compile(r"[^0-9]")

ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _normalize_digits(value: str) -> str:
    return value.translate(ARABIC_DIGIT_TRANSLATION)


def _normalize_numeric_text(value: str) -> str:
    normalized = _normalize_digits(value)
    normalized = normalized.replace("٬", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("٫", ".")
    normalized = normalized.replace(" ", "")
    normalized = NUMBER_CLEANUP_PATTERN.sub("", normalized)
    return normalized


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = _normalize_numeric_text(value)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_valid_date(value: str | None) -> bool:
    if value is None:
        return False

    candidate = _normalize_digits(value.strip())
    if not DATE_CANDIDATE_PATTERN.match(candidate):
        return False

    for date_format in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            datetime.strptime(candidate, date_format)
            return True
        except ValueError:
            continue

    return False


def _normalize_id(value: str | None) -> str:
    if value is None:
        return ""
    normalized = _normalize_digits(value)
    normalized = ID_CLEANUP_PATTERN.sub("", normalized)
    return normalized


def _numeric_diff(value_a: float, value_b: float) -> float:
    return abs(value_a - value_b)


def _best_expected_total(
    quantity: float,
    price: float,
    discount: float | None,
) -> tuple[float, str]:
    subtotal = quantity * price
    candidates: list[tuple[float, str]] = [(subtotal, "quantity_x_price")]

    if discount is not None:
        candidates.append((subtotal - discount, "absolute_discount"))
        candidates.append((subtotal * (1 - (discount / 100.0)), "percent_discount"))

    best_value, best_method = candidates[0]
    return best_value, best_method


def validate_daily_invoice_extraction(
    extraction_result: dict[str, Any],
    total_tolerance: float = 1.0,
) -> dict[str, Any]:
    rows = extraction_result.get("rows") or []
    issues: list[str] = []
    warnings: list[str] = []
    row_checks: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row_issues: list[str] = []
        movement_date = row.get("movement_date")
        notice_number = row.get("notice_number")
        invoice_number = row.get("invoice_number")

        if movement_date and not _is_valid_date(movement_date):
            row_issues.append(f"row[{row_index}]: invalid movement_date '{movement_date}'.")

        if notice_number and len(_normalize_id(notice_number)) < 4:
            row_issues.append(f"row[{row_index}]: notice_number looks invalid '{notice_number}'.")

        if invoice_number and len(_normalize_id(invoice_number)) < 4:
            row_issues.append(f"row[{row_index}]: invoice_number looks invalid '{invoice_number}'.")

        quantity = _safe_float(row.get("quantity"))
        price = _safe_float(row.get("price"))
        discount = _safe_float(row.get("discount"))
        total = _safe_float(row.get("total"))

        if quantity is None:
            row_issues.append(f"row[{row_index}]: quantity is not numeric '{row.get('quantity')}'.")
        if price is None:
            row_issues.append(f"row[{row_index}]: price is not numeric '{row.get('price')}'.")
        if total is None:
            row_issues.append(f"row[{row_index}]: total is not numeric '{row.get('total')}'.")

        expected_total = None
        expected_method = None
        delta = None
        if quantity is not None and price is not None and total is not None:
            expected_total, expected_method = _best_expected_total(
                quantity=quantity,
                price=price,
                discount=discount,
            )
            delta = _numeric_diff(total, expected_total)

            if delta > total_tolerance:
                row_issues.append(
                    f"row[{row_index}]: total mismatch. expected {expected_total:.3f} "
                    f"({expected_method}), got {total:.3f}, delta={delta:.3f}."
                )

        row_checks.append(
            {
                "row_index": row_index,
                "row_page_number": row.get("page_number"),
                "expected_total": expected_total,
                "expected_method": expected_method,
                "delta": delta,
                "row_issue_count": len(row_issues),
                "row_issues": row_issues,
            }
        )
        issues.extend(row_issues)

    if not rows:
        warnings.append("No daily invoice rows available for validation.")

    status = "validation_passed" if not issues else "validation_failed"
    return {
        "status": status,
        "row_count": len(rows),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "row_checks": row_checks,
    }


def validate_monthly_statement_extraction(
    extraction_result: dict[str, Any],
) -> dict[str, Any]:
    rows = extraction_result.get("rows") or []
    issues: list[str] = []
    warnings: list[str] = []
    row_checks: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row_issues: list[str] = []

        row_date = row.get("date")
        if row_date and not _is_valid_date(row_date):
            row_issues.append(f"row[{row_index}]: invalid date '{row_date}'.")

        numeric_fields = ["debit", "credit", "balance"]
        parsed_numeric_count = 0
        for field_name in numeric_fields:
            raw_value = row.get(field_name)
            if raw_value is None:
                continue

            parsed_value = _safe_float(raw_value)
            if parsed_value is None:
                row_issues.append(
                    f"row[{row_index}]: {field_name} is not numeric '{raw_value}'."
                )
            else:
                parsed_numeric_count += 1

        if not row_date and parsed_numeric_count == 0:
            row_issues.append(
                f"row[{row_index}]: row has neither valid date nor numeric amounts."
            )

        row_checks.append(
            {
                "row_index": row_index,
                "row_page_number": row.get("page_number"),
                "row_issue_count": len(row_issues),
                "row_issues": row_issues,
            }
        )
        issues.extend(row_issues)

    if not rows:
        warnings.append("No monthly statement rows available for validation.")

    status = "validation_passed" if not issues else "validation_failed"
    return {
        "status": status,
        "row_count": len(rows),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "row_checks": row_checks,
    }


def validate_return_invoice_extraction(
    extraction_result: dict[str, Any],
) -> dict[str, Any]:
    rows = extraction_result.get("rows") or []
    issues: list[str] = []
    warnings: list[str] = []
    row_checks: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row_issues: list[str] = []

        row_date = row.get("date")
        reference_number = row.get("reference_number")

        if row_date and not _is_valid_date(row_date):
            row_issues.append(f"row[{row_index}]: invalid return date '{row_date}'.")

        if reference_number and len(_normalize_id(reference_number)) < 3:
            row_issues.append(
                f"row[{row_index}]: reference_number looks invalid '{reference_number}'."
            )

        row_checks.append(
            {
                "row_index": row_index,
                "row_page_number": row.get("page_number"),
                "row_issue_count": len(row_issues),
                "row_issues": row_issues,
            }
        )
        issues.extend(row_issues)

    if not rows:
        warnings.append("No return rows available for validation.")

    status = "validation_passed" if not issues else "validation_failed"
    return {
        "status": status,
        "row_count": len(rows),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "row_checks": row_checks,
    }


def validate_signature_result(
    signature_payload: dict[str, Any] | None,
    require_signature: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    signature_pages: list[int] = []
    if signature_payload is not None:
        raw_signature_pages = signature_payload.get("signature_pages") or []
        signature_pages = [int(page_number) for page_number in raw_signature_pages]

    signature_present = len(signature_pages) > 0

    if require_signature and not signature_present:
        issues.append("Signature is required for this document type, but none was detected.")

    if not require_signature and not signature_present:
        warnings.append("No signature detected (not required for this document type).")

    status = "validation_passed" if not issues else "validation_failed"
    return {
        "status": status,
        "require_signature": require_signature,
        "signature_present": signature_present,
        "signature_pages": signature_pages,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


def _should_require_signature(document_type: str) -> bool:
    # Business rule can be tuned later. Current default:
    # require signatures for daily and return invoices.
    return document_type in {"daily_invoice", "return_invoice"}


def _validate_by_document_type(
    document_type: str,
    extraction_result: dict[str, Any],
    total_tolerance: float,
) -> dict[str, Any]:
    if document_type == "daily_invoice":
        return validate_daily_invoice_extraction(
            extraction_result=extraction_result,
            total_tolerance=total_tolerance,
        )
    if document_type == "monthly_statement":
        return validate_monthly_statement_extraction(extraction_result=extraction_result)
    if document_type == "return_invoice":
        return validate_return_invoice_extraction(extraction_result=extraction_result)

    return {
        "status": "validation_passed",
        "row_count": 0,
        "issue_count": 0,
        "warning_count": 1,
        "issues": [],
        "warnings": [f"No validator configured for document type '{document_type}'."],
        "row_checks": [],
    }


def validate_extraction_route_result(
    route_result: dict[str, Any],
    signature_payload: dict[str, Any] | None = None,
    total_tolerance: float = 1.0,
) -> dict[str, Any]:
    document_id = str(route_result.get("document_id", ""))
    document_type = str(route_result.get("document_type", "unknown"))
    extraction_result = route_result.get("extraction_result") or {}

    issues: list[str] = []
    warnings: list[str] = []
    page_validations: list[dict[str, Any]] = []

    if document_type == "mixed":
        page_extractions = extraction_result.get("page_extractions") or []

        for page_item in page_extractions:
            page_number = page_item.get("page_number")
            page_document_type = str(page_item.get("page_document_type", "unknown"))
            page_extraction_result = page_item.get("extraction_result") or {}

            page_validation = _validate_by_document_type(
                document_type=page_document_type,
                extraction_result=page_extraction_result,
                total_tolerance=total_tolerance,
            )
            page_validations.append(
                {
                    "page_number": page_number,
                    "document_type": page_document_type,
                    "validation": page_validation,
                }
            )
            issues.extend(page_validation.get("issues", []))
            warnings.extend(page_validation.get("warnings", []))
    else:
        main_validation = _validate_by_document_type(
            document_type=document_type,
            extraction_result=extraction_result,
            total_tolerance=total_tolerance,
        )
        page_validations.append(
            {
                "page_number": None,
                "document_type": document_type,
                "validation": main_validation,
            }
        )
        issues.extend(main_validation.get("issues", []))
        warnings.extend(main_validation.get("warnings", []))

    signature_validation = validate_signature_result(
        signature_payload=signature_payload,
        require_signature=_should_require_signature(document_type),
    )
    issues.extend(signature_validation.get("issues", []))
    warnings.extend(signature_validation.get("warnings", []))

    status = "validation_passed" if not issues else "validation_failed"

    return {
        "document_id": document_id,
        "document_type": document_type,
        "status": status,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "page_validations": page_validations,
        "signature_validation": signature_validation,
    }
