from app.schemas.document import DocumentTypeResult, SpatialMappingResponse
from app.services.extractors.daily_invoice_extractor import extract_daily_invoice
from app.services.extractors.mixed_extractor import extract_mixed_document
from app.services.extractors.monthly_statement_extractor import (
    extract_monthly_statement,
)
from app.services.extractors.return_extractor import extract_return_invoice


SUPPORTED_DOCUMENT_TYPES = {
    "daily_invoice",
    "monthly_statement",
    "return_invoice",
    "mixed",
    "unknown",
}


def canonical_document_type(document_type: str) -> str:
    normalized_type = (document_type or "").strip().lower()

    if normalized_type == "mixed_document":
        return "mixed"

    if normalized_type not in SUPPORTED_DOCUMENT_TYPES:
        return "unknown"

    return normalized_type


def build_page_route_plan(
    classification_result: DocumentTypeResult,
) -> list[dict]:
    page_plan: list[dict] = []

    for page_result in classification_result.page_results:
        page_type = canonical_document_type(page_result.document_type)

        page_plan.append(
            {
                "page_number": page_result.page_number,
                "document_type": page_type,
                "confidence": page_result.confidence,
                "target_extractor": f"{page_type}_extractor",
                "evidence": page_result.evidence,
            }
        )

    return page_plan


def route_document_for_extraction(
    document_id: str,
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult,
) -> dict:
    document_type = canonical_document_type(classification_result.document_type)
    page_route_plan = build_page_route_plan(classification_result)

    if document_type == "daily_invoice":
        extraction_result = extract_daily_invoice(
            spatial_result=spatial_result,
            classification_result=classification_result,
        )
        return {
            "document_id": document_id,
            "document_type": document_type,
            "target_extractor": "daily_invoice_extractor",
            "status": extraction_result.get("status", "ready_for_extraction"),
            "review_required": False,
            "confidence": classification_result.confidence,
            "evidence": classification_result.evidence,
            "page_route_plan": page_route_plan,
            "extraction_result": extraction_result,
        }

    if document_type == "monthly_statement":
        extraction_result = extract_monthly_statement(
            spatial_result=spatial_result,
            classification_result=classification_result,
        )
        return {
            "document_id": document_id,
            "document_type": document_type,
            "target_extractor": "monthly_statement_extractor",
            "status": extraction_result.get("status", "ready_for_extraction"),
            "review_required": False,
            "confidence": classification_result.confidence,
            "evidence": classification_result.evidence,
            "page_route_plan": page_route_plan,
            "extraction_result": extraction_result,
        }

    if document_type == "return_invoice":
        extraction_result = extract_return_invoice(
            spatial_result=spatial_result,
            classification_result=classification_result,
        )
        return {
            "document_id": document_id,
            "document_type": document_type,
            "target_extractor": "return_extractor",
            "status": extraction_result.get("status", "ready_for_extraction"),
            "review_required": False,
            "confidence": classification_result.confidence,
            "evidence": classification_result.evidence,
            "page_route_plan": page_route_plan,
            "extraction_result": extraction_result,
        }

    if document_type == "mixed":
        extraction_result = extract_mixed_document(
            spatial_result=spatial_result,
            classification_result=classification_result,
        )
        mixed_details = (
            classification_result.mixed_details.model_dump()
            if classification_result.mixed_details is not None
            else None
        )

        return {
            "document_id": document_id,
            "document_type": document_type,
            "target_extractor": "mixed_extractor",
            "status": extraction_result.get("status", "split_by_page_then_extract"),
            "review_required": True,
            "confidence": classification_result.confidence,
            "evidence": classification_result.evidence,
            "mixed_details": mixed_details,
            "page_route_plan": page_route_plan,
            "extraction_result": extraction_result,
        }

    return {
        "document_id": document_id,
        "document_type": "unknown",
        "target_extractor": None,
        "status": "manual_review_required",
        "review_required": True,
        "confidence": classification_result.confidence,
        "evidence": classification_result.evidence,
        "page_route_plan": page_route_plan,
        "warnings": [
            "Document type could not be classified with high confidence."
        ],
    }
