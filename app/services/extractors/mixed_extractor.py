from app.schemas.document import DocumentTypeResult, SpatialMappingResponse
from app.services.extractors.daily_invoice_extractor import extract_daily_invoice
from app.services.extractors.monthly_statement_extractor import extract_monthly_statement
from app.services.extractors.return_extractor import extract_return_invoice


def _table_only_spatial_result(
    spatial_result: SpatialMappingResponse,
    table_index: int,
) -> SpatialMappingResponse:
    table = spatial_result.tables[table_index]

    return SpatialMappingResponse(
        document_id=spatial_result.document_id,
        page_count=1,
        tables=[table],
        spatial_json_path=spatial_result.spatial_json_path,
        status=spatial_result.status,
        warnings=spatial_result.warnings,
    )


def _run_extractor_by_type(
    page_spatial_result: SpatialMappingResponse,
    page_document_type: str,
    classification_result: DocumentTypeResult,
) -> dict:
    if page_document_type == "daily_invoice":
        return extract_daily_invoice(
            spatial_result=page_spatial_result,
            classification_result=classification_result,
        )

    if page_document_type == "monthly_statement":
        return extract_monthly_statement(
            spatial_result=page_spatial_result,
            classification_result=classification_result,
        )

    if page_document_type == "return_invoice":
        return extract_return_invoice(
            spatial_result=page_spatial_result,
            classification_result=classification_result,
        )

    return {
        "document_id": page_spatial_result.document_id,
        "document_type": "unknown",
        "status": "manual_review_required",
        "row_count": 0,
        "rows": [],
        "warnings": [
            f"No extractor available for page type '{page_document_type}'.",
        ],
    }


def extract_mixed_document(
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult,
) -> dict:
    page_type_map: dict[int, str] = {
        page_result.page_number: page_result.document_type
        for page_result in classification_result.page_results
    }

    page_extractions: list[dict] = []
    unknown_pages: list[int] = []

    for table_index, table in enumerate(spatial_result.tables):
        page_number = table.page_number
        page_document_type = page_type_map.get(page_number, "unknown")

        page_spatial_result = _table_only_spatial_result(
            spatial_result=spatial_result,
            table_index=table_index,
        )

        page_extraction_result = _run_extractor_by_type(
            page_spatial_result=page_spatial_result,
            page_document_type=page_document_type,
            classification_result=classification_result,
        )

        page_extractions.append(
            {
                "page_number": page_number,
                "page_document_type": page_document_type,
                "extraction_result": page_extraction_result,
            }
        )

        if page_document_type == "unknown":
            unknown_pages.append(page_number)

    warnings: list[str] = []

    if unknown_pages:
        warnings.append(
            "Some pages are unknown and need manual review: "
            + ", ".join(str(page_number) for page_number in sorted(unknown_pages))
        )

    if classification_result.mixed_details is None:
        warnings.append("Mixed classification details were not provided.")

    return {
        "document_id": spatial_result.document_id,
        "document_type": "mixed",
        "status": "split_by_page_then_extract",
        "review_required": True,
        "page_extractions": page_extractions,
        "type_counts": classification_result.type_counts,
        "mixed_details": (
            classification_result.mixed_details.model_dump()
            if classification_result.mixed_details is not None
            else None
        ),
        "warnings": warnings,
    }
