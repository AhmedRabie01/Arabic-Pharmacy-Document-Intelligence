import json
from pathlib import Path

from app.core.config import PROCESSED_EXTRACTION_JSON_DIR
from app.schemas.document import DocumentTypeResult, SpatialMappingResponse
from app.services.extractors import route_document_for_extraction


def get_extraction_output_path(
    document_id: str,
    output_dir: Path | None = None,
) -> Path:
    target_dir = output_dir or PROCESSED_EXTRACTION_JSON_DIR
    return target_dir / f"{document_id}.json"


def save_extraction_result(
    document_id: str,
    extraction_result: dict,
    output_path: Path | None = None,
) -> str:
    target_path = output_path or get_extraction_output_path(document_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            extraction_result,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    return str(target_path)


def route_and_save_document_extraction(
    document_id: str,
    spatial_result: SpatialMappingResponse,
    classification_result: DocumentTypeResult,
) -> tuple[dict, str]:
    route_result = route_document_for_extraction(
        document_id=document_id,
        spatial_result=spatial_result,
        classification_result=classification_result,
    )
    output_path = save_extraction_result(
        document_id=document_id,
        extraction_result=route_result,
    )
    return route_result, output_path
