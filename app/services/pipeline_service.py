import json
from pathlib import Path
from time import perf_counter

from app.core.config import (
    PROCESSED_PIPELINE_JSON_DIR,
    PROCESSED_SIGNATURE_JSON_DIR,
)
from app.schemas.document import SignatureDetectionResult
from app.services.document_ai_service import run_document_ai
from app.services.document_classifier import (
    classify_and_save_document_type,
    save_document_classification_result,
)
from app.services.document_loader import load_document
from app.services.extraction_service import (
    route_and_save_document_extraction,
    save_extraction_result,
)
from app.services.image_preprocessor import preprocess_document_images
from app.services.openai_extraction_service import run_openai_direct_extraction
from app.services.signature_yolo_detector import detect_signature_with_yolo
from app.services.spatial_mapper import spatial_map_document_ai_result
from app.services.database_service import persist_pipeline_payload
from app.services.supplier_profile_service import infer_supplier_profile
from app.services.validation_service import validate_extraction_route_result


def _split_extracted_text_by_page(extracted_text: str | None) -> dict[int, str]:
    if not extracted_text:
        return {}

    page_map: dict[int, str] = {}
    current_page_number: int | None = None
    current_lines: list[str] = []

    for raw_line in extracted_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[PAGE ") and line.endswith("]"):
            if current_page_number is not None:
                page_map[current_page_number] = "\n".join(current_lines).strip()

            page_number_text = line.replace("[PAGE ", "").replace("]", "").strip()
            current_page_number = int(page_number_text) if page_number_text.isdigit() else None
            current_lines = []
            continue

        if current_page_number is not None:
            current_lines.append(raw_line)

    if current_page_number is not None:
        page_map[current_page_number] = "\n".join(current_lines).strip()

    if not page_map and extracted_text.strip():
        page_map[1] = extracted_text.strip()

    return page_map


def _build_full_text_from_page_map(page_map: dict[int, str]) -> str | None:
    if not page_map:
        return None

    full_text_parts: list[str] = []
    for page_number in sorted(page_map):
        page_text = page_map[page_number].strip()
        if not page_text:
            continue
        full_text_parts.append(f"[PAGE {page_number}]")
        full_text_parts.append(page_text)

    if not full_text_parts:
        return None

    return "\n\n".join(full_text_parts)


def _merge_page_text_sources(
    document_ai_text: str | None,
    digital_text_by_page: dict[int, str],
) -> tuple[str | None, dict[str, object]]:
    ocr_text_by_page = _split_extracted_text_by_page(document_ai_text)
    page_numbers = sorted(set(ocr_text_by_page) | set(digital_text_by_page))

    merged_page_text: dict[int, str] = {}
    digital_page_count = 0
    ocr_only_page_count = 0

    for page_number in page_numbers:
        digital_text = (digital_text_by_page.get(page_number) or "").strip()
        ocr_text = (ocr_text_by_page.get(page_number) or "").strip()

        if digital_text:
            digital_page_count += 1
            if ocr_text and ocr_text != digital_text:
                merged_page_text[page_number] = f"{digital_text}\n{ocr_text}"
            else:
                merged_page_text[page_number] = digital_text
            continue

        if ocr_text:
            ocr_only_page_count += 1
            merged_page_text[page_number] = ocr_text

    merged_text = _build_full_text_from_page_map(merged_page_text)
    metadata = {
        "used_digital_text": digital_page_count > 0,
        "digital_text_pages": digital_page_count,
        "ocr_only_pages": ocr_only_page_count,
    }
    return merged_text, metadata


def _count_route_rows(route_result: dict) -> int:
    document_type = str(route_result.get("document_type", "unknown"))
    extraction_result = route_result.get("extraction_result") or {}

    if document_type == "mixed":
        total_rows = 0
        for page_item in extraction_result.get("page_extractions") or []:
            page_extraction_result = page_item.get("extraction_result") or {}
            total_rows += int(page_extraction_result.get("row_count") or 0)
        return total_rows

    return int(extraction_result.get("row_count") or 0)


def _has_monthly_summary_without_rows(route_result: dict, classification_result: dict) -> bool:
    document_type = str(classification_result.get("document_type", "unknown"))
    extraction_result = route_result.get("extraction_result") or {}

    if document_type == "monthly_statement":
        footer_summary = extraction_result.get("footer_summary") or {}
        return not extraction_result.get("row_count") and bool(footer_summary)

    if document_type == "mixed":
        for page_item in extraction_result.get("page_extractions") or []:
            if str(page_item.get("page_document_type")) != "monthly_statement":
                continue
            page_extraction_result = page_item.get("extraction_result") or {}
            if page_extraction_result.get("row_count"):
                continue
            if page_extraction_result.get("footer_summary"):
                return True

    return False


def _build_review_decision(
    document_ai_provider: str,
    warnings: list[str],
    classification_result: dict,
    extraction_route_result: dict,
    validation_result: dict,
) -> dict:
    reasons: list[str] = []
    recommended_action = "auto_approve"
    document_type = str(classification_result.get("document_type", "unknown"))
    validation_status = str(validation_result.get("status", "validation_failed"))
    total_rows = _count_route_rows(extraction_route_result)
    has_summary_only_monthly = _has_monthly_summary_without_rows(
        route_result=extraction_route_result,
        classification_result=classification_result,
    )

    if document_type == "unknown":
        reasons.append("Document type is unknown.")
        recommended_action = "reprocess_or_manual_entry"

    if total_rows == 0 and not has_summary_only_monthly:
        reasons.append("No structured rows were extracted.")
        recommended_action = "manual_review"

    if validation_status == "validation_failed":
        reasons.append("Validation failed.")
        recommended_action = "manual_review"

    if document_type == "mixed":
        reasons.append("Mixed document requires page-level review.")
        recommended_action = "manual_review"

    review_warning_tokens = [
        "fallback triggered",
        "experimental provider",
        "recovered from grouped row text",
        "may be unreliable",
        "validation completed with issues",
    ]
    matching_warning_count = 0
    for warning in warnings:
        normalized_warning = warning.lower()
        if any(token in normalized_warning for token in review_warning_tokens):
            matching_warning_count += 1

    if matching_warning_count > 0:
        reasons.append(
            f"{matching_warning_count} pipeline warning(s) indicate low-confidence extraction."
        )
        recommended_action = "manual_review"

    if document_ai_provider in {"glm_ocr_ollama", "qwen3_vl_ollama"}:
        reasons.append("Experimental provider was used.")
        recommended_action = "manual_review"

    if document_type != "unknown" and total_rows == 0 and not has_summary_only_monthly:
        return {
            "status": "failed_extraction",
            "recommended_action": "reprocess_or_manual_entry",
            "reasons": reasons,
            "document_type": document_type,
            "row_count": total_rows,
            "validation_status": validation_status,
        }

    if reasons:
        return {
            "status": "needs_human_review",
            "recommended_action": recommended_action,
            "reasons": reasons,
            "document_type": document_type,
            "row_count": total_rows,
            "validation_status": validation_status,
        }

    return {
        "status": "approved_auto",
        "recommended_action": "auto_approve",
        "reasons": [],
        "document_type": document_type,
        "row_count": total_rows,
        "validation_status": validation_status,
    }


def _to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {}


def _extract_page_number_from_path(image_path: str) -> int:
    path = Path(image_path)
    stem = path.stem

    if stem.startswith("page_"):
        raw_number = stem.replace("page_", "", 1)
        if raw_number.isdigit():
            return int(raw_number)

    return 0


def get_pipeline_output_path(document_id: str) -> Path:
    return PROCESSED_PIPELINE_JSON_DIR / f"{document_id}.json"


def get_signature_output_path(document_id: str) -> Path:
    return PROCESSED_SIGNATURE_JSON_DIR / f"{document_id}.json"


def save_json_payload(payload: dict, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return str(output_path)


def detect_signatures_for_document_pages(
    document_id: str,
    image_paths: list[str],
    conf_threshold: float | None = None,
) -> tuple[dict, str]:
    page_results: list[dict] = []
    signature_pages: list[int] = []

    for image_path in image_paths:
        page_number = _extract_page_number_from_path(image_path)

        signature_result: SignatureDetectionResult = detect_signature_with_yolo(
            image_path=image_path,
            conf_threshold=conf_threshold,
        )
        signature_result_dict = _to_dict(signature_result)

        page_result = {
            "page_number": page_number,
            "image_path": image_path,
            **signature_result_dict,
        }
        page_results.append(page_result)

        if signature_result.signature_present:
            signature_pages.append(page_number)

    best_page_result = None
    if page_results:
        best_page_result = max(page_results, key=lambda item: item["confidence"])

    signature_payload = {
        "document_id": document_id,
        "page_count": len(page_results),
        "signature_pages": sorted(signature_pages),
        "best_page_result": best_page_result,
        "pages": page_results,
    }

    signature_json_path = save_json_payload(
        payload=signature_payload,
        output_path=get_signature_output_path(document_id),
    )

    return signature_payload, signature_json_path


def run_full_document_pipeline(
    document_id: str,
    file_path: str,
    document_ai_provider: str = "auto",
    run_signature_detection: bool = True,
    save_to_database: bool = True,
    signature_conf_threshold: float | None = None,
) -> dict:
    pipeline_started_at = perf_counter()
    warnings: list[str] = []
    pipeline_payload: dict = {
        "document_id": document_id,
        "status": "pipeline_running",
        "failed_step": None,
        "error": None,
        "steps": {
            "load": {"status": "pending", "result": None},
            "preprocess": {"status": "pending", "result": None},
            "document_ai": {"status": "pending", "result": None},
            "spatial": {"status": "pending", "result": None},
            "classification": {
                "status": "pending",
                "result": None,
                "json_path": None,
            },
            "extraction": {
                "status": "pending",
                "result": None,
                "json_path": None,
            },
            "signature": {
                "status": "pending",
                "enabled": run_signature_detection,
                "result": None,
                "json_path": None,
            },
            "validation": {
                "status": "pending",
                "result": None,
            },
        },
        "outputs": {
            "source_file_path": file_path,
            "database_persistence_requested": save_to_database,
            "metadata_json_path": None,
            "digital_text_json_path": None,
            "classification_text_source": None,
            "supplier_profile": None,
            "ocr_json_path": None,
            "openai_usage_summary": None,
            "spatial_json_path": None,
            "classification_json_path": None,
            "extraction_json_path": None,
            "signature_json_path": None,
            "validation_status": None,
            "pipeline_json_path": None,
            "timing": {
                "stages": {},
                "total_seconds": None,
            },
        },
        "review": {
            "status": "pending",
            "recommended_action": None,
            "reasons": [],
        },
        "warnings": warnings,
    }

    def _fail(step_name: str, error: Exception) -> dict:
        pipeline_payload["status"] = "pipeline_failed"
        pipeline_payload["failed_step"] = step_name
        pipeline_payload["error"] = str(error)
        pipeline_payload["review"] = {
            "status": "failed_extraction",
            "recommended_action": "reprocess_or_manual_entry",
            "reasons": [f"Step '{step_name}' failed."],
        }
        warnings.append(f"Step '{step_name}' failed: {error}")
        pipeline_payload["outputs"]["timing"]["total_seconds"] = round(
            perf_counter() - pipeline_started_at,
            3,
        )
        pipeline_payload["outputs"]["pipeline_json_path"] = str(
            get_pipeline_output_path(document_id)
        )
        pipeline_json_path = save_json_payload(
            payload=pipeline_payload,
            output_path=get_pipeline_output_path(document_id),
        )
        pipeline_payload["outputs"]["pipeline_json_path"] = pipeline_json_path
        if save_to_database:
            try:
                pipeline_payload["outputs"]["database_persistence"] = persist_pipeline_payload(
                    pipeline_payload=pipeline_payload,
                )
            except Exception as database_error:
                warnings.append(f"Database persistence failed: {database_error}")
        else:
            pipeline_payload["outputs"]["database_persistence"] = {
                "status": "skipped",
                "reason": "disabled_by_user",
            }
        return pipeline_payload

    # 1) Load document and render pages into processed/images.
    try:
        stage_started_at = perf_counter()
        load_result = load_document(
            document_id=document_id,
            file_path=file_path,
        )
        pipeline_payload["steps"]["load"]["status"] = "completed"
        pipeline_payload["steps"]["load"]["result"] = _to_dict(load_result)
        pipeline_payload["outputs"]["metadata_json_path"] = load_result.metadata_path
        pipeline_payload["outputs"]["digital_text_json_path"] = (
            load_result.digital_text_json_path
        )
        pipeline_payload["outputs"]["timing"]["stages"]["load"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("load", error)

    # 2) Preprocess page images for OCR.
    try:
        stage_started_at = perf_counter()
        preprocess_result = preprocess_document_images(
            document_id=document_id,
            page_images=load_result.page_images,
        )
        pipeline_payload["steps"]["preprocess"]["status"] = "completed"
        pipeline_payload["steps"]["preprocess"]["result"] = _to_dict(preprocess_result)
        pipeline_payload["outputs"]["timing"]["stages"]["preprocess"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("preprocess", error)

    if document_ai_provider.lower().strip() == "openai":
        try:
            stage_started_at = perf_counter()
            openai_result = run_openai_direct_extraction(
                document_id=document_id,
                image_paths=preprocess_result.preprocessed_images,
                source="preprocessed",
            )
            pipeline_payload["steps"]["document_ai"]["status"] = "completed"
            pipeline_payload["steps"]["document_ai"]["result"] = {
                "document_id": openai_result["document_id"],
                "provider": openai_result["provider"],
                "source": openai_result["source"],
                "page_count": openai_result["page_count"],
                "status": openai_result["status"],
                "output_json_path": openai_result["output_json_path"],
                "usage_summary": openai_result.get("usage_summary"),
                "warnings": openai_result["warnings"],
            }
            pipeline_payload["outputs"]["ocr_json_path"] = openai_result["output_json_path"]
            pipeline_payload["outputs"]["openai_usage_summary"] = openai_result.get(
                "usage_summary"
            )
            pipeline_payload["outputs"]["classification_text_source"] = "openai_direct_extraction"
            warnings.extend(openai_result["warnings"])
            pipeline_payload["outputs"]["timing"]["stages"]["document_ai"] = round(
                perf_counter() - stage_started_at,
                3,
            )
        except Exception as error:
            return _fail("document_ai", error)

        pipeline_payload["steps"]["spatial"]["status"] = "skipped"
        pipeline_payload["steps"]["spatial"]["result"] = {
            "status": "bypassed_for_openai_direct_extraction",
            "reason": "OpenAI provider returns structured extraction directly.",
        }
        pipeline_payload["outputs"]["timing"]["stages"]["spatial"] = 0.0

        classification_result = openai_result["classification_result"]
        extraction_route_result = openai_result["route_result"]
        signature_payload = None

        try:
            stage_started_at = perf_counter()
            classification_json_path = save_document_classification_result(
                document_id=document_id,
                classification_result=classification_result,
            )
            pipeline_payload["steps"]["classification"]["status"] = "completed"
            pipeline_payload["steps"]["classification"]["result"] = _to_dict(
                classification_result
            )
            pipeline_payload["steps"]["classification"]["json_path"] = classification_json_path
            pipeline_payload["outputs"]["classification_json_path"] = classification_json_path
            pipeline_payload["outputs"]["timing"]["stages"]["classification"] = round(
                perf_counter() - stage_started_at,
                3,
            )
        except Exception as error:
            return _fail("classification", error)

        pipeline_payload["outputs"]["supplier_profile"] = infer_supplier_profile(
            source_file_path=file_path,
            extracted_text=json.dumps(extraction_route_result, ensure_ascii=False),
        )

        try:
            stage_started_at = perf_counter()
            extraction_json_path = save_extraction_result(
                document_id=document_id,
                extraction_result=extraction_route_result,
            )
            pipeline_payload["steps"]["extraction"]["status"] = "completed"
            pipeline_payload["steps"]["extraction"]["result"] = extraction_route_result
            pipeline_payload["steps"]["extraction"]["json_path"] = extraction_json_path
            pipeline_payload["outputs"]["extraction_json_path"] = extraction_json_path
            pipeline_payload["outputs"]["timing"]["stages"]["extraction"] = round(
                perf_counter() - stage_started_at,
                3,
            )
        except Exception as error:
            return _fail("extraction", error)

        if run_signature_detection:
            try:
                stage_started_at = perf_counter()
                signature_payload = openai_result["signature_payload"]
                signature_json_path = save_json_payload(
                    payload=signature_payload,
                    output_path=get_signature_output_path(document_id),
                )
                pipeline_payload["steps"]["signature"]["status"] = "completed"
                pipeline_payload["steps"]["signature"]["result"] = signature_payload
                pipeline_payload["steps"]["signature"]["json_path"] = signature_json_path
                pipeline_payload["outputs"]["signature_json_path"] = signature_json_path
                pipeline_payload["outputs"]["timing"]["stages"]["signature"] = round(
                    perf_counter() - stage_started_at,
                    3,
                )
            except Exception as error:
                return _fail("signature", error)
        else:
            pipeline_payload["steps"]["signature"]["status"] = "skipped"
            warnings.append("Signature detection step was skipped.")
            pipeline_payload["outputs"]["timing"]["stages"]["signature"] = 0.0

        try:
            stage_started_at = perf_counter()
            validation_result = validate_extraction_route_result(
                route_result=extraction_route_result,
                signature_payload=signature_payload,
            )
            pipeline_payload["steps"]["validation"]["status"] = "completed"
            pipeline_payload["steps"]["validation"]["result"] = validation_result
            pipeline_payload["outputs"]["validation_status"] = validation_result.get("status")
            pipeline_payload["review"] = _build_review_decision(
                document_ai_provider="openai",
                warnings=warnings,
                classification_result=_to_dict(classification_result),
                extraction_route_result=extraction_route_result,
                validation_result=validation_result,
            )

            if validation_result.get("status") == "validation_failed":
                pipeline_payload["status"] = "pipeline_completed_with_validation_issues"
                warnings.append(
                    "Validation completed with issues. Review steps.validation.result.issues."
                )
            else:
                pipeline_payload["status"] = "pipeline_completed"
            pipeline_payload["outputs"]["timing"]["stages"]["validation"] = round(
                perf_counter() - stage_started_at,
                3,
            )
        except Exception as error:
            return _fail("validation", error)

        pipeline_payload["outputs"]["timing"]["total_seconds"] = round(
            perf_counter() - pipeline_started_at,
            3,
        )
        pipeline_payload["outputs"]["pipeline_json_path"] = str(
            get_pipeline_output_path(document_id)
        )
        pipeline_json_path = save_json_payload(
            payload=pipeline_payload,
            output_path=get_pipeline_output_path(document_id),
        )
        pipeline_payload["outputs"]["pipeline_json_path"] = pipeline_json_path
        if save_to_database:
            try:
                pipeline_payload["outputs"]["database_persistence"] = persist_pipeline_payload(
                    pipeline_payload=pipeline_payload,
                )
            except Exception as database_error:
                warnings.append(f"Database persistence failed: {database_error}")
        else:
            pipeline_payload["outputs"]["database_persistence"] = {
                "status": "skipped",
                "reason": "disabled_by_user",
            }

        return pipeline_payload

    # 3) Run OCR/document AI with configured provider strategy.
    try:
        stage_started_at = perf_counter()
        document_ai_result = run_document_ai(
            document_id=document_id,
            image_paths=preprocess_result.preprocessed_images,
            source="preprocessed",
            provider=document_ai_provider,
        )
        pipeline_payload["steps"]["document_ai"]["status"] = "completed"
        pipeline_payload["steps"]["document_ai"]["result"] = _to_dict(document_ai_result)
        pipeline_payload["outputs"]["ocr_json_path"] = document_ai_result.output_json_path
        warnings.extend(document_ai_result.warnings)
        pipeline_payload["outputs"]["timing"]["stages"]["document_ai"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("document_ai", error)

    classification_text, classification_text_metadata = _merge_page_text_sources(
        document_ai_text=document_ai_result.extracted_text,
        digital_text_by_page=load_result.digital_text_by_page,
    )
    if classification_text_metadata["used_digital_text"]:
        pipeline_payload["outputs"]["classification_text_source"] = "digital_text_plus_document_ai"
        warnings.append(
            "Digital PDF text layer was used to strengthen page classification."
        )
    else:
        pipeline_payload["outputs"]["classification_text_source"] = "document_ai_only"

    pipeline_payload["outputs"]["supplier_profile"] = infer_supplier_profile(
        source_file_path=file_path,
        extracted_text=classification_text,
    )

    # 4) Map OCR lines to detected table regions.
    try:
        stage_started_at = perf_counter()
        spatial_result = spatial_map_document_ai_result(document_ai_result)
        pipeline_payload["steps"]["spatial"]["status"] = "completed"
        pipeline_payload["steps"]["spatial"]["result"] = _to_dict(spatial_result)
        pipeline_payload["outputs"]["spatial_json_path"] = spatial_result.spatial_json_path
        warnings.extend(spatial_result.warnings)
        pipeline_payload["outputs"]["timing"]["stages"]["spatial"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("spatial", error)

    # 5) Classify document type and save classification JSON.
    try:
        stage_started_at = perf_counter()
        classification_result, classification_json_path = classify_and_save_document_type(
            document_id=document_id,
            spatial_result=spatial_result,
            extracted_text=classification_text,
        )
        pipeline_payload["steps"]["classification"]["status"] = "completed"
        pipeline_payload["steps"]["classification"]["result"] = _to_dict(
            classification_result
        )
        pipeline_payload["steps"]["classification"]["json_path"] = classification_json_path
        pipeline_payload["outputs"]["classification_json_path"] = classification_json_path
        pipeline_payload["outputs"]["timing"]["stages"]["classification"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("classification", error)

    # 6) Route to extractor and save extraction JSON.
    try:
        stage_started_at = perf_counter()
        extraction_route_result, extraction_json_path = route_and_save_document_extraction(
            document_id=document_id,
            spatial_result=spatial_result,
            classification_result=classification_result,
        )
        pipeline_payload["steps"]["extraction"]["status"] = "completed"
        pipeline_payload["steps"]["extraction"]["result"] = extraction_route_result
        pipeline_payload["steps"]["extraction"]["json_path"] = extraction_json_path
        pipeline_payload["outputs"]["extraction_json_path"] = extraction_json_path
        pipeline_payload["outputs"]["timing"]["stages"]["extraction"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("extraction", error)

    # 7) Signature detection runs on original rendered page images.
    signature_payload = None
    if run_signature_detection:
        try:
            stage_started_at = perf_counter()
            signature_payload, signature_json_path = detect_signatures_for_document_pages(
                document_id=document_id,
                image_paths=load_result.page_images,
                conf_threshold=signature_conf_threshold,
            )
            pipeline_payload["steps"]["signature"]["status"] = "completed"
            pipeline_payload["steps"]["signature"]["result"] = signature_payload
            pipeline_payload["steps"]["signature"]["json_path"] = signature_json_path
            pipeline_payload["outputs"]["signature_json_path"] = signature_json_path
            pipeline_payload["outputs"]["timing"]["stages"]["signature"] = round(
                perf_counter() - stage_started_at,
                3,
            )
        except Exception as error:
            return _fail("signature", error)
    else:
        pipeline_payload["steps"]["signature"]["status"] = "skipped"
        warnings.append("Signature detection step was skipped.")
        pipeline_payload["outputs"]["timing"]["stages"]["signature"] = 0.0

    # 8) Validate extracted results and signature requirement rules.
    try:
        stage_started_at = perf_counter()
        validation_result = validate_extraction_route_result(
            route_result=extraction_route_result,
            signature_payload=signature_payload,
        )
        pipeline_payload["steps"]["validation"]["status"] = "completed"
        pipeline_payload["steps"]["validation"]["result"] = validation_result
        pipeline_payload["outputs"]["validation_status"] = validation_result.get("status")
        pipeline_payload["review"] = _build_review_decision(
            document_ai_provider=document_ai_result.provider,
            warnings=warnings,
            classification_result=_to_dict(classification_result),
            extraction_route_result=extraction_route_result,
            validation_result=validation_result,
        )

        if validation_result.get("status") == "validation_failed":
            pipeline_payload["status"] = "pipeline_completed_with_validation_issues"
            warnings.append(
                "Validation completed with issues. Review steps.validation.result.issues."
            )
        else:
            pipeline_payload["status"] = "pipeline_completed"
        pipeline_payload["outputs"]["timing"]["stages"]["validation"] = round(
            perf_counter() - stage_started_at,
            3,
        )
    except Exception as error:
        return _fail("validation", error)

    pipeline_payload["outputs"]["timing"]["total_seconds"] = round(
        perf_counter() - pipeline_started_at,
        3,
    )
    pipeline_payload["outputs"]["pipeline_json_path"] = str(
        get_pipeline_output_path(document_id)
    )
    pipeline_json_path = save_json_payload(
        payload=pipeline_payload,
        output_path=get_pipeline_output_path(document_id),
    )
    pipeline_payload["outputs"]["pipeline_json_path"] = pipeline_json_path
    if save_to_database:
        try:
            pipeline_payload["outputs"]["database_persistence"] = persist_pipeline_payload(
                pipeline_payload=pipeline_payload,
            )
        except Exception as database_error:
            warnings.append(f"Database persistence failed: {database_error}")
    else:
        pipeline_payload["outputs"]["database_persistence"] = {
            "status": "skipped",
            "reason": "disabled_by_user",
        }

    return pipeline_payload
