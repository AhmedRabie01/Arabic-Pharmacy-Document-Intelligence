import json

from app.core.config import (
    DOCUMENT_AI_PROVIDER,
    ENABLE_FALLBACK,
    FALLBACK_CONFIDENCE_THRESHOLD,
    FALLBACK_PROVIDER,
    PRIMARY_PROVIDER,
)
from app.schemas.document import DocumentAIResult
from app.services.ocr_engines.chandra_engine import run_chandra_ocr_on_image
from app.services.ocr_engines.glm_ocr_engine import run_glm_ocr_on_image
from app.services.ocr_engines.openai_engine import run_openai_ocr_on_image
from app.services.ocr_engines.qwen3_vl_engine import run_qwen3_vl_on_image
from app.services.ocr_service import run_ocr_on_document


TABLE_KEYWORDS = [
    "Ø§Ø³Ù… Ø§Ù„ØµÙ†Ù",
    "Ø§Ù„ÙƒÙ…ÙŠØ©",
    "Ø§Ù„Ø³Ø¹Ø±",
    "Ø§Ù„Ø®ØµÙ…",
    "Ø§Ù„Ø§Ø¬Ù…Ø§Ù„Ù‰",
    "Ø§Ù„Ø§Ø¬Ù…Ø§Ù„ÙŠ",
    "ØªØ§Ø±ÙŠØ® Ø§Ù„Ø­Ø±ÙƒØ©",
    "Ø±Ù‚Ù… Ø§Ù„Ø§Ø´Ø¹Ø§Ø±",
    "ÙØ§ØªÙˆØ±Ø©",
]


def contains_table_keywords(text: str) -> bool:
    normalized = text.replace("Ø¥", "Ø§").replace("Ø£", "Ø§").replace("Ø¢", "Ø§")
    return any(keyword in normalized for keyword in TABLE_KEYWORDS)


def run_document_ai(
    document_id: str,
    image_paths: list[str],
    source: str = "preprocessed",
    provider: str | None = None,
) -> DocumentAIResult:
    selected_provider = (provider or DOCUMENT_AI_PROVIDER).lower().strip()

    if selected_provider == "auto":
        return run_auto_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "paddleocr":
        return run_paddle_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "chandra_ollama":
        return run_chandra_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "openai":
        return run_openai_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "glm_ocr_ollama":
        return run_glm_ocr_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "qwen3_vl_ollama":
        return run_qwen3_vl_provider(
            document_id=document_id,
            image_paths=image_paths,
            source=source,
        )

    if selected_provider == "gemini":
        return provider_not_implemented(
            document_id=document_id,
            provider=selected_provider,
            source=source,
            page_count=len(image_paths),
        )

    if selected_provider == "azure_document_intelligence":
        return provider_not_implemented(
            document_id=document_id,
            provider=selected_provider,
            source=source,
            page_count=len(image_paths),
        )

    if selected_provider == "google_document_ai":
        return provider_not_implemented(
            document_id=document_id,
            provider=selected_provider,
            source=source,
            page_count=len(image_paths),
        )

    raise ValueError(f"Unsupported provider: {selected_provider}")


def run_auto_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    primary_provider = PRIMARY_PROVIDER.lower().strip()
    fallback_provider = FALLBACK_PROVIDER.lower().strip()

    if primary_provider != "paddleocr":
        return provider_not_implemented(
            document_id=document_id,
            provider="auto",
            source=source,
            page_count=len(image_paths),
        )

    primary_result = run_paddle_provider(
        document_id=document_id,
        image_paths=image_paths,
        source=source,
    )

    should_fallback = should_run_fallback(primary_result)

    if not should_fallback:
        primary_result.provider = "auto:paddleocr"
        primary_result.warnings.append(
            "Fallback was not triggered because primary provider result passed quality checks."
        )
        return primary_result

    if not ENABLE_FALLBACK:
        primary_result.provider = "auto:paddleocr"
        primary_result.warnings.append(
            "Fallback was recommended but ENABLE_FALLBACK is false."
        )
        return primary_result

    if fallback_provider != "chandra_ollama":
        primary_result.provider = "auto:paddleocr"
        primary_result.warnings.append(
            f"Fallback provider {fallback_provider} is not implemented yet."
        )
        return primary_result

    fallback_pages, fallback_extracted_text = _collect_layout_pages(
        document_id=document_id,
        image_paths=image_paths,
        page_runner=run_chandra_ocr_on_image,
    )
    parsed_pages = parse_chandra_pages(fallback_pages)

    return DocumentAIResult(
        document_id=document_id,
        provider="auto:paddleocr+chandra_ollama",
        source=source,
        page_count=len(image_paths),
        raw_output={
            "primary": primary_result.raw_output,
            "fallback": fallback_pages,
        },
        extracted_text=primary_result.extracted_text,
        structured_data={
            "primary_provider": "paddleocr",
            "fallback_provider": "chandra_ollama",
            "fallback_structured_data": {
                "pages": parsed_pages,
            },
            "fallback_extracted_text": fallback_extracted_text,
        },
        confidence=primary_result.confidence,
        output_json_path=primary_result.output_json_path,
        status="completed_with_fallback",
        warnings=[
            f"Fallback triggered because PaddleOCR confidence was below {FALLBACK_CONFIDENCE_THRESHOLD} "
            "or quality checks failed.",
            *primary_result.warnings,
            "Chandra/Ollama output may not include confidence or line-level boxes.",
        ],
    )


def should_run_fallback(result: DocumentAIResult) -> bool:
    if result.confidence is None:
        return True

    if result.confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        return True

    if not result.extracted_text:
        return True

    if not contains_table_keywords(result.extracted_text):
        return True

    return False


def run_paddle_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    ocr_result = run_ocr_on_document(
        document_id=document_id,
        image_paths=image_paths,
        source=source,
    )

    return DocumentAIResult(
        document_id=document_id,
        provider="paddleocr",
        source=source,
        page_count=ocr_result.page_count,
        raw_output=ocr_result.model_dump(),
        extracted_text=ocr_result.full_text,
        structured_data=None,
        confidence=ocr_result.average_confidence,
        output_json_path=ocr_result.ocr_json_path,
        status="completed",
        warnings=[],
    )


def _collect_layout_pages(
    document_id: str,
    image_paths: list[str],
    page_runner,
) -> tuple[list[dict], str]:
    pages = []
    full_text_parts = []

    for page_number, image_path in enumerate(image_paths, start=1):
        page_result = page_runner(
            image_path=image_path,
            page_number=page_number,
            document_id=document_id,
        )

        page_dump = page_result.model_dump()
        pages.append(page_dump)
        full_text_parts.append(f"[PAGE {page_number}]\n{page_result.page_text}")

    return pages, "\n\n".join(full_text_parts)


def _build_layout_provider_result(
    *,
    document_id: str,
    image_paths: list[str],
    source: str,
    provider_name: str,
    page_runner,
    warning_message: str,
) -> DocumentAIResult:
    primary_result = run_paddle_provider(
        document_id=document_id,
        image_paths=image_paths,
        source=source,
    )
    layout_pages, layout_extracted_text = _collect_layout_pages(
        document_id=document_id,
        image_paths=image_paths,
        page_runner=page_runner,
    )
    parsed_pages = parse_chandra_pages(layout_pages)

    return DocumentAIResult(
        document_id=document_id,
        provider=provider_name,
        source=source,
        page_count=len(image_paths),
        raw_output={
            "primary": primary_result.raw_output,
            "fallback": layout_pages,
        },
        extracted_text=primary_result.extracted_text,
        structured_data={
            "primary_provider": "paddleocr",
            "fallback_provider": provider_name,
            "fallback_structured_data": {
                "pages": parsed_pages,
            },
            "fallback_extracted_text": layout_extracted_text,
        },
        confidence=primary_result.confidence,
        output_json_path=primary_result.output_json_path,
        status="completed",
        warnings=[warning_message],
    )


def run_chandra_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    return _build_layout_provider_result(
        document_id=document_id,
        image_paths=image_paths,
        source=source,
        provider_name="chandra_ollama",
        page_runner=run_chandra_ocr_on_image,
        warning_message="Chandra/Ollama output may not include confidence or line-level boxes.",
    )


def run_openai_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    return _build_layout_provider_result(
        document_id=document_id,
        image_paths=image_paths,
        source=source,
        provider_name="openai",
        page_runner=run_openai_ocr_on_image,
        warning_message="OpenAI layout output may not include confidence or line-level boxes.",
    )


def run_glm_ocr_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    pages = []
    full_text_parts = []

    for page_number, image_path in enumerate(image_paths, start=1):
        page_result = run_glm_ocr_on_image(
            image_path=image_path,
            page_number=page_number,
            document_id=document_id,
        )

        page_dump = page_result.model_dump()
        pages.append(page_dump)
        full_text_parts.append(f"[PAGE {page_number}]\n{page_result.page_text}")

    parsed_pages = parse_chandra_pages(pages)

    return DocumentAIResult(
        document_id=document_id,
        provider="glm_ocr_ollama",
        source=source,
        page_count=len(image_paths),
        raw_output=pages,
        extracted_text="\n\n".join(full_text_parts),
        structured_data={
            "pages": parsed_pages,
        },
        confidence=None,
        output_json_path=None,
        status="completed_experimental",
        warnings=[
            "GLM OCR experimental provider returns direct extraction-style JSON.",
            "This provider is intended for few-shot benchmarking and adapter experiments, not the current spatial-mapping path.",
        ],
    )


def run_qwen3_vl_provider(
    document_id: str,
    image_paths: list[str],
    source: str,
) -> DocumentAIResult:
    pages = []
    full_text_parts = []

    for page_number, image_path in enumerate(image_paths, start=1):
        page_result = run_qwen3_vl_on_image(
            image_path=image_path,
            page_number=page_number,
            document_id=document_id,
        )

        page_dump = page_result.model_dump()
        pages.append(page_dump)
        full_text_parts.append(f"[PAGE {page_number}]\n{page_result.page_text}")

    parsed_pages = parse_chandra_pages(pages)

    return DocumentAIResult(
        document_id=document_id,
        provider="qwen3_vl_ollama",
        source=source,
        page_count=len(image_paths),
        raw_output=pages,
        extracted_text="\n\n".join(full_text_parts),
        structured_data={
            "pages": parsed_pages,
        },
        confidence=None,
        output_json_path=None,
        status="completed_experimental",
        warnings=[
            "Qwen3-VL experimental provider returns direct extraction-style JSON.",
            "This provider is intended for few-shot benchmarking and adapter experiments, not the current spatial-mapping path.",
        ],
    )


def provider_not_implemented(
    document_id: str,
    provider: str,
    source: str,
    page_count: int,
) -> DocumentAIResult:
    return DocumentAIResult(
        document_id=document_id,
        provider=provider,
        source=source,
        page_count=page_count,
        raw_output=None,
        extracted_text=None,
        structured_data=None,
        confidence=None,
        output_json_path=None,
        status="not_implemented",
        warnings=[
            f"{provider} provider is configured but not implemented yet."
        ],
    )


def extract_json_from_text(text: str):
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("[PAGE"):
        parts = cleaned.split("\n", 1)
        if len(parts) == 2:
            cleaned = parts[1].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    first_array = cleaned.find("[")
    last_array = cleaned.rfind("]")

    if first_array != -1 and last_array != -1 and last_array > first_array:
        possible_json = cleaned[first_array:last_array + 1]
        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            pass

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")

    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        possible_json = cleaned[first_obj:last_obj + 1]
        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            pass

    return None


def parse_chandra_pages(raw_pages: list[dict]) -> list[dict]:
    parsed_pages = []

    for page in raw_pages:
        page_number = page.get("page_number")
        page_text = page.get("page_text", "")

        parsed_content = extract_json_from_text(page_text)
        parsed_successfully = parsed_content is not None

        parsed_pages.append(
            {
                "page_number": page_number,
                "image_path": page.get("image_path"),
                "parsed_content": parsed_content,
                "raw_page_text": page_text,
                "parsed_successfully": parsed_successfully,
                "parse_warning": None if parsed_successfully else "Chandra output was not valid complete JSON.",
            }
        )

    return parsed_pages
