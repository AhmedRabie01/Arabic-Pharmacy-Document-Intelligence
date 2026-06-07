from pathlib import Path
import json

import fitz  # PyMuPDF
from PIL import Image

from app.core.config import (
    PROCESSED_IMAGES_DIR,
    PROCESSED_METADATA_DIR,
    PROCESSED_TEXT_DIR,
    PDF_RENDER_DPI,
    IMAGE_FORMAT,
)
from app.schemas.document import DocumentLoadResponse


PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_file_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in PDF_EXTENSIONS:
        return "pdf"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    raise ValueError(f"Unsupported file type: {extension}")


def create_document_image_dir(document_id: str) -> Path:
    output_dir = PROCESSED_IMAGES_DIR / document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def pdf_has_digital_text(pdf_path: Path) -> bool:
    with fitz.open(pdf_path) as document:
        for page in document:
            text = page.get_text("text").strip()
            if text:
                return True

    return False


def extract_pdf_text_by_page(pdf_path: Path) -> dict[int, str]:
    page_text_map: dict[int, str] = {}

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                page_text_map[page_index] = text

    return page_text_map


def render_pdf_to_images(pdf_path: Path, document_id: str) -> list[str]:
    output_dir = create_document_image_dir(document_id)
    image_paths: list[str] = []

    zoom = PDF_RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            image_path = output_dir / f"page_{page_index}.{IMAGE_FORMAT}"
            pixmap.save(image_path)

            image_paths.append(str(image_path))

    return image_paths


def prepare_image_file(image_path: Path, document_id: str) -> list[str]:
    output_dir = create_document_image_dir(document_id)
    output_path = output_dir / f"page_1.{IMAGE_FORMAT}"

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.save(output_path)

    return [str(output_path)]


def get_metadata_path(document_id: str) -> Path:
    return PROCESSED_METADATA_DIR / f"{document_id}.json"


def get_digital_text_path(document_id: str) -> Path:
    return PROCESSED_TEXT_DIR / f"{document_id}_digital_text.json"


def save_digital_text(document_id: str, digital_text_by_page: dict[int, str]) -> str:
    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = get_digital_text_path(document_id)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "document_id": document_id,
                "page_count": len(digital_text_by_page),
                "pages": digital_text_by_page,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def save_document_metadata(result: DocumentLoadResponse) -> Path:
    PROCESSED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = get_metadata_path(result.document_id)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.model_dump(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata_path


def load_document(document_id: str, file_path: str) -> DocumentLoadResponse:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = get_file_type(path)
    metadata_path = get_metadata_path(document_id)

    if file_type == "pdf":
        has_text = pdf_has_digital_text(path)
        digital_text_by_page = extract_pdf_text_by_page(path) if has_text else {}
        page_images = render_pdf_to_images(path, document_id)
        digital_text_json_path = (
            save_digital_text(document_id, digital_text_by_page)
            if digital_text_by_page
            else None
        )

        result = DocumentLoadResponse(
            document_id=document_id,
            original_file_path=str(path),
            file_type=file_type,
            page_count=len(page_images),
            has_digital_text=has_text,
            page_images=page_images,
            digital_text_by_page=digital_text_by_page,
            digital_text_json_path=digital_text_json_path,
            metadata_path=str(metadata_path),
            status="loaded",
        )

        save_document_metadata(result)
        return result

    if file_type == "image":
        page_images = prepare_image_file(path, document_id)

        result = DocumentLoadResponse(
            document_id=document_id,
            original_file_path=str(path),
            file_type=file_type,
            page_count=1,
            has_digital_text=False,
            page_images=page_images,
            digital_text_by_page={},
            digital_text_json_path=None,
            metadata_path=str(metadata_path),
            status="loaded",
        )

        save_document_metadata(result)
        return result

    raise ValueError(f"Unsupported file type: {file_type}")
