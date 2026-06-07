from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
import shutil

from fastapi import HTTPException, UploadFile

from app.core.config import ALL_INVOICES_DIR, ALLOWED_EXTENSIONS, PROCESSED_METADATA_DIR
from app.schemas.document import UploadResponse


def validate_file_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}"
        )

    return extension


def make_safe_filename(filename: str) -> str:
    return filename.replace(" ", "_")


def get_upload_metadata_path(document_id: str) -> Path:
    return PROCESSED_METADATA_DIR / f"{document_id}_upload.json"


def save_upload_metadata(result: UploadResponse) -> str:
    PROCESSED_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = get_upload_metadata_path(result.document_id)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.model_dump(mode="json"),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def save_uploaded_invoice(file: UploadFile) -> UploadResponse:
    extension = validate_file_extension(file.filename)

    document_id = f"doc_{uuid4().hex[:12]}"
    safe_filename = make_safe_filename(file.filename)
    saved_filename = f"{document_id}_{safe_filename}"

    saved_path = ALL_INVOICES_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_bytes = saved_path.stat().st_size

    result = UploadResponse(
        document_id=document_id,
        original_filename=file.filename,
        saved_filename=saved_filename,
        saved_path=str(saved_path),
        file_extension=extension,
        file_size_bytes=file_size_bytes,
        upload_time=datetime.utcnow(),
        upload_metadata_path=None,
        status="uploaded",
        next_step="document_loader"
    )
    result.upload_metadata_path = save_upload_metadata(result)
    return result
