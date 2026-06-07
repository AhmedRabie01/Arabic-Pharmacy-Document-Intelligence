from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import ALL_INVOICES_DIR
from app.schemas.document import (
    DatabaseSaveResponse,
    ProcessDocumentResponse,
    RecentDocumentLedgerResponse,
    SupplierMonthlyLedgerResponse,
    UploadResponse,
)
from app.services.database_service import (
    get_supplier_monthly_ledger,
    list_recent_documents,
    persist_pipeline_result_by_document_id,
)
from app.services.pipeline_service import run_full_document_pipeline
from app.services.upload_service import save_uploaded_invoice


router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    return save_uploaded_invoice(file)


def _resolve_file_path(document_id: str, file_path: str | None) -> str:
    if file_path is not None:
        candidate_path = Path(file_path)
        if not candidate_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File does not exist: {file_path}",
            )
        return str(candidate_path)

    candidates = sorted(ALL_INVOICES_DIR.glob(f"{document_id}_*"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                "Could not auto-resolve file path from document_id. "
                "Pass file_path explicitly or upload the document first."
            ),
        )

    return str(candidates[-1])


@router.post("/process", response_model=ProcessDocumentResponse)
async def process_document(
    document_id: str,
    file_path: str | None = None,
    document_ai_provider: str = "auto",
    run_signature_detection: bool = True,
    save_to_database: bool = False,
    signature_conf_threshold: float | None = None,
):
    resolved_file_path = _resolve_file_path(
        document_id=document_id,
        file_path=file_path,
    )

    try:
        return run_full_document_pipeline(
            document_id=document_id,
            file_path=resolved_file_path,
            document_ai_provider=document_ai_provider,
            run_signature_detection=run_signature_detection,
            save_to_database=save_to_database,
            signature_conf_threshold=signature_conf_threshold,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/save-result", response_model=DatabaseSaveResponse)
async def save_processed_result(document_id: str):
    try:
        return persist_pipeline_result_by_document_id(document_id=document_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/analytics/suppliers/monthly-ledger", response_model=SupplierMonthlyLedgerResponse)
async def get_supplier_monthly_ledger_report(
    year: int,
    month: int,
    supplier_id: str | None = None,
):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")

    return get_supplier_monthly_ledger(
        year=year,
        month=month,
        supplier_id=supplier_id,
    )


@router.get("/analytics/documents/recent", response_model=RecentDocumentLedgerResponse)
async def get_recent_documents_report(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    return list_recent_documents(limit=limit)
