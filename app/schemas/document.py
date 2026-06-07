from datetime import datetime
from typing import Any
from pydantic import BaseModel,Field


class UploadResponse(BaseModel):
    document_id: str
    original_filename: str
    saved_filename: str
    saved_path: str
    file_extension: str
    file_size_bytes: int
    upload_time: datetime
    upload_metadata_path: str | None = None
    status: str
    next_step: str 


class DocumentLoadResponse(BaseModel):
    document_id: str
    original_file_path: str
    file_type: str
    page_count: int
    has_digital_text: bool
    page_images: list[str]
    digital_text_by_page: dict[int, str] = Field(default_factory=dict)
    digital_text_json_path: str | None = None
    metadata_path: str | None = None
    status: str

class ImagePreprocessResponse(BaseModel):
    document_id: str
    input_images: list[str]
    preprocessed_images: list[str]
    page_count: int
    status: str

class OCRLine(BaseModel):
    text: str
    confidence: float | None = None
    box: list[list[float]] | None = None


class OCRPageResult(BaseModel):
    page_number: int
    image_path: str
    lines: list[OCRLine]
    page_text: str
    average_confidence: float | None = None


class OCRDocumentResult(BaseModel):
    document_id: str
    engine: str
    source: str
    page_count: int
    pages: list[OCRPageResult]
    full_text: str
    average_confidence: float | None = None
    ocr_json_path: str | None = None
    ocr_text_path: str | None = None
    status: str

class DocumentAIResult(BaseModel):
    document_id: str
    provider: str
    source: str
    page_count: int
    raw_output: dict | list | str | None = None
    extracted_text: str | None = None
    structured_data: dict | None = None
    confidence: float | None = None
    output_json_path: str | None = None
    status: str
    warnings: list[str] = Field(default_factory=list)

class SpatialMappedLine(BaseModel):
    text: str
    confidence: float | None = None
    box: list[list[float]] | None = None
    normalized_x: float
    normalized_y: float

class SpatialMappedCell(BaseModel):
    column_index: int
    column_name: str
    lines: list[SpatialMappedLine] = Field(default_factory=list)
    cell_text: str = ""

class SpatialMappedRow(BaseModel):
    row_index: int
    normalized_y: float
    lines: list[SpatialMappedLine]
    row_text: str
    cells: list[SpatialMappedCell] = Field(default_factory=list)
    row_type: str = "unknown"

class SpatialColumnAnchor(BaseModel):
    column_index: int
    column_name: str
    normalized_x: float

class ExtractedTableRow(BaseModel):
    movement_date: str | None = None
    notice_number: str | None = None
    invoice_number: str | None = None
    account_name: str | None = None
    item_name: str | None = None
    quantity: str | None = None
    price: str | None = None
    discount: str | None = None
    total: str | None = None

class SpatialTableRegion(BaseModel):
    page_number: int
    table_bbox: list[float]
    lines: list[SpatialMappedLine]
    line_count: int
    grouped_rows: list[SpatialMappedRow] = Field(default_factory=list)
    row_count: int = 0
    column_anchors: list[SpatialColumnAnchor] = Field(default_factory=list)
    extracted_rows: list[ExtractedTableRow] = Field(default_factory=list)

class SpatialMappingResponse(BaseModel):
    document_id: str
    page_count: int
    tables: list[SpatialTableRegion]
    spatial_json_path: str | None = None
    status: str
    warnings: list[str] = Field(default_factory=list)

class PageDocumentTypeResult(BaseModel):
    page_number: int
    document_type: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)

class MixedClassificationDetails(BaseModel):
    detected_types: list[str] = Field(default_factory=list)
    recommended_flow: str | None = None
    notes: list[str] = Field(default_factory=list)

class DocumentTypeResult(BaseModel):
    document_type: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    type_counts: dict[str, int] = Field(default_factory=dict)
    page_results: list[PageDocumentTypeResult] = Field(default_factory=list)
    mixed_details: MixedClassificationDetails | None = None

class SignatureDetectionResult(BaseModel):
    signature_present: bool
    signer_name: str | None = None
    signature_bbox: list[float] | None = None
    nearby_text: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ProcessDocumentResponse(BaseModel):
    document_id: str
    status: str
    failed_step: str | None = None
    error: str | None = None
    steps: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DatabaseSaveResponse(BaseModel):
    document_id: str
    status: str
    database_path: str
    supplier_id: str | None = None
    document_type: str | None = None
    source_pipeline_json_path: str | None = None


class SupplierMonthlyLedgerRow(BaseModel):
    supplier_id: str
    supplier_name: str | None = None
    statement_document_count: int
    movement_row_count: int
    ordered_amount: float
    payment_amount: float
    net_movement: float
    closing_balance: float | None = None
    basis: str


class SupplierMonthlyLedgerResponse(BaseModel):
    year: int
    month: int
    month_key: str
    supplier_filter: str | None = None
    database_path: str
    supplier_summaries: list[SupplierMonthlyLedgerRow] = Field(default_factory=list)


class RecentDocumentLedgerItem(BaseModel):
    document_id: str
    supplier_id: str | None = None
    supplier_name: str | None = None
    original_filename: str | None = None
    saved_filename: str | None = None
    file_extension: str | None = None
    file_size_bytes: int | None = None
    upload_time: str | None = None
    document_type: str | None = None
    pipeline_status: str | None = None
    review_status: str | None = None
    review_action: str | None = None
    processed_at: str | None = None
    source_file_path: str | None = None


class RecentDocumentLedgerResponse(BaseModel):
    limit: int
    database_path: str
    documents: list[RecentDocumentLedgerItem] = Field(default_factory=list)

