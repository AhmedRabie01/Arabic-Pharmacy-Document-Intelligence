import json
from pathlib import Path

from PIL import Image

from app.core.config import PROCESSED_SPATIAL_JSON_DIR
from app.schemas.document import (
    DocumentAIResult,
    SpatialMappedLine,
    SpatialMappingResponse,
    SpatialTableRegion,
    SpatialMappedRow,
    SpatialMappedCell,
    SpatialColumnAnchor,
    ExtractedTableRow,
)


NORMALIZED_SCALE = 1000.0
EXPECTED_DAILY_INVOICE_FIELDS = [
    "movement_date",
    "notice_number",
    "invoice_number",
    "account_name",
    "item_name",
    "quantity",
    "price",
    "discount",
    "total",
]


def parse_chandra_bbox(bbox: str) -> list[float]:
    values = [float(value) for value in bbox.split()]

    if len(values) != 4:
        raise ValueError(f"Invalid Chandra bbox format: {bbox}")

    x1, y1, x2, y2 = values

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


def get_image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.width, image.height


def get_box_center(box: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)

    return center_x, center_y


def normalize_point(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    normalized_x = (x / image_width) * NORMALIZED_SCALE
    normalized_y = (y / image_height) * NORMALIZED_SCALE

    return normalized_x, normalized_y


def is_point_inside_bbox(
    x: float,
    y: float,
    bbox: list[float],
) -> bool:
    x1, y1, x2, y2 = bbox

    return x1 <= x <= x2 and y1 <= y <= y2


def describe_chandra_parsed_content(parsed_content: object) -> str:
    if isinstance(parsed_content, dict):
        keys = sorted(str(key) for key in parsed_content.keys())
        return f"dict[{', '.join(keys)}]"

    if isinstance(parsed_content, list):
        if not parsed_content:
            return "list[empty]"

        first_item = parsed_content[0]
        if isinstance(first_item, dict):
            keys = sorted(str(key) for key in first_item.keys())
            return f"list[dict[{', '.join(keys)}]]"

        return f"list[{type(first_item).__name__}]"

    return type(parsed_content).__name__


def extract_chandra_table_bboxes(
    parsed_content: object,
) -> tuple[list[list[float]], str | None]:
    table_bboxes: list[list[float]] = []

    if isinstance(parsed_content, dict):
        return (
            [],
            "Chandra parsed content uses an object schema, but spatial mapping "
            "currently expects a layout-item list with label/bbox fields. "
            f"Detected schema: {describe_chandra_parsed_content(parsed_content)}.",
        )

    if not isinstance(parsed_content, list):
        return (
            [],
            "Chandra parsed content is not a list. "
            f"Detected schema: {describe_chandra_parsed_content(parsed_content)}.",
        )

    for item in parsed_content:
        if not isinstance(item, dict):
            return (
                [],
                "Chandra parsed content contains non-dict items, so table bboxes "
                "cannot be extracted safely. "
                f"Detected schema: {describe_chandra_parsed_content(parsed_content)}.",
            )

        label = str(item.get("label", "")).lower()

        if label != "table":
            continue

        bbox = item.get("bbox")

        if not bbox:
            continue

        table_bboxes.append(parse_chandra_bbox(bbox))

    if table_bboxes:
        return table_bboxes, None

    return (
        [],
        "No table bbox items were found in Chandra parsed content. "
        f"Detected schema: {describe_chandra_parsed_content(parsed_content)}.",
    )


def map_ocr_lines_to_table(
    ocr_lines: list[dict],
    table_bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[SpatialMappedLine]:
    mapped_lines: list[SpatialMappedLine] = []

    for line in ocr_lines:
        text = str(line.get("text", "")).strip()
        box = line.get("box")

        if not text:
            continue

        if not box:
            continue

        center_x, center_y = get_box_center(box)

        normalized_x, normalized_y = normalize_point(
            x=center_x,
            y=center_y,
            image_width=image_width,
            image_height=image_height,
        )

        if not is_point_inside_bbox(
            x=normalized_x,
            y=normalized_y,
            bbox=table_bbox,
        ):
            continue

        mapped_lines.append(
            SpatialMappedLine(
                text=text,
                confidence=line.get("confidence"),
                box=box,
                normalized_x=round(normalized_x, 2),
                normalized_y=round(normalized_y, 2),
            )
        )

    return mapped_lines


def find_chandra_page_by_number(
    chandra_pages: list[dict],
    page_number: int,
) -> dict | None:
    for page in chandra_pages:
        if page.get("page_number") == page_number:
            return page

    return None
def group_lines_into_rows(
    mapped_lines: list[SpatialMappedLine],
    y_tolerance: float = 12.0,
    sort_right_to_left: bool = True,
) -> list[SpatialMappedRow]:
    if not mapped_lines:
        return []

    sorted_lines = sorted(
        mapped_lines,
        key=lambda line: line.normalized_y,
    )

    row_groups: list[list[SpatialMappedLine]] = []

    for line in sorted_lines:
        placed = False

        for row in row_groups:
            row_y_values = [item.normalized_y for item in row]
            row_average_y = sum(row_y_values) / len(row_y_values)

            if abs(line.normalized_y - row_average_y) <= y_tolerance:
                row.append(line)
                placed = True
                break

        if not placed:
            row_groups.append([line])

    grouped_rows: list[SpatialMappedRow] = []

    for row_index, row_lines in enumerate(row_groups):
        row_lines_sorted_by_x = sorted(
            row_lines,
            key=lambda line: line.normalized_x,
            reverse=sort_right_to_left,
        )

        row_y_values = [
            line.normalized_y for line in row_lines_sorted_by_x
        ]

        average_row_y = sum(row_y_values) / len(row_y_values)

        row_text = " | ".join(
            line.text for line in row_lines_sorted_by_x
        )

        grouped_rows.append(
            SpatialMappedRow(
                row_index=row_index,
                normalized_y=average_row_y,
                lines=row_lines_sorted_by_x,
                row_text=row_text,
            )
        )

    return grouped_rows

def build_column_anchors_from_header_row(
    grouped_rows: list[SpatialMappedRow],
) -> list[SpatialColumnAnchor]:
    """
    Build column anchors from the first grouped row.

    We assume row 0 is the table header row.
    Each header line becomes a column anchor.
    """

    if not grouped_rows:
        return []

    header_row = grouped_rows[0]

    anchors: list[SpatialColumnAnchor] = []

    for column_index, line in enumerate(header_row.lines):
        anchors.append(
            SpatialColumnAnchor(
                column_index=column_index,
                column_name=line.text,
                normalized_x=line.normalized_x,
            )
        )

    return anchors

def find_nearest_column_anchor(
    line: SpatialMappedLine,
    column_anchors: list[SpatialColumnAnchor],
) -> SpatialColumnAnchor | None:
    """
    Find the nearest column anchor for a line using normalized_x.
    """

    if not column_anchors:
        return None

    nearest_anchor = min(
        column_anchors,
        key=lambda anchor: abs(line.normalized_x - anchor.normalized_x),
    )

    return nearest_anchor

def assign_row_lines_to_columns(
    row: SpatialMappedRow,
    column_anchors: list[SpatialColumnAnchor],
) -> list[SpatialMappedCell]:
    """
    Assign every line in a row to the nearest column anchor.
    Multiple lines can belong to the same column.
    """

    if not column_anchors:
        return []

    cells_by_column_index: dict[int, SpatialMappedCell] = {}

    for anchor in column_anchors:
        cells_by_column_index[anchor.column_index] = SpatialMappedCell(
            column_index=anchor.column_index,
            column_name=anchor.column_name,
            lines=[],
            cell_text="",
        )

    for line in row.lines:
        nearest_anchor = find_nearest_column_anchor(
            line=line,
            column_anchors=column_anchors,
        )

        if nearest_anchor is None:
            continue

        cell = cells_by_column_index[nearest_anchor.column_index]
        cell.lines.append(line)

    cells: list[SpatialMappedCell] = []

    for anchor in column_anchors:
        cell = cells_by_column_index[anchor.column_index]

        sorted_cell_lines = sorted(
            cell.lines,
            key=lambda line: line.normalized_x,
            reverse=True,
        )

        cell.lines = sorted_cell_lines
        cell.cell_text = " ".join(
            line.text for line in sorted_cell_lines
        ).strip()

        cells.append(cell)

    return cells

def assign_table_rows_to_columns(
    grouped_rows: list[SpatialMappedRow],
    column_anchors: list[SpatialColumnAnchor],
) -> list[SpatialMappedRow]:
    """
    Assign cells to every grouped row using column anchors.
    """

    rows_with_cells: list[SpatialMappedRow] = []

    for row in grouped_rows:
        row.cells = assign_row_lines_to_columns(
            row=row,
            column_anchors=column_anchors,
        )

        rows_with_cells.append(row)

    return rows_with_cells

def detect_row_type(row: SpatialMappedRow) -> str:
    """
    Detect whether a row is header, data, summary, or unknown.
    """

    row_text = row.row_text.strip()

    if not row_text:
        return "empty"

    if row.row_index == 0:
        return "header"

    normalized_text = row_text.replace(" ", "")

    if "الاجمالى" in normalized_text or "الإجمالى" in normalized_text:
        has_date = "/" in normalized_text
        has_quantity = "1.000" in normalized_text or "2.000" in normalized_text

        if not has_date and not has_quantity:
            return "summary"

    has_date_like_text = "/" in row_text
    has_numeric_cells = sum(
        1 for cell in row.cells if any(char.isdigit() for char in cell.cell_text)
    )

    if has_date_like_text and has_numeric_cells >= 3:
        return "data"

    return "unknown"
def detect_table_row_types(
    
    grouped_rows: list[SpatialMappedRow],
) -> list[SpatialMappedRow]:
    """
    Add row_type to every grouped row.
    """

    for row in grouped_rows:
        row.row_type = detect_row_type(row)

    return grouped_rows

def build_column_field_map_by_order(
    column_anchors: list[SpatialColumnAnchor],
) -> dict[int, str]:
    """
    Map detected column indexes to stable internal field names using expected order.

    This is more robust than relying on noisy Arabic OCR header text.
    """

    column_field_map: dict[int, str] = {}

    for index, anchor in enumerate(column_anchors):
        if index >= len(EXPECTED_DAILY_INVOICE_FIELDS):
            continue

        column_field_map[anchor.column_index] = EXPECTED_DAILY_INVOICE_FIELDS[index]

    return column_field_map

def _legacy_normalize_column_name(column_name: str) -> str | None:
    """
    Legacy helper kept only for reference.
    Task 6E uses column order mapping instead of OCR header text.
    """

    cleaned = column_name.replace(" ", "").strip()

    if "تاريخ" in cleaned or "ناررح" in cleaned or "الحركة" in cleaned:
        return "movement_date"

    if "اشعار" in cleaned or "الإشعار" in cleaned or "رفم" in cleaned:
        return "notice_number"

    if "فاتورة" in cleaned:
        return "invoice_number"

    if "الحساب" in cleaned:
        return "account_name"

    if "الصنف" in cleaned:
        return "item_name"

    if "الكمية" in cleaned:
        return "quantity"

    if "السعر" in cleaned:
        return "price"

    if "الخصم" in cleaned:
        return "discount"

    if "الاجمالى" in cleaned or "الإجمالى" in cleaned or "الاجمالي" in cleaned:
        return "total"

    return None

def extract_structured_row_from_mapped_row(
    row: SpatialMappedRow,
    column_field_map: dict[int, str],
) -> ExtractedTableRow:
    """
    Convert one mapped data row into a structured extracted row.
    """

    extracted_data: dict[str, str] = {}

    for cell in row.cells:
        # Map by detected column position, not by noisy OCR header text.
        field_name = column_field_map.get(cell.column_index)

        if field_name is None:
            continue

        extracted_data[field_name] = cell.cell_text.strip()

    return ExtractedTableRow(**extracted_data)

def extract_structured_rows_from_table(
    grouped_rows: list[SpatialMappedRow],
    column_field_map: dict[int, str],
) -> list[ExtractedTableRow]:
    """
    Extract only data rows into structured table rows.
    Header and summary rows are ignored.
    """

    extracted_rows: list[ExtractedTableRow] = []

    for row in grouped_rows:
        if row.row_type != "data":
            continue

        extracted_row = extract_structured_row_from_mapped_row(
            row=row,
            column_field_map=column_field_map,
        )
        extracted_rows.append(extracted_row)

    return extracted_rows

def get_spatial_output_path(document_id: str) -> Path:
    return PROCESSED_SPATIAL_JSON_DIR / f"{document_id}.json"

def save_spatial_mapping_output(result: SpatialMappingResponse) -> None:
    if not result.spatial_json_path:
        raise ValueError("spatial_json_path is missing.")

    output_path = Path(result.spatial_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "model_dump"):
        result_data = result.model_dump()
    else:
        result_data = result.dict()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

def spatial_map_document_ai_result(
    result: DocumentAIResult,
) -> SpatialMappingResponse:
    result_data = result.model_dump()

    document_id = result_data["document_id"]
    raw_output = result_data.get("raw_output") or {}
    structured_data = result_data.get("structured_data") or {}

    primary_output = raw_output.get("primary") or {}
    paddle_pages = primary_output.get("pages") or []

    fallback_structured_data = structured_data.get("fallback_structured_data") or {}
    chandra_pages = fallback_structured_data.get("pages") or []

    tables: list[SpatialTableRegion] = []
    warnings: list[str] = []

    for paddle_page in paddle_pages:
        page_number = paddle_page.get("page_number")
        image_path = paddle_page.get("image_path")
        ocr_lines = paddle_page.get("lines") or []

        if not image_path:
            warnings.append(f"Page {page_number}: missing image_path.")
            continue

        if not Path(image_path).exists():
            warnings.append(f"Page {page_number}: image file does not exist.")
            continue

        chandra_page = find_chandra_page_by_number(
            chandra_pages=chandra_pages,
            page_number=page_number,
        )

        if not chandra_page:
            warnings.append(f"Page {page_number}: no Chandra page found.")
            continue

        parsed_content = chandra_page.get("parsed_content")

        if not parsed_content:
            warnings.append(f"Page {page_number}: no parsed Chandra content.")
            continue

        table_bboxes, table_bbox_warning = extract_chandra_table_bboxes(parsed_content)

        if not table_bboxes:
            if table_bbox_warning:
                warnings.append(f"Page {page_number}: {table_bbox_warning}")
            else:
                warnings.append(f"Page {page_number}: no table bbox found.")
            continue

        image_width, image_height = get_image_size(image_path)

        for table_bbox in table_bboxes:
            mapped_lines = map_ocr_lines_to_table(
                ocr_lines=ocr_lines,
                table_bbox=table_bbox,
                image_width=image_width,
                image_height=image_height,
            )

            grouped_rows = group_lines_into_rows(
                mapped_lines=mapped_lines,
                y_tolerance=12.0,
                sort_right_to_left=True,
            )
            
            column_anchors = build_column_anchors_from_header_row(
                grouped_rows=grouped_rows,
            )

            column_field_map = build_column_field_map_by_order(
                column_anchors=column_anchors,
            )
            
            grouped_rows = assign_table_rows_to_columns(
                grouped_rows=grouped_rows,
                column_anchors=column_anchors,
            )

            grouped_rows = detect_table_row_types(
                grouped_rows=grouped_rows,
            )

            extracted_rows = extract_structured_rows_from_table(
                grouped_rows=grouped_rows,
                column_field_map=column_field_map,
            )
            
            tables.append(
                SpatialTableRegion(
                    page_number=page_number,
                    table_bbox=table_bbox,
                    lines=mapped_lines,
                    line_count=len(mapped_lines),
                    grouped_rows=grouped_rows,
                    row_count=len(grouped_rows),
                    column_anchors=column_anchors,
                    extracted_rows=extracted_rows,
                )
            )

    spatial_json_path = get_spatial_output_path(document_id)

    spatial_result = SpatialMappingResponse(
        document_id=document_id,
        page_count=len(paddle_pages),
        tables=tables,
        spatial_json_path=str(spatial_json_path),
        status="spatial_mapping_completed",
        warnings=warnings,
    )

    save_spatial_mapping_output(spatial_result)

    return spatial_result
