from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3

from app.core.config import BASE_DIR, DATABASE_URL, PROCESSED_PIPELINE_JSON_DIR
from app.services.supplier_profile_service import infer_supplier_profile
from app.services.upload_service import get_upload_metadata_path


def _get_sqlite_path() -> Path:
    if not DATABASE_URL.startswith("sqlite:///"):
        raise ValueError("Only sqlite DATABASE_URL values are supported.")

    raw_path = DATABASE_URL.removeprefix("sqlite:///")
    if raw_path.startswith("./"):
        return BASE_DIR / raw_path[2:]
    if raw_path.startswith("/"):
        return Path(raw_path)
    return BASE_DIR / raw_path


DB_PATH = _get_sqlite_path()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, str],
) -> None:
    existing_rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_columns = {row["name"] for row in existing_rows}

    for column_name, column_sql in expected_columns.items():
        if column_name in existing_columns:
            continue
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def init_database() -> None:
    with _get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_file_path TEXT,
                original_filename TEXT,
                saved_filename TEXT,
                file_extension TEXT,
                file_size_bytes INTEGER,
                upload_time TEXT,
                upload_metadata_path TEXT,
                supplier_id TEXT,
                supplier_name TEXT,
                supplier_confidence REAL,
                pipeline_status TEXT,
                review_status TEXT,
                review_action TEXT,
                document_type TEXT,
                processed_at TEXT,
                pipeline_json_path TEXT,
                extraction_json_path TEXT,
                classification_json_path TEXT
            );

            CREATE TABLE IF NOT EXISTS document_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                document_type TEXT,
                extraction_status TEXT,
                row_count INTEGER DEFAULT 0,
                header_json TEXT,
                footer_summary_json TEXT,
                UNIQUE(document_id, page_number),
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS monthly_statement_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                page_number INTEGER,
                supplier_id TEXT,
                movement_date TEXT,
                movement_month TEXT,
                reference_number TEXT,
                note TEXT,
                debit REAL,
                credit REAL,
                balance REAL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS daily_invoice_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                page_number INTEGER,
                supplier_id TEXT,
                invoice_date TEXT,
                invoice_month TEXT,
                invoice_number TEXT,
                customer_name TEXT,
                item_name TEXT,
                quantity_text TEXT,
                price REAL,
                discount_text TEXT,
                total REAL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """
        )
        _ensure_table_columns(
            connection,
            "documents",
            {
                "original_filename": "TEXT",
                "saved_filename": "TEXT",
                "file_extension": "TEXT",
                "file_size_bytes": "INTEGER",
                "upload_time": "TEXT",
                "upload_metadata_path": "TEXT",
            },
        )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = (
        text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
        .replace(",", "")
        .replace("٫", ".")
        .replace(" ", "")
    )
    try:
        return float(normalized)
    except ValueError:
        return None


def _normalize_iso_date(value: object) -> str | None:
    if value is None:
        return None

    text = (
        str(value)
        .translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
        .strip()
        .replace("/", "-")
    )
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def _month_key(date_text: str | None) -> str | None:
    if not date_text:
        return None
    return date_text[:7] if len(date_text) >= 7 else None


def _json_or_none(value: object) -> str | None:
    if not value:
        return None
    return json.dumps(value, ensure_ascii=False)


def _load_upload_metadata(document_id: str) -> dict:
    metadata_path = get_upload_metadata_path(document_id)
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _page_number_from_rows(rows: list[dict], fallback_page_number: int | None) -> int:
    for row in rows:
        row_page_number = row.get("page_number")
        if isinstance(row_page_number, int):
            return row_page_number
        if isinstance(row_page_number, str) and row_page_number.isdigit():
            return int(row_page_number)

    return fallback_page_number or 1


def _iter_page_payloads(route_result: dict) -> list[dict]:
    document_type = str(route_result.get("document_type", "unknown"))
    extraction_result = route_result.get("extraction_result") or {}
    page_route_plan = {
        int(item.get("page_number", 0)): item
        for item in route_result.get("page_route_plan") or []
    }

    if document_type == "mixed":
        payloads: list[dict] = []
        for page_item in extraction_result.get("page_extractions") or []:
            page_number = int(page_item.get("page_number", 0))
            page_extraction_result = page_item.get("extraction_result") or {}
            payloads.append(
                {
                    "page_number": page_number,
                    "document_type": page_item.get("page_document_type"),
                    "plan": page_route_plan.get(page_number),
                    "extraction_result": page_extraction_result,
                }
            )
        return payloads

    extraction_rows = extraction_result.get("rows") or []
    page_headers = extraction_result.get("page_headers") or {}
    page_footer_summaries = extraction_result.get("page_footer_summaries") or {}
    page_row_counts = extraction_result.get("page_row_counts") or {}

    page_numbers: set[int] = set()
    for key in page_headers.keys():
        if str(key).isdigit():
            page_numbers.add(int(key))
    for key in page_footer_summaries.keys():
        if str(key).isdigit():
            page_numbers.add(int(key))
    for key in page_row_counts.keys():
        if str(key).isdigit():
            page_numbers.add(int(key))
        elif isinstance(key, int):
            page_numbers.add(key)
    for row in extraction_rows:
        row_page_number = row.get("page_number")
        if isinstance(row_page_number, int):
            page_numbers.add(row_page_number)
        elif isinstance(row_page_number, str) and row_page_number.isdigit():
            page_numbers.add(int(row_page_number))

    if not page_numbers:
        page_number = _page_number_from_rows(extraction_rows, fallback_page_number=1)
        return [
            {
                "page_number": page_number,
                "document_type": document_type,
                "plan": page_route_plan.get(page_number),
                "extraction_result": extraction_result,
            }
        ]

    payloads: list[dict] = []
    for page_number in sorted(page_numbers):
        page_rows = [
            row
            for row in extraction_rows
            if str(row.get("page_number")) == str(page_number)
        ]
        page_header = page_headers.get(str(page_number)) or page_headers.get(page_number) or {}
        page_footer_summary = (
            page_footer_summaries.get(str(page_number))
            or page_footer_summaries.get(page_number)
            or {}
        )
        payloads.append(
            {
                "page_number": page_number,
                "document_type": document_type,
                "plan": page_route_plan.get(page_number),
                "extraction_result": {
                    **extraction_result,
                    "header": page_header or extraction_result.get("header") or {},
                    "footer_summary": page_footer_summary
                    or extraction_result.get("footer_summary")
                    or {},
                    "rows": page_rows,
                    "row_count": len(page_rows),
                    "page_row_counts": {page_number: len(page_rows)},
                },
            }
        )

    return payloads


def _extract_daily_header_and_footer(extraction_result: dict) -> tuple[dict, dict]:
    header = extraction_result.get("header") or {}
    footer_summary = extraction_result.get("footer_summary") or {}
    return header, footer_summary


def _extract_monthly_header_and_footer(extraction_result: dict) -> tuple[dict, dict]:
    header = extraction_result.get("header") or {}
    footer_summary = extraction_result.get("footer_summary") or {}
    return header, footer_summary


def persist_pipeline_payload(pipeline_payload: dict) -> dict:
    init_database()

    document_id = str(pipeline_payload.get("document_id", "")).strip()
    if not document_id:
        raise ValueError("Pipeline payload is missing document_id.")

    source_file_path = str(
        (pipeline_payload.get("outputs") or {}).get("source_file_path", "")
    )
    upload_metadata = _load_upload_metadata(document_id)
    if not source_file_path:
        source_file_path = str(upload_metadata.get("saved_path", ""))

    supplier_profile = (pipeline_payload.get("outputs") or {}).get("supplier_profile") or {}
    if not supplier_profile or supplier_profile.get("supplier_id") in {None, "", "unknown"}:
        supplier_profile = infer_supplier_profile(
            source_file_path=source_file_path,
            extracted_text=json.dumps(
                (pipeline_payload.get("steps") or {}).get("extraction", {}).get("result"),
                ensure_ascii=False,
            ),
        )

    review = pipeline_payload.get("review") or {}
    if not review:
        pipeline_status = str(pipeline_payload.get("status", ""))
        if pipeline_status == "pipeline_completed":
            review = {
                "status": "approved_auto",
                "recommended_action": "auto_approve",
            }
        elif pipeline_status == "pipeline_completed_with_validation_issues":
            review = {
                "status": "needs_human_review",
                "recommended_action": "manual_review",
            }
        else:
            review = {
                "status": "failed_extraction",
                "recommended_action": "reprocess_or_manual_entry",
            }
    extraction_step = (pipeline_payload.get("steps") or {}).get("extraction") or {}
    route_result = extraction_step.get("result") or {}
    document_type = str(route_result.get("document_type", "unknown"))

    with _get_connection() as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

        connection.execute(
            """
            INSERT INTO documents (
                document_id, source_file_path, supplier_id, supplier_name, supplier_confidence,
                original_filename, saved_filename, file_extension, file_size_bytes, upload_time,
                upload_metadata_path,
                pipeline_status, review_status, review_action, document_type, processed_at,
                pipeline_json_path, extraction_json_path, classification_json_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                source_file_path or None,
                supplier_profile.get("supplier_id"),
                supplier_profile.get("display_name"),
                supplier_profile.get("confidence"),
                upload_metadata.get("original_filename"),
                upload_metadata.get("saved_filename"),
                upload_metadata.get("file_extension"),
                upload_metadata.get("file_size_bytes"),
                upload_metadata.get("upload_time"),
                upload_metadata.get("upload_metadata_path") or str(get_upload_metadata_path(document_id)),
                pipeline_payload.get("status"),
                review.get("status"),
                review.get("recommended_action"),
                document_type,
                datetime.utcnow().isoformat(),
                (pipeline_payload.get("outputs") or {}).get("pipeline_json_path"),
                (pipeline_payload.get("outputs") or {}).get("extraction_json_path"),
                (pipeline_payload.get("outputs") or {}).get("classification_json_path"),
            ),
        )

        for page_payload in _iter_page_payloads(route_result):
            page_number = int(page_payload["page_number"])
            page_document_type = str(page_payload.get("document_type", "unknown"))
            page_extraction_result = page_payload.get("extraction_result") or {}
            rows = page_extraction_result.get("rows") or []

            if page_document_type == "daily_invoice":
                header, footer_summary = _extract_daily_header_and_footer(page_extraction_result)
            else:
                header, footer_summary = _extract_monthly_header_and_footer(page_extraction_result)

            connection.execute(
                """
                INSERT INTO document_pages (
                    document_id, page_number, document_type, extraction_status, row_count,
                    header_json, footer_summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    page_number,
                    page_document_type,
                    page_extraction_result.get("status"),
                    int(page_extraction_result.get("row_count") or 0),
                    _json_or_none(header),
                    _json_or_none(footer_summary),
                ),
            )

            if page_document_type == "monthly_statement":
                for row in rows:
                    movement_date = _normalize_iso_date(row.get("date"))
                    connection.execute(
                        """
                        INSERT INTO monthly_statement_rows (
                            document_id, page_number, supplier_id, movement_date, movement_month,
                            reference_number, note, debit, credit, balance
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            page_number,
                            supplier_profile.get("supplier_id"),
                            movement_date,
                            _month_key(movement_date),
                            row.get("reference_number"),
                            row.get("note"),
                            _safe_float(row.get("debit")),
                            _safe_float(row.get("credit")),
                            _safe_float(row.get("balance")),
                        ),
                    )

            if page_document_type == "daily_invoice":
                invoice_date = _normalize_iso_date(
                    header.get("invoice_date") or page_extraction_result.get("invoice_date")
                )
                customer_name = header.get("customer_name") or header.get("account_name")
                invoice_number = header.get("invoice_number") or header.get("document_number")

                for row in rows:
                    connection.execute(
                        """
                        INSERT INTO daily_invoice_rows (
                            document_id, page_number, supplier_id, invoice_date, invoice_month,
                            invoice_number, customer_name, item_name, quantity_text, price,
                            discount_text, total
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            page_number,
                            supplier_profile.get("supplier_id"),
                            invoice_date,
                            _month_key(invoice_date),
                            invoice_number or row.get("invoice_number"),
                            customer_name,
                            row.get("item_name") or row.get("item_name_ar"),
                            row.get("quantity"),
                            _safe_float(row.get("price")),
                            row.get("discount"),
                            _safe_float(row.get("total")),
                        ),
                    )

        connection.commit()

    return {
        "document_id": document_id,
        "database_path": str(DB_PATH),
        "supplier_id": supplier_profile.get("supplier_id"),
        "document_type": document_type,
        "status": "persisted",
    }


def persist_pipeline_result_by_document_id(document_id: str) -> dict:
    document_id = str(document_id).strip()
    if not document_id:
        raise ValueError("document_id is required.")

    pipeline_json_path = PROCESSED_PIPELINE_JSON_DIR / f"{document_id}.json"
    if not pipeline_json_path.exists():
        raise FileNotFoundError(
            f"Pipeline JSON not found for document_id={document_id}: {pipeline_json_path}"
        )

    with pipeline_json_path.open("r", encoding="utf-8") as file:
        pipeline_payload = json.load(file)

    persistence_result = persist_pipeline_payload(pipeline_payload)
    persistence_result["source_pipeline_json_path"] = str(pipeline_json_path)
    return persistence_result


def get_supplier_monthly_ledger(
    year: int,
    month: int,
    supplier_id: str | None = None,
) -> dict:
    init_database()
    month_key = f"{year:04d}-{month:02d}"
    params: list[object] = [month_key]
    supplier_filter_sql = ""
    if supplier_id:
        supplier_filter_sql = "AND msr.supplier_id = ?"
        params.append(supplier_id)

    with _get_connection() as connection:
        rows = connection.execute(
            f"""
            WITH filtered AS (
                SELECT
                    msr.supplier_id,
                    COALESCE(d.supplier_name, msr.supplier_id) AS supplier_name,
                    msr.document_id,
                    msr.movement_date,
                    COALESCE(msr.debit, 0) AS debit,
                    COALESCE(msr.credit, 0) AS credit,
                    msr.balance
                FROM monthly_statement_rows msr
                JOIN documents d ON d.document_id = msr.document_id
                WHERE msr.movement_month = ?
                {supplier_filter_sql}
            )
            SELECT
                supplier_id,
                supplier_name,
                COUNT(DISTINCT document_id) AS statement_document_count,
                COUNT(*) AS movement_row_count,
                ROUND(SUM(debit), 3) AS ordered_amount,
                ROUND(SUM(credit), 3) AS payment_amount,
                ROUND(SUM(debit) - SUM(credit), 3) AS net_movement
            FROM filtered
            GROUP BY supplier_id, supplier_name
            ORDER BY ordered_amount DESC, supplier_name ASC
            """,
            params,
        ).fetchall()

        summaries: list[dict] = []
        for row in rows:
            closing_balance_row = connection.execute(
                """
                SELECT balance, movement_date
                FROM monthly_statement_rows
                WHERE supplier_id = ? AND movement_month = ?
                ORDER BY movement_date DESC, id DESC
                LIMIT 1
                """,
                (row["supplier_id"], month_key),
            ).fetchone()

            summaries.append(
                {
                    "supplier_id": row["supplier_id"],
                    "supplier_name": row["supplier_name"],
                    "statement_document_count": row["statement_document_count"],
                    "movement_row_count": row["movement_row_count"],
                    "ordered_amount": row["ordered_amount"] or 0.0,
                    "payment_amount": row["payment_amount"] or 0.0,
                    "net_movement": row["net_movement"] or 0.0,
                    "closing_balance": (
                        closing_balance_row["balance"] if closing_balance_row is not None else None
                    ),
                    "basis": "monthly_statement_debit_credit_ledger",
                }
            )

    return {
        "year": year,
        "month": month,
        "month_key": month_key,
        "supplier_filter": supplier_id,
        "database_path": str(DB_PATH),
        "supplier_summaries": summaries,
    }


def list_recent_documents(limit: int = 50) -> dict:
    init_database()
    with _get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                document_id,
                supplier_id,
                supplier_name,
                original_filename,
                saved_filename,
                file_extension,
                file_size_bytes,
                upload_time,
                document_type,
                pipeline_status,
                review_status,
                review_action,
                processed_at,
                source_file_path
            FROM documents
            ORDER BY processed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "limit": limit,
        "database_path": str(DB_PATH),
        "documents": [dict(row) for row in rows],
    }
