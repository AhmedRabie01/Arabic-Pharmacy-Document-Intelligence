from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.database_service import get_supplier_monthly_ledger, persist_pipeline_payload

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def build_monthly_payload(
    *,
    document_id: str,
    source_file_name: str,
    supplier_id: str,
    supplier_name: str,
    rows: list[dict],
    footer_summary: dict,
) -> dict:
    return {
        "document_id": document_id,
        "status": "pipeline_completed",
        "review": {
            "status": "approved_auto",
            "recommended_action": "auto_approve",
        },
        "outputs": {
            "source_file_path": f"demo/{source_file_name}",
            "supplier_profile": {
                "supplier_id": supplier_id,
                "display_name": supplier_name,
                "confidence": 1.0,
            },
            "pipeline_json_path": None,
            "extraction_json_path": None,
            "classification_json_path": None,
        },
        "steps": {
            "extraction": {
                "result": {
                    "document_type": "monthly_statement",
                    "extraction_result": {
                        "status": "completed",
                        "row_count": len(rows),
                        "header": {
                            "supplier_name": supplier_name,
                            "statement_type": "demo_seed",
                        },
                        "footer_summary": footer_summary,
                        "rows": rows,
                    },
                }
            }
        },
    }


def main() -> None:
    today = date.today()
    month_start = today.replace(day=1)
    year = month_start.year
    month = month_start.month

    demo_documents = [
        build_monthly_payload(
            document_id="demo_ledger_elfath_2026_06",
            source_file_name="الفتح_demo_statement.pdf",
            supplier_id="elfath",
            supplier_name="الفتح",
            rows=[
                {
                    "date": month_start.isoformat(),
                    "reference_number": "D-1001",
                    "note": "مبيعات",
                    "debit": "12000",
                    "credit": "0",
                    "balance": "12000",
                },
                {
                    "date": month_start.replace(day=10).isoformat(),
                    "reference_number": "D-1002",
                    "note": "مبيعات",
                    "debit": "8000",
                    "credit": "0",
                    "balance": "20000",
                },
                {
                    "date": month_start.replace(day=18).isoformat(),
                    "reference_number": "P-1001",
                    "note": "استلام نقديه",
                    "debit": "0",
                    "credit": "5000",
                    "balance": "15000",
                },
            ],
            footer_summary={
                "total_debit": "20000",
                "total_credit": "5000",
                "current_balance": "15000",
            },
        ),
        build_monthly_payload(
            document_id="demo_ledger_etihad_2026_06",
            source_file_name="الاتحاد_demo_statement.pdf",
            supplier_id="etihad",
            supplier_name="الاتحاد",
            rows=[
                {
                    "date": month_start.replace(day=2).isoformat(),
                    "reference_number": "D-2001",
                    "note": "مبيعات",
                    "debit": "18000",
                    "credit": "0",
                    "balance": "18000",
                },
                {
                    "date": month_start.replace(day=12).isoformat(),
                    "reference_number": "D-2002",
                    "note": "مبيعات",
                    "debit": "12000",
                    "credit": "0",
                    "balance": "30000",
                },
                {
                    "date": month_start.replace(day=22).isoformat(),
                    "reference_number": "P-2001",
                    "note": "استلام نقديه",
                    "debit": "0",
                    "credit": "7000",
                    "balance": "23000",
                },
            ],
            footer_summary={
                "total_debit": "30000",
                "total_credit": "7000",
                "current_balance": "23000",
            },
        ),
        build_monthly_payload(
            document_id="demo_ledger_ramco_2026_06",
            source_file_name="رامكو_demo_statement.pdf",
            supplier_id="ramco",
            supplier_name="رامكو",
            rows=[
                {
                    "date": month_start.replace(day=4).isoformat(),
                    "reference_number": "D-3001",
                    "note": "مبيعات",
                    "debit": "9500",
                    "credit": "0",
                    "balance": "9500",
                },
                {
                    "date": month_start.replace(day=11).isoformat(),
                    "reference_number": "D-3002",
                    "note": "مبيعات",
                    "debit": "6000",
                    "credit": "0",
                    "balance": "15500",
                },
                {
                    "date": month_start.replace(day=25).isoformat(),
                    "reference_number": "P-3001",
                    "note": "استلام نقديه",
                    "debit": "0",
                    "credit": "3000",
                    "balance": "12500",
                },
            ],
            footer_summary={
                "total_debit": "15500",
                "total_credit": "3000",
                "current_balance": "12500",
            },
        ),
        build_monthly_payload(
            document_id="demo_ledger_dar_elsaydala_2026_06",
            source_file_name="دار_الصيدلة_demo_statement.pdf",
            supplier_id="dar_elsaydala",
            supplier_name="دار الصيدلة",
            rows=[
                {
                    "date": month_start.replace(day=5).isoformat(),
                    "reference_number": "D-4001",
                    "note": "مبيعات",
                    "debit": "9200",
                    "credit": "0",
                    "balance": "9200",
                },
                {
                    "date": month_start.replace(day=27).isoformat(),
                    "reference_number": "P-4001",
                    "note": "استلام نقديه",
                    "debit": "0",
                    "credit": "9200",
                    "balance": "0",
                },
            ],
            footer_summary={
                "total_debit": "9200",
                "total_credit": "9200",
                "current_balance": "0",
            },
        ),
    ]

    for payload in demo_documents:
        persist_pipeline_payload(payload)

    ledger = get_supplier_monthly_ledger(year=year, month=month)
    print(f"Seeded demo supplier ledger data for {year:04d}-{month:02d}")
    for summary in ledger["supplier_summaries"]:
        print(
            f"- {summary['supplier_name']}: ordered={summary['ordered_amount']}, "
            f"payments={summary['payment_amount']}, closing={summary['closing_balance']}"
        )


if __name__ == "__main__":
    main()
