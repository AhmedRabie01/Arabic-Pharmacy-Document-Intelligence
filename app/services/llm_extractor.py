import json
import requests

from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def build_invoice_extraction_prompt(ocr_text: str) -> str:
    return f"""
You are an information extraction system for Arabic pharmacy supplier invoices.

Task:
Extract structured data from the OCR text.

Important rules:
- Do not invent missing values.
- If a value is not found, use null.
- Preserve Arabic names as they appear.
- Return valid JSON only.
- Do not add explanation.
- This document may be a daily invoice, monthly statement, return invoice, or mixed document.

Return this JSON schema:

{{
  "document_type": "daily_invoice | monthly_statement | return_invoice | mixed_document | unknown",
  "supplier_name": null,
  "pharmacy_name": null,
  "invoice_number": null,
  "invoice_date": null,
  "statement_month": null,
  "total_amount": null,
  "items": [
    {{
      "item_name": null,
      "quantity": null,
      "unit_price": null,
      "discount": null,
      "line_total": null,
      "invoice_reference": null,
      "notice_number": null,
      "movement_date": null
    }}
  ],
  "warnings": []
}}

OCR text:
\"\"\"
{ocr_text}
\"\"\"
"""


def extract_invoice_json_with_ollama(ocr_text: str) -> dict:
    prompt = build_invoice_extraction_prompt(ocr_text)

    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    response_text = data.get("response", "")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "document_type": "unknown",
            "supplier_name": None,
            "pharmacy_name": None,
            "invoice_number": None,
            "invoice_date": None,
            "statement_month": None,
            "total_amount": None,
            "items": [],
            "warnings": [
                "LLM did not return valid JSON",
                response_text
            ]
        }