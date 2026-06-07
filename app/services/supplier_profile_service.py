from __future__ import annotations

from pathlib import Path
import re


SUPPLIER_PROFILES = [
    {
        "supplier_id": "elfath",
        "display_name": "الفتح",
        "aliases": [
            "الفتح",
            "al-fath",
            "al-fath pharma",
            "مخزن الفتح فارما",
        ],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "etihad",
        "display_name": "الاتحاد",
        "aliases": ["الاتحاد", "etihad"],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "dar_elsaydala",
        "display_name": "دار الصيدلة",
        "aliases": [
            "دار الصيدلة",
            "دار الصيادلة",
            "شركة دار الصيادلة",
            "pharmachain",
        ],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "elshahat",
        "display_name": "الشحات",
        "aliases": ["الشحات", "مخزن الشحات", "الشحات فارم"],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "tesla",
        "display_name": "تيسلا",
        "aliases": ["تيسلا", "tesla"],
        "supported_document_types": ["monthly_statement", "daily_invoice", "return_invoice"],
    },
    {
        "supplier_id": "ibn_sina",
        "display_name": "ابن سينا",
        "aliases": ["ابن سينا", "ibn sina"],
        "supported_document_types": ["monthly_statement", "daily_invoice", "return_invoice"],
    },
    {
        "supplier_id": "nest_pharma",
        "display_name": "نست فارما",
        "aliases": ["نست فارما", "nest pharma"],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "pharma",
        "display_name": "فارما",
        "aliases": ["فارما", "pharma"],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
    {
        "supplier_id": "ramco",
        "display_name": "رامكو",
        "aliases": ["رامكو", "ramco"],
        "supported_document_types": ["monthly_statement", "daily_invoice"],
    },
]


def _normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ة", "ه").replace("ى", "ي")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def infer_supplier_profile(
    source_file_path: str,
    extracted_text: str | None,
) -> dict:
    source_name = Path(source_file_path).stem
    haystack_parts = [source_name]
    if extracted_text:
        haystack_parts.append(extracted_text[:8000])

    haystack = _normalize_text(" ".join(haystack_parts))
    best_match: dict | None = None

    best_match_confidence = -1.0

    for profile in SUPPLIER_PROFILES:
        for alias in profile["aliases"]:
            normalized_alias = _normalize_text(alias)
            if not normalized_alias:
                continue

            if normalized_alias in haystack:
                confidence = 0.95 if normalized_alias in _normalize_text(source_name) else 0.8
                candidate_match = {
                    "supplier_id": profile["supplier_id"],
                    "display_name": profile["display_name"],
                    "matched_alias": alias,
                    "confidence": confidence,
                    "supported_document_types": profile["supported_document_types"],
                }
                if confidence > best_match_confidence:
                    best_match = candidate_match
                    best_match_confidence = confidence

    if best_match is None:
        return {
            "supplier_id": "unknown",
            "display_name": None,
            "matched_alias": None,
            "confidence": 0.0,
            "supported_document_types": [],
        }

    return best_match
