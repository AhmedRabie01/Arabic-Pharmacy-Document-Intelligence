from __future__ import annotations

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import PROCESSED_PIPELINE_JSON_DIR
from app.services.database_service import init_database, persist_pipeline_payload


def main() -> int:
    init_database()
    processed = 0
    failed: list[dict] = []

    for pipeline_path in sorted(PROCESSED_PIPELINE_JSON_DIR.glob("*.json")):
        try:
            with pipeline_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            persist_pipeline_payload(payload)
            processed += 1
        except Exception as error:
            failed.append(
                {
                    "path": str(pipeline_path),
                    "error": str(error),
                }
            )

    print(
        json.dumps(
            {
                "processed": processed,
                "failed_count": len(failed),
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
