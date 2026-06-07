# Testing

This project currently uses a mix of:

- lightweight automated verification
- dataset-based evaluation
- manual full-pipeline runs for heavy OCR / vision providers

The old root-level `test_*.py` files were removed because they were not real tests. They were hardcoded local debug scripts tied to one developer machine and one document ID.

## Automated checks

### 1. Compile all Python files

```powershell
@'
import py_compile
from pathlib import Path

failed = []
for path in Path('.').rglob('*.py'):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        failed.append((str(path), str(exc)))

print('FAILED', len(failed))
for path, error in failed:
    print(path)
    print(error)
'@ | C:\Users\ahmed\anaconda3\envs\ocr\python.exe -
```

Expected result:
- `FAILED 0`

### 2. Evaluation against approved reviewed pages

```powershell
C:\Users\ahmed\anaconda3\envs\ocr\python.exe scripts/evaluate_approved_dataset.py
```

Output:
- writes `data/approved_dataset/evaluation_report.json`

Note:
- this requires a local approved dataset under `data/approved_dataset/`
- that dataset is not included in the public repository

Use this to track:
- page-type accuracy
- row-count match rate
- row-field accuracy
- footer-summary accuracy
- evaluation coverage

## API smoke tests

### 1. Start backend

```powershell
conda run -n ocr uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Health endpoint

Open:
- `http://127.0.0.1:8000/`

Expected JSON:

```json
{
  "app": "Pharmacy Invoice OCR System",
  "environment": "local",
  "debug": false,
  "status": "running"
}
```

### 3. Web app

Open:
- `http://127.0.0.1:8000/app`

Expected:
- HTML page loads successfully

### 4. Upload API

Upload a PDF through:
- `POST /documents/upload`

Expected:
- `document_id`
- saved path under `rawdata/all_invoices`

## Manual full-pipeline testing

Use the built-in web app:

- `http://127.0.0.1:8000/app`

This remains intentionally manual for heavy document runs because:
- PaddleOCR is heavy
- OpenAI / Ollama providers can be slow
- local model availability and API credentials affect runtime and output

The web app exposes:
- `auto`
- `paddleocr`
- `chandra_ollama`
- `openai`

and lets you inspect:
- high-level result summary
- extraction preview
- advanced pipeline JSON
- review decision

## What is not covered by automated tests yet

- full heavy OCR pipeline execution on every approved document
- deterministic Chandra / Qwen / GLM output checks
- signature model network/download behavior
- UI visual regression checks

These remain manual/runtime-dependent surfaces.
