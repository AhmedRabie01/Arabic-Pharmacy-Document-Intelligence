# Arabic Pharmacy Document Intelligence

Local-first OCR and document intelligence system for Arabic pharmacy supplier documents, focused on:

- daily supplier invoices
- monthly supplier statements

The system is designed as a production-style extraction service, not a notebook demo. It provides:

- FastAPI backend
- browser-based review UI
- local OCR pipeline with PaddleOCR
- Ollama-based layout fallback
- OpenAI direct extraction mode
- validation and review gating
- SQLite persistence and supplier analytics

## Product use case

This project helps pharmacies and accounting teams turn supplier PDFs and scanned pages into structured, reviewable business data.

Typical outputs include:

- monthly statement movement rows
- monthly balances and statement summaries
- daily invoice item rows
- supplier-level ordered amount tracking
- recent processed documents and review status

## Current scope

Supported document families:

- `monthly_statement`
- `daily_invoice`

Current providers:

- `auto`
- `paddleocr`
- `chandra_ollama`
- `openai`

Provider behavior:

- `auto`: PaddleOCR first, Chandra fallback
- `paddleocr`: current local OCR-first path
- `chandra_ollama`: current layout-aware local path
- `openai`: direct page extraction path that bypasses Paddle spatial reconstruction

## Core features

- Upload and process PDF or image documents
- Detect digital-text PDFs and preserve page text
- Extract structured rows for monthly statements and daily invoices
- Validate extracted content before approval
- Mark results as:
  - `approved_auto`
  - `needs_human_review`
  - `failed_extraction`
- Save processed results on demand to SQLite
- Track supplier ordered amounts from monthly statements
- Measure pipeline timing by stage
- Record OpenAI token usage and estimated cost for OpenAI runs

## Architecture

High-level flow:

1. Upload document
2. Load and render pages
3. Preprocess page images
4. Run provider path
5. Classify / normalize extracted result
6. Validate rows and signature rules
7. Save outputs and optional database record
8. Review in the web UI

Local path:

- PaddleOCR for OCR text
- Chandra via Ollama for fallback layout understanding
- spatial mapping and extractor routing

OpenAI path:

- direct page extraction with compact few-shot examples
- bypasses:
  - PaddleOCR
  - Chandra
  - spatial mapping
  - row grouping
  - column assignment

## Main outputs

Generated runtime artifacts are written locally under `processed/` and are intentionally excluded from Git.

Important output layers:

- `processed/metadata`
- `processed/text`
- `processed/ocr_json`
- `processed/chandra_json`
- `processed/openai_json`
- `processed/spatial_json`
- `processed/classification_json`
- `processed/extraction_json`
- `processed/signature_json`
- `processed/pipeline_json`

## Supplier analytics

The application includes a supplier ledger view powered by SQLite.

Example business question:

- `الفتح` ordered amount this month
- `الاتحاد` ordered amount this month

The ledger is computed from saved monthly-statement movements and exposed through the API and web UI.

## Quick start

### 1. Install dependencies

```powershell
conda create -n ocr python=3.11 -y
conda activate ocr
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` from `.env.example` and set only what you need.

For OpenAI mode:

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
```

For local Ollama mode:

```env
CHANDRA_BASE_URL=http://localhost:11434
CHANDRA_MODEL=fredrezones55/chandra-ocr-2:patch
```

### 3. Run the backend

```powershell
conda run -n ocr uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- API health: `http://127.0.0.1:8000/`
- Web app: `http://127.0.0.1:8000/app`

## Docker

Backend container:

```powershell
docker compose up --build api
```

See [DOCKER.md](./DOCKER.md) for details.

## API surface

Main routes:

- `POST /documents/upload`
- `POST /documents/process`
- `POST /documents/save-result`
- `GET /documents/analytics/suppliers/monthly-ledger`
- `GET /documents/analytics/documents/recent`

## OpenAI mode

When `document_ai_provider=openai`:

- pages are sent to OpenAI directly
- compact monthly and daily few-shot examples are used
- OpenAI returns normalized JSON
- token usage and estimated cost are recorded
- signature presence and signer-name detection are attempted as best effort

This mode is useful for benchmarking direct vision extraction against the local OCR pipeline.

## Validation and review

The system is intentionally not “extract and trust blindly”.

Validation includes:

- monthly date / numeric checks
- daily quantity / price / total checks
- signature requirement rules by document type

Results are then routed into a review decision:

- auto-approve
- manual review
- failed extraction

## Testing

See [TESTING.md](./TESTING.md).

Automated checks currently cover:

- Python compile integrity
- API surface smoke checks
- evaluation script execution

Heavy OCR and provider benchmarking remain runtime-dependent and are tested through the app and saved outputs.

## Repository hygiene

This repository is prepared for public code sharing:

- private client documents are ignored
- generated pipeline outputs are ignored
- local databases are ignored
- secrets are ignored

Public repo should contain:

- source code
- configuration examples
- Docker files
- evaluation scripts
- product documentation

Private review datasets and client-derived documents are intentionally excluded from the public repository.

Private/local runtime state should stay outside Git.

## Positioning

This repository is suitable as:

- a freelance-ready OCR / document AI service foundation
- a local-first invoice intelligence backend
- a supplier statement extraction workflow
- a portfolio project demonstrating production-minded AI engineering

## Notes

- OpenAI direct extraction is integrated, but output quality still depends on the document family and page quality.
- Local provider paths remain important for cost control and offline-first operation.
- For real deployment, you should benchmark on your own supplier set and tune provider choice by document family.
