# Contributing

## Scope

This repository is structured as a product-oriented document intelligence service. Changes should preserve:

- support for `monthly_statement` and `daily_invoice`
- clear provider separation
- review-safe handling of client data
- reproducible evaluation and testing workflow

## Before opening a pull request

1. Keep client documents, generated outputs, and secrets out of Git.
2. Prefer small, task-focused commits.
3. Update docs when behavior, setup, or API contracts change.
4. Include validation notes for extraction or analytics changes.

## Development checks

Run the service locally:

```powershell
conda run -n ocr uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Run the evaluator:

```powershell
C:\Users\ahmed\anaconda3\envs\ocr\python.exe scripts/evaluate_approved_dataset.py
```

Review testing guidance:

- [TESTING.md](TESTING.md)

## Data handling

- Do not commit real supplier PDFs, screenshots, or generated `processed/` outputs.
- Use sanitized examples only.
- Keep `.env` private and update `.env.example` when config changes.

## Pull request quality bar

Good pull requests for this repo are:

- narrow in scope
- clear about the affected provider path
- explicit about validation impact
- explicit about user-facing UI or analytics changes
