# Docker Usage

## Build and run the API

```powershell
docker compose up --build api
```

API URLs:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/app`

## Notes

- The compose setup assumes Ollama is running on the host machine.
- Inside Docker, the app reaches Ollama through:
  - `http://host.docker.internal:11434`
- Persistent project data is mounted from the host:
  - `rawdata/`
  - `processed/`
  - `data/`
  - `invoice_ocr.db`

## First run checklist

1. Start Ollama on the host if you want Chandra / Qwen / GLM providers.
2. Run:

```powershell
docker compose up --build api
```

3. Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/app`

## Useful commands

Stop containers:

```powershell
docker compose down
```

Rebuild after code changes:

```powershell
docker compose up --build api
```
