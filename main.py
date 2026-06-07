from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import WEB_DIR, settings
from app.services.database_service import init_database
from app.utils.file_system import create_required_directories


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_required_directories()
    init_database()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.include_router(router)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def health_check():
    return {
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "debug": settings.DEBUG,
        "status": "running"
    }


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def web_app():
    return FileResponse(Path(WEB_DIR / "index.html"))
