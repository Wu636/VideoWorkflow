from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.video_workflow.logging_runtime import configure_logging
from src.video_workflow.runtime_settings import runtime_settings

runtime_settings.load()
configure_logging()

from src.video_workflow.server.routers import files, projects, prompt_templates, system, webhooks, workflow


@asynccontextmanager
async def lifespan(_: FastAPI):
    migration = projects.project_service.migrate_all_seedance_shots_to_multimodal()
    if migration["shot_count"]:
        logger.info(
            "Seedance 全模态默认迁移完成: projects=%s shots=%s",
            migration["project_count"],
            migration["shot_count"],
        )
    await projects.render_queue.start()
    try:
        yield
    finally:
        await projects.render_queue.stop()


app = FastAPI(title="VideoWorkflow API", version="1.0.0", lifespan=lifespan)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    # Quick tunnels are used only as an Ark media bridge.  Do not expose the
    # project/settings API through the temporary public hostname.
    host = request.headers.get("host", "").split(":", 1)[0].lower()
    if host.endswith(".trycloudflare.com") and not request.url.path.startswith(
        "/api/projects/seedance-assets/"
    ):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("HTTP %s %s 发生未处理异常", request.method, request.url.path)
        raise
    if request.url.path not in {"/api/logs", "/health"}:
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(level, "HTTP %s %s -> %s (%.0fms)", request.method, request.url.path, response.status_code, (time.monotonic() - started) * 1000)
    return response

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from src.video_workflow.config import settings

app.include_router(workflow.router, prefix="/api")
app.include_router(files.router, prefix="/api")
app.include_router(webhooks.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(prompt_templates.router, prefix="/api")
app.include_router(projects.public_router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR), name="static")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
