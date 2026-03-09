from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.video_workflow.server.routers import workflow, files

app = FastAPI(title="VideoWorkflow API", version="0.1.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from src.video_workflow.config import settings

app.include_router(workflow.router, prefix="/api")
app.include_router(files.router, prefix="/api")
app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR), name="static")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
