from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.video_workflow.logging_runtime import clear_logs, log_file_path, runtime_log_handler
from src.video_workflow.runtime_settings import runtime_settings


logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


class SettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    clear_keys: list[str] = Field(default_factory=list)


@router.get("/settings")
async def get_settings():
    return runtime_settings.public_payload()


@router.put("/settings")
async def update_settings(request: SettingsUpdateRequest):
    try:
        runtime_settings.update(request.values, request.clear_keys)
        from src.video_workflow.server.routers import projects
        projects.render_queue.reload_settings()
        logger.info("运行配置已更新: %s", ", ".join(sorted(set(request.values) | set(request.clear_keys))))
        return runtime_settings.public_payload()
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/logs")
async def get_logs(
    level: str | None = None,
    search: str = "",
    limit: int = Query(default=500, ge=1, le=3000),
):
    return {"records": runtime_log_handler.query(level, search, limit), "file": str(log_file_path())}


@router.delete("/logs")
async def delete_logs():
    clear_logs()
    logger.info("运行日志已由用户清空")
    return {"cleared": True}


@router.get("/logs/download")
async def download_logs():
    path = log_file_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="video_workflow.log")
