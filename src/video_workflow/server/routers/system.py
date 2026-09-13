from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.video_workflow.logging_runtime import clear_logs, log_file_path, runtime_log_handler
from src.video_workflow.runtime_settings import runtime_settings
from src.video_workflow.model_registry import model_registry


logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


class SettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    clear_keys: list[str] = Field(default_factory=list)


class ModelRouteUpdateRequest(BaseModel):
    connection_id: str
    model_id: str


class ModelTestRequest(BaseModel):
    model_id: str | None = None


class ModelManualCreateRequest(BaseModel):
    model_id: str
    label: str = ""
    capabilities: list[str] = Field(default_factory=lambda: ["text", "json"])
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_price_per_million: float | None = None


class ProviderConnectionCreateRequest(BaseModel):
    name: str
    protocol: str = "openai_chat"
    base_url: str
    api_key: str | None = None
    models_url: str | None = None
    api_key_header: str = "Authorization"
    enabled: bool = True


class ProviderConnectionUpdateRequest(BaseModel):
    name: str | None = None
    protocol: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    models_url: str | None = None
    api_key_header: str | None = None
    enabled: bool | None = None


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


@router.get("/model-registry")
async def get_model_registry():
    """Return public provider/model catalog data; API keys are masked."""
    return model_registry.public_payload()


@router.post("/model-registry/connections")
async def create_model_registry_connection(request: ProviderConnectionCreateRequest):
    try:
        model_registry.add_connection(request.model_dump())
        return model_registry.public_payload()
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/model-registry/connections/{connection_id}")
async def update_model_registry_connection(connection_id: str, request: ProviderConnectionUpdateRequest):
    try:
        model_registry.update_connection(connection_id, request.model_dump(exclude_none=True))
        return model_registry.public_payload()
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/model-registry/connections/{connection_id}")
async def delete_model_registry_connection(connection_id: str):
    try:
        model_registry.delete_connection(connection_id)
        return model_registry.public_payload()
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model-registry/connections/{connection_id}/discover")
async def discover_model_registry_connection(connection_id: str):
    try:
        result = await model_registry.discover(connection_id)
        return {**result, "registry": model_registry.public_payload()}
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model-registry/connections/{connection_id}/test")
async def test_model_registry_connection(connection_id: str, request: ModelTestRequest):
    try:
        result = await model_registry.test_connection(connection_id, request.model_id)
        return {**result, "registry": model_registry.public_payload()}
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model-registry/connections/{connection_id}/models")
async def add_model_registry_model(connection_id: str, request: ModelManualCreateRequest):
    try:
        model_registry.add_model(connection_id, request.model_dump())
        return model_registry.public_payload()
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/model-registry/routes/{route_id}")
async def update_model_registry_route(route_id: str, request: ModelRouteUpdateRequest):
    try:
        model_registry.set_route(route_id, request.connection_id, request.model_id)
        from src.video_workflow.server.routers import projects
        projects.render_queue.reload_settings()
        return model_registry.public_payload()
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
