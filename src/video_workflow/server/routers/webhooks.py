import json
import logging
from pathlib import Path

import aiofiles
import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.video_workflow.config import settings
from src.video_workflow.core.video_task_registry import find_video_task, update_video_task
from src.video_workflow.types import GenerationStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class GrsaiVideoWebhookPayload(BaseModel):
    id: str
    url: str = ""
    progress: int | None = None
    status: str
    fail_reason: str = ""


def _verify_webhook_token(request: Request) -> None:
    expected = settings.GRSAI_WEBHOOK_TOKEN
    if not expected:
        return

    provided = request.headers.get("x-webhook-token") or request.query_params.get("token")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook token")


def _load_script(session_dir: Path) -> dict:
    script_path = session_dir / "script.json"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Script not found")

    with open(script_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _save_script(session_dir: Path, payload: dict) -> None:
    script_path = session_dir / "script.json"
    with open(script_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


async def _download_video_to_scene(session_dir: Path, scene_id: int, remote_url: str) -> str:
    video_dir = session_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    output_path = video_dir / f"{scene_id}_video.mp4"
    timeout = httpx.Timeout(connect=20.0, read=180.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(remote_url)
        response.raise_for_status()
        async with aiofiles.open(output_path, "wb") as file_handle:
            await file_handle.write(response.content)

    return str(output_path)


def _apply_scene_status(
    script_payload: dict,
    scene_id: int,
    status: str,
    video_path: str | None = None,
    fail_reason: str | None = None,
) -> bool:
    scenes = script_payload.get("scenes")
    if not isinstance(scenes, list):
        return False

    target_scene = None
    for scene in scenes:
        if isinstance(scene, dict) and scene.get("id") == scene_id:
            target_scene = scene
            break

    if not isinstance(target_scene, dict):
        return False

    normalized_status = status.strip().lower()
    if normalized_status == "succeeded":
        target_scene["video_status"] = GenerationStatus.COMPLETED.value
        target_scene["error_message"] = None
        if video_path:
            target_scene["video_path"] = video_path
    elif normalized_status == "failed":
        target_scene["video_status"] = GenerationStatus.FAILED.value
        target_scene["error_message"] = fail_reason or "Video generation failed"
    else:
        target_scene["video_status"] = GenerationStatus.PROCESSING.value

    return True


@router.post("/grsai/video")
async def handle_grsai_video_webhook(payload: GrsaiVideoWebhookPayload, request: Request):
    _verify_webhook_token(request)

    session_dir, task_entry = find_video_task(settings.OUTPUT_DIR, payload.id)
    if session_dir is None or task_entry is None:
        logger.warning("Received webhook for unknown task_id=%s", payload.id)
        return {"code": 0, "msg": "ignored"}

    scene_id = task_entry.get("scene_id")
    if not isinstance(scene_id, int):
        logger.error("Task entry missing scene_id for task_id=%s", payload.id)
        return {"code": 0, "msg": "ignored"}

    update_video_task(
        session_dir=session_dir,
        task_id=payload.id,
        status=payload.status,
        progress=payload.progress,
        url=payload.url or None,
        fail_reason=payload.fail_reason or None,
    )

    script_payload = _load_script(session_dir)

    normalized_status = payload.status.strip().lower()
    resolved_video_path = None

    if normalized_status == "succeeded" and payload.url:
        try:
            resolved_video_path = await _download_video_to_scene(session_dir, scene_id, payload.url)
        except Exception as exc:
            logger.error("Webhook video download failed for task_id=%s: %s", payload.id, exc)
            normalized_status = "failed"
            payload.fail_reason = f"Download failed: {exc}"

    updated = _apply_scene_status(
        script_payload=script_payload,
        scene_id=scene_id,
        status=normalized_status,
        video_path=resolved_video_path,
        fail_reason=payload.fail_reason,
    )
    if updated:
        _save_script(session_dir, script_payload)

    return {"code": 0, "msg": "success"}
