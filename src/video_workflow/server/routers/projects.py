from __future__ import annotations

import asyncio
import csv
import io
import logging
import mimetypes
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from src.video_workflow.config import settings
from src.video_workflow.domain import (
    ApprovalStatus,
    AssetRole,
    CharacterProfile,
    FinalizeRequest,
    Project,
    ProjectBrief,
    ProjectStatus,
    Review,
    Shot,
)
from src.video_workflow.integrations.comfyui import ComfyUIClient, H3WorkflowBuilder
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.services.finalize import Finalizer
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.storage import ProjectStore

router = APIRouter(prefix="/projects", tags=["projects"])
logger = logging.getLogger(__name__)
public_router = APIRouter(prefix="/review", tags=["public-review"])

store = ProjectStore(settings.DATABASE_PATH)
project_service = ProjectService(store)
render_queue = RenderQueue(store, project_service)
finalizer = Finalizer(store, project_service)


class StoryboardGenerateRequest(BaseModel):
    shot_count: int | None = Field(default=None, ge=1, le=500)
    count_mode: Literal["manual", "ai"] = "manual"
    user_suggestions: str = Field(default="", max_length=4000)


class ScriptRewriteRequest(BaseModel):
    mode: Literal["auto", "expand", "shorten"] = "auto"
    user_suggestions: str = Field(default="", max_length=4000)


class StoryboardApprovalRequest(BaseModel):
    approved: bool
    comment: str = ""
    reviewer: str = ""


class RenderRequest(BaseModel):
    shot_ids: list[str] | None = None


class KeyframeRequest(BaseModel):
    shot_ids: list[str] | None = None
    image_provider: str | None = None
    image_model: str | None = None
    revision_mode: Literal["fresh", "iterate"] = "fresh"
    user_suggestions: str = Field(default="", max_length=4000)


class KeyframeExportRequest(BaseModel):
    shot_ids: list[str] | None = None


class ReorderRequest(BaseModel):
    shot_ids: list[str]


class PublicReviewRequest(BaseModel):
    target_type: str = "storyboard"
    target_id: str
    decision: ApprovalStatus
    comment: str = ""
    reviewer: str = ""


def _not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def _bundle(project_id: str) -> dict[str, Any]:
    project = store.get_project(project_id)
    if project is None:
        raise _not_found("Project not found")
    return {
        "project": project,
        "shots": store.list_shots(project_id),
        "assets": store.list_assets(project_id),
        "jobs": store.list_jobs(project_id),
        "reviews": store.list_reviews(project_id),
        "deliveries": store.list_deliveries(project_id),
    }


@router.get("")
async def list_projects():
    return store.list_projects()


@router.post("")
async def create_project(brief: ProjectBrief):
    return project_service.create_project(brief)


@router.get("/legacy/sessions")
async def list_legacy_sessions():
    return project_service.list_legacy_sessions()


@router.post("/legacy/import/{session_id}")
async def import_legacy_session(session_id: str):
    try:
        return project_service.import_legacy_session(session_id)
    except FileNotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}")
async def get_project(project_id: str):
    return _bundle(project_id)


@router.put("/{project_id}")
async def update_project(project_id: str, project: Project):
    if project.id != project_id:
        raise HTTPException(status_code=400, detail="Project id mismatch")
    if store.get_project(project_id) is None:
        raise _not_found("Project not found")
    return project_service.update_project(project)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    project_dir = project_service.project_dir(project_id).resolve()
    projects_root = settings.PROJECTS_DIR.resolve()
    if not store.delete_project(project_id):
        raise _not_found("Project not found")
    if project_dir.exists() and projects_root in project_dir.parents:
        shutil.rmtree(project_dir)
    return {"deleted": True}


@router.post("/{project_id}/storyboard/generate")
async def generate_storyboard(project_id: str, request: StoryboardGenerateRequest):
    try:
        shots = await project_service.generate_storyboard(
            project_id,
            request.shot_count,
            request.count_mode,
            request.user_suggestions,
        )
        return {"shots": shots, "project": store.get_project(project_id)}
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{project_id}/brief/analyze")
async def analyze_brief(project_id: str):
    try:
        return await project_service.analyze_brief(project_id)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI 剧本分析失败: {exc}") from exc


@router.post("/{project_id}/brief/rewrite")
async def rewrite_brief(project_id: str, request: ScriptRewriteRequest):
    try:
        return await project_service.rewrite_story(project_id, request.mode, request.user_suggestions)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI 剧本改写失败: {exc}") from exc


def _extract_script_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        raise ValueError("文本编码无法识别，请转换为 UTF-8 后重试")
    if suffix == ".docx":
        from docx import Document
        document = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()).strip()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    raise ValueError("支持 TXT、Markdown、DOCX 和可提取文字的 PDF 剧本")


@router.post("/{project_id}/script/upload")
async def upload_script(project_id: str, file: UploadFile = File(...)):
    project = project_service.require_project(project_id)
    filename = Path(file.filename or "script.txt").name
    content = await file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="剧本文件请控制在 10MB 以内")
    try:
        text = _extract_script_text(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not text:
        raise HTTPException(status_code=400, detail="文件中没有提取到可用文字")
    destination_dir = project_service.project_dir(project_id) / "documents"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid4().hex}{Path(filename).suffix.lower()}"
    destination.write_bytes(content)
    asset = project_service.register_existing_asset(
        project_id,
        destination,
        role=AssetRole.OTHER,
        name=filename,
        description="客户上传的原始剧情脚本",
    )
    asset.mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    store.save_asset(asset)
    project.brief.story = text[:300_000]
    project.ai_recommended_shot_count = None
    project_service.update_project(project)
    return {"asset": asset, "project": project, "extracted_characters": len(project.brief.story)}


@router.post("/{project_id}/storyboard/approve")
async def approve_storyboard(project_id: str, request: StoryboardApprovalRequest):
    try:
        project = project_service.approve_storyboard(project_id, request.approved, request.comment)
        review = Review(
            project_id=project_id,
            target_type="storyboard",
            target_id=project_id,
            decision=ApprovalStatus.APPROVED if request.approved else ApprovalStatus.CHANGES_REQUESTED,
            comment=request.comment,
            reviewer=request.reviewer,
        )
        store.save_review(review)
        return {"project": project, "review": review}
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@router.post("/{project_id}/storyboard/reorder")
async def reorder_storyboard(project_id: str, request: ReorderRequest):
    shots = {shot.id: shot for shot in store.list_shots(project_id)}
    if set(request.shot_ids) != set(shots):
        raise HTTPException(status_code=400, detail="Reorder list must contain every shot exactly once")
    ordered = []
    for ordinal, shot_id in enumerate(request.shot_ids, start=1):
        shot = shots[shot_id]
        shot.ordinal = ordinal
        ordered.append(store.save_shot(shot))
    project_service.invalidate_storyboard(project_id)
    return ordered


@router.get("/{project_id}/storyboard.csv")
async def export_storyboard(project_id: str):
    project = store.get_project(project_id)
    if project is None:
        raise _not_found("Project not found")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["镜号", "时长", "叙事", "对白", "场景", "景别", "机位", "运镜", "主体动作", "通用分镜提示词", "首帧提示词", "H3提示词", "生成模式", "审核状态"])
    storyboard_changed = False
    for shot in store.list_shots(project_id):
        writer.writerow(
            [
                shot.ordinal,
                shot.duration_seconds,
                shot.narrative,
                shot.dialogue,
                shot.scene_description,
                shot.shot_size,
                shot.camera_angle,
                shot.camera_motion,
                shot.subject_motion,
                shot.visual_prompt,
                shot.keyframe_prompt,
                shot.video_prompt,
                (shot.resolved_generation_mode or shot.generation_mode).value,
                shot.approval_status.value,
            ]
        )
    payload = "\ufeff" + output.getvalue()
    return StreamingResponse(
        iter([payload.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{project.id}-storyboard.csv"'},
    )


@router.put("/{project_id}/shots/{shot_id}")
async def update_shot(project_id: str, shot_id: str, shot: Shot):
    if shot.id != shot_id or shot.project_id != project_id:
        raise HTTPException(status_code=400, detail="Shot identity mismatch")
    existing = store.get_shot(shot_id)
    if existing is None:
        raise _not_found("Shot not found")
    if shot.video_prompt != existing.video_prompt:
        shot.video_prompt_source = shot.video_prompt
    shot.version += 1
    saved = store.save_shot(shot)
    project_service.invalidate_storyboard(project_id)
    return saved


@router.post("/{project_id}/shots")
async def create_shot(project_id: str, shot: Shot):
    if shot.project_id != project_id:
        raise HTTPException(status_code=400, detail="Shot project mismatch")
    project_service.require_project(project_id)
    if not shot.video_prompt_source:
        shot.video_prompt_source = shot.video_prompt
    saved = store.save_shot(shot)
    project_service.invalidate_storyboard(project_id)
    return saved


@router.post("/{project_id}/shots/blank")
async def create_blank_shot(project_id: str):
    project_service.require_project(project_id)
    shots = store.list_shots(project_id)
    shot = Shot(
        project_id=project_id,
        ordinal=len(shots) + 1,
        title=f"镜头 {len(shots) + 1}",
        h3_turbo=False,
        h3_steps=20,
    )
    store.save_shot(shot)
    project_service.invalidate_storyboard(project_id)
    return shot


@router.delete("/{project_id}/shots/{shot_id}")
async def delete_shot(project_id: str, shot_id: str):
    shots = [shot for shot in store.list_shots(project_id) if shot.id != shot_id]
    if len(shots) == len(store.list_shots(project_id)):
        raise _not_found("Shot not found")
    for ordinal, shot in enumerate(shots, start=1):
        shot.ordinal = ordinal
    store.replace_shots(project_id, shots)
    project_service.invalidate_storyboard(project_id)
    return {"deleted": True}


@router.post("/{project_id}/assets")
async def upload_asset(
    project_id: str,
    file: UploadFile = File(...),
    role: AssetRole = Form(AssetRole.OTHER),
    name: str = Form(""),
    character_id: str | None = Form(None),
    description: str = Form(""),
):
    project_service.require_project(project_id)
    filename = Path(file.filename or "asset.bin").name
    suffix = Path(filename).suffix
    destination_dir = project_service.project_dir(project_id) / "assets"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid4().hex}{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    asset = project_service.register_existing_asset(
        project_id,
        destination,
        role=role,
        name=name or filename,
        description=description,
    )
    asset.mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    asset.character_id = character_id
    return store.save_asset(asset)


@router.delete("/{project_id}/assets/{asset_id}")
async def delete_asset(project_id: str, asset_id: str):
    asset = store.get_asset(asset_id)
    if asset is None or asset.project_id != project_id:
        raise _not_found("Asset not found")
    if not store.delete_asset(asset_id):
        raise _not_found("Asset not found")
    for shot in store.list_shots(project_id):
        changed = False
        if asset_id in shot.reference_asset_ids:
            shot.reference_asset_ids = [item for item in shot.reference_asset_ids if item != asset_id]
            changed = True
        if shot.keyframe_asset_id == asset_id:
            shot.keyframe_asset_id = None
            shot.image_path = None
            shot.image_status = "pending"
            changed = True
        if shot.last_frame_asset_id == asset_id:
            shot.last_frame_asset_id = None
            changed = True
        if changed:
            store.save_shot(shot)
            storyboard_changed = True
    project = project_service.require_project(project_id)
    character_changed = False
    for character in project.characters:
        if asset_id in character.reference_asset_ids:
            character.reference_asset_ids = [item for item in character.reference_asset_ids if item != asset_id]
            character_changed = True
    if character_changed:
        store.save_project(project)
        storyboard_changed = True
    path = resolve_media_path(asset.path).resolve()
    project_dir = project_service.project_dir(project_id).resolve()
    if path.exists() and project_dir in path.parents:
        path.unlink()
    if storyboard_changed:
        project_service.invalidate_storyboard(project_id)
    return {"deleted": True}


@router.post("/{project_id}/keyframes/generate")
async def generate_keyframes(project_id: str, request: KeyframeRequest):
    try:
        return await project_service.generate_keyframes(
            project_id,
            request.shot_ids,
            request.image_provider,
            request.image_model,
            request.revision_mode,
            request.user_suggestions,
        )
    except Exception as exc:
        logger.exception("项目 %s 的分镜首帧生成失败", project_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _safe_archive_component(value: str, fallback: str) -> str:
    clean = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", (value or "").strip()).strip(" ._")
    return (clean or fallback)[:80]


def _build_keyframe_archive(project_id: str, shot_ids: list[str] | None) -> tuple[Path, str, int]:
    project = project_service.require_project(project_id)
    selected = set(shot_ids) if shot_ids is not None else None
    entries: list[tuple[Shot, Path]] = []
    for shot in store.list_shots(project_id):
        if selected is not None and shot.id not in selected:
            continue
        asset = store.get_asset(shot.keyframe_asset_id) if shot.keyframe_asset_id else None
        if asset is None or asset.project_id != project_id:
            continue
        path = resolve_media_path(asset.path)
        if path.is_file():
            entries.append((shot, path))
    if not entries:
        raise ValueError("所选镜头中没有可导出的已生成分镜图")

    handle = tempfile.NamedTemporaryFile(prefix="videoworkflow-keyframes-", suffix=".zip", delete=False)
    archive_path = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for shot, source in entries:
                title = _safe_archive_component(shot.title, f"镜头_{shot.ordinal:03d}")
                suffix = source.suffix.lower() or ".png"
                archive.write(source, arcname=f"镜头_{shot.ordinal:03d}_{title}{suffix}")
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    project_title = _safe_archive_component(project.brief.title, project.id)
    return archive_path, f"{project_title}_分镜图_{len(entries)}张.zip", len(entries)


@router.post("/{project_id}/keyframes/export")
async def export_keyframes(project_id: str, request: KeyframeExportRequest):
    try:
        archive_path, filename, _ = await asyncio.to_thread(
            _build_keyframe_archive,
            project_id,
            request.shot_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.post("/{project_id}/render/plan")
async def plan_render(project_id: str):
    try:
        shots = project_service.plan_all_shots(project_id)
        project = project_service.require_project(project_id)
        project.status = ProjectStatus.RENDER_PLAN_APPROVED
        store.save_project(project)
        return shots
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/render")
async def enqueue_render(project_id: str, request: RenderRequest):
    try:
        return render_queue.enqueue(project_id, request.shot_ids)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/jobs")
async def list_jobs(project_id: str):
    project_service.require_project(project_id)
    return store.list_jobs(project_id)


@router.post("/{project_id}/jobs/{job_id}/cancel")
async def cancel_job(project_id: str, job_id: str):
    job = store.get_job(job_id)
    if job is None or job.project_id != project_id:
        raise _not_found("Job not found")
    return await render_queue.cancel(job_id)


@router.get("/{project_id}/comfyui/preflight")
async def comfyui_preflight(project_id: str):
    project_service.require_project(project_id)
    client = ComfyUIClient()
    builder = H3WorkflowBuilder()
    try:
        system = await client.health()
        preflight = await client.preflight(builder.preflight_requirements())
        return {"online": True, "system": system, **preflight}
    except Exception as exc:
        return {"online": False, "ok": False, "error": str(exc)}


@router.post("/{project_id}/subtitles/generate")
async def generate_subtitles(project_id: str):
    project_service.require_project(project_id)
    lines: list[str] = []
    cursor = 0.0
    for index, shot in enumerate(store.list_shots(project_id), start=1):
        text = (shot.dialogue or shot.narrative).strip()
        if not text:
            cursor += shot.duration_seconds
            continue
        start = cursor
        end = cursor + shot.duration_seconds
        lines.extend([str(index), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""])
        cursor = end
    path = project_service.project_dir(project_id) / "assets" / f"subtitles-{uuid4().hex}.srt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return project_service.register_existing_asset(project_id, path, AssetRole.OTHER, "自动字幕")


@router.post("/{project_id}/finalize")
async def finalize_project(project_id: str, request: FinalizeRequest):
    try:
        return await asyncio.to_thread(finalizer.finalize, project_id, request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/download/{kind}/{item_id}")
async def download_project_file(project_id: str, kind: str, item_id: str):
    if kind == "asset":
        item = store.get_asset(item_id)
        path = resolve_media_path(item.path) if item and item.project_id == project_id else None
    elif kind == "job":
        item = store.get_job(item_id)
        path = resolve_media_path(item.output_path) if item and item.project_id == project_id and item.output_path else None
    elif kind == "delivery":
        item = next((delivery for delivery in store.list_deliveries(project_id) if delivery.id == item_id), None)
        path = resolve_media_path(item.output_path) if item else None
    elif kind == "preview":
        item = next((delivery for delivery in store.list_deliveries(project_id) if delivery.id == item_id), None)
        path = resolve_media_path(item.preview_path) if item and item.preview_path else None
    else:
        raise HTTPException(status_code=400, detail="Invalid file kind")
    if path is None or not path.exists():
        raise _not_found("File not found")
    return FileResponse(path, filename=path.name)


@public_router.get("/{token}")
async def public_review(token: str):
    project = next((item for item in store.list_projects() if item.review_token == token), None)
    if project is None:
        raise _not_found("Review link not found")
    bundle = _bundle(project.id)
    bundle["project"] = project.model_copy(update={"review_token": ""})
    bundle["assets"] = [asset.model_copy(update={"path": ""}) for asset in bundle["assets"]]
    bundle["jobs"] = [job.model_copy(update={"workflow_snapshot": {}, "input_snapshot": {}, "output_path": None}) for job in bundle["jobs"]]
    bundle["deliveries"] = [
        delivery.model_copy(
            update={
                "output_path": "",
                "preview_path": "available" if delivery.preview_path else None,
                "subtitle_path": None,
            }
        )
        for delivery in bundle["deliveries"]
    ]
    return bundle


@public_router.post("/{token}")
async def submit_public_review(token: str, request: PublicReviewRequest):
    project = next((item for item in store.list_projects() if item.review_token == token), None)
    if project is None:
        raise _not_found("Review link not found")
    if request.decision not in {ApprovalStatus.APPROVED, ApprovalStatus.CHANGES_REQUESTED}:
        raise HTTPException(status_code=400, detail="Review decision must be approved or changes_requested")
    if request.target_type == "shot":
        shot = store.get_shot(request.target_id)
        if shot is None or shot.project_id != project.id:
            raise _not_found("Shot not found")
    elif request.target_type == "storyboard":
        if request.target_id != project.id:
            raise HTTPException(status_code=400, detail="Storyboard target mismatch")
    elif request.target_type == "delivery":
        delivery = next((item for item in store.list_deliveries(project.id) if item.id == request.target_id), None)
        if delivery is None:
            raise _not_found("Delivery not found")
    else:
        raise HTTPException(status_code=400, detail="Unsupported review target")
    review = Review(
        project_id=project.id,
        target_type=request.target_type,
        target_id=request.target_id,
        decision=request.decision,
        comment=request.comment,
        reviewer=request.reviewer,
    )
    store.save_review(review)
    if request.target_type == "shot":
        shot.approval_status = request.decision
        store.save_shot(shot)
    elif request.target_type == "storyboard":
        project.status = ProjectStatus.STORYBOARD_APPROVED if request.decision == ApprovalStatus.APPROVED else ProjectStatus.STORYBOARD_DRAFT
        store.save_project(project)
        for shot in store.list_shots(project.id):
            shot.approval_status = request.decision
            store.save_shot(shot)
    elif request.target_type == "delivery":
        project.status = ProjectStatus.DELIVERED if request.decision == ApprovalStatus.APPROVED else ProjectStatus.FINAL_REVIEW
        store.save_project(project)
    return review


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
