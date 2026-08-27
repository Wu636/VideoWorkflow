from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import (
    Asset,
    AssetRole,
    AssetType,
    GenerationMode,
    JobStatus,
    JobType,
    ProjectStatus,
    RenderJob,
    ShotContinuityMode,
    utc_now,
)
from src.video_workflow.integrations.comfyui import (
    ComfyUIClient,
    H3WorkflowBuilder,
    H3WorkflowRequest,
    h3_diffusion_model,
    h3_frames_for_seconds,
    h3_text_encoder,
    resolve_h3_model_profile,
    resolve_h3_text_encoder_profile,
)
from src.video_workflow.integrations.seedance import (
    SeedanceClient,
    estimate_seedance_cost,
    resolve_seedance_model,
    validate_seedance_resolution,
)
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.audio import DialogueAudioService
from src.video_workflow.storage import ProjectStore

logger = logging.getLogger(__name__)


def _is_transient_seedance_error(exc: BaseException) -> bool:
    """Classify Seedance failures that are worth an automatic resubmission.

    Tunnel jitter can surface either as our pre-submit readability ValueError
    or as Ark's own "timeout while fetching resource" 400; Ark 5xx is also
    transient.  Genuine parameter/prompt errors must fail fast instead.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    text = str(exc)
    return (
        "timeout while fetching resource" in text
        or "素材公网地址不可读取" in text
        or "temporarily unavailable" in text.lower()
        or "service unavailable" in text.lower()
    )


class RenderQueue:
    def __init__(self, store: ProjectStore, projects: ProjectService):
        self.store = store
        self.projects = projects
        self.client = ComfyUIClient()
        self.seedance_client = SeedanceClient()
        self.builder = H3WorkflowBuilder()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def reload_settings(self) -> None:
        """Refresh clients after browser-managed runtime settings change."""
        self.client = ComfyUIClient()
        self.seedance_client = SeedanceClient()
        self.builder = H3WorkflowBuilder()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self.store.recover_interrupted_jobs()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="videoworkflow-h3-render-queue")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    def enqueue(self, project_id: str, shot_ids: list[str] | None = None) -> list[RenderJob]:
        project = self.projects.require_project(project_id)
        shots = self.store.list_shots(project_id)
        if shot_ids:
            selected = set(shot_ids)
            shots = [shot for shot in shots if shot.id in selected]
        jobs: list[RenderJob] = []
        active_shot_ids = {
            job.shot_id
            for job in self.store.list_jobs(project_id)
            if job.status in {JobStatus.QUEUED, JobStatus.SUBMITTING, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
        }
        planned_shots = [
            self.projects.plan_shot(project_id, shot.id)
            for shot in shots
            if shot.id not in active_shot_ids
        ]
        blockers: list[str] = []
        for shot in planned_shots:
            has_keyframe = bool(shot.keyframe_asset_id or shot.image_path)
            if shot.resolved_generation_mode == GenerationMode.I2V and not has_keyframe:
                blockers.append(f"镜头 {shot.ordinal} 缺少完整首帧")
            elif (
                shot.generation_mode == GenerationMode.AUTO
                and shot.resolved_generation_mode == GenerationMode.R2V
                and len(shot.character_ids) > 1
                and not has_keyframe
            ):
                blockers.append(f"镜头 {shot.ordinal} 是多人镜头，但只有独立人物参考图")
        if blockers:
            summary = "；".join(blockers[:8])
            suffix = f"；另有 {len(blockers) - 8} 镜" if len(blockers) > 8 else ""
            raise ValueError(
                f"高质量门禁已阻止提交：{summary}{suffix}。请先在“分镜图 / I2V 首帧”区域生成这些镜头的完整首帧；"
                "若确实要从多素材自由重组，请在单镜参数中明确选择 R2V。"
            )
        for shot in planned_shots:
            model_profile = resolve_h3_model_profile(shot.h3_model_profile)
            text_encoder_profile = resolve_h3_text_encoder_profile(shot.h3_text_encoder_profile)
            h3_parameters = {
                "width": shot.h3_width or project.brief.width,
                "height": shot.h3_height or project.brief.height,
                "frames": shot.render_frames,
                "turbo": shot.h3_turbo,
                "steps": shot.h3_steps,
                "scheduler": shot.h3_scheduler,
                "denoise": shot.h3_denoise,
                "lora_strength": shot.h3_lora_strength,
                "low_vram": shot.h3_low_vram,
                "shift_video": shot.h3_shift_video,
                "shift_audio": shot.h3_shift_audio,
                "model_profile": model_profile,
                "text_encoder_profile": text_encoder_profile,
                # Freeze concrete filenames when the job is queued. Runtime
                # settings may change later, but an existing A/B test must stay
                # reproducible and show exactly which weights were used.
                "diffusion_models": {
                    GenerationMode.I2V.value: h3_diffusion_model(model_profile, GenerationMode.I2V),
                    GenerationMode.R2V.value: h3_diffusion_model(model_profile, GenerationMode.R2V),
                },
                "text_encoder_model": h3_text_encoder(text_encoder_profile),
            }
            quality_guard = {
                "uses_composed_first_frame": shot.resolved_generation_mode == GenerationMode.I2V,
                "keyframe_asset_id": shot.keyframe_asset_id,
                "reference_count": len(shot.reference_asset_ids),
                "prompt_characters": len(shot.video_prompt),
                "audio_postprocess": settings.H3_POSTPROCESS_AUDIO,
                "audio_mode": settings.H3_AUDIO_MODE,
                "issues": self.projects.shot_quality_issues(shot),
            }
            job = RenderJob(
                project_id=project_id,
                shot_id=shot.id,
                type=JobType.VIDEO,
                mode=shot.resolved_generation_mode,
                seed=shot.h3_seed or random.randint(1, 2**31 - 1),
                input_snapshot={
                    "shot_version": shot.version,
                    "storyboard_version": project.storyboard_version,
                    "h3_parameters": h3_parameters,
                    "quality_guard": quality_guard,
                },
            )
            jobs.append(self.store.save_job(job))
            shot.video_status = "queued"
            self.store.save_shot(shot)
        if jobs:
            project.status = ProjectStatus.RENDERING
            self.store.save_project(project)
        return jobs

    def enqueue_seedance(
        self,
        project_id: str,
        shot_ids: list[str] | None = None,
        *,
        model_id: str | None = None,
        resolution: str | None = None,
        generate_audio: bool = True,
    ) -> list[RenderJob]:
        project = self.projects.require_project(project_id)
        model = resolve_seedance_model(model_id)
        resolution = validate_seedance_resolution(model.id, resolution)
        shots = self.store.list_shots(project_id)
        if shot_ids is not None:
            selected = set(shot_ids)
            shots = [shot for shot in shots if shot.id in selected]
        if not shots:
            raise ValueError("请至少选择一个镜头")
        assets = self.store.list_assets(project_id)
        active_shot_ids = {
            job.shot_id
            for job in self.store.list_jobs(project_id)
            if job.status in {JobStatus.QUEUED, JobStatus.SUBMITTING, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
        }
        jobs: list[RenderJob] = []
        candidates = [shot for shot in shots if shot.id not in active_shot_ids]
        reference_map = {
            shot.id: self.projects.seedance_reference_assets(shot, assets)
            for shot in candidates
        }
        blockers = [
            f"镜头 {shot.ordinal} 没有分镜图或参考素材"
            for shot in candidates
            if not reference_map[shot.id]
        ]
        if blockers:
            raise ValueError(
                "Seedance 全模态参考门禁：" + "；".join(blockers[:8]) + "。请先生成分镜首帧或绑定参考素材。"
            )
        for shot in candidates:
            refs = reference_map[shot.id]
            prompt = shot.seedance_prompt.strip() or self.projects.compile_seedance_prompt(project, shot, assets)
            shot.seedance_prompt = prompt
            shot.seedance_prompt_version = "seedance-2.0-official-2026-08"
            estimate = estimate_seedance_cost(
                model.id,
                resolution,
                project.brief.aspect_ratio,
                [shot.duration_seconds],
                has_video_input=any(asset.type == AssetType.VIDEO for asset in refs),
            )
            job = RenderJob(
                project_id=project_id,
                shot_id=shot.id,
                provider="ark_seedance",
                mode=GenerationMode.R2V,
                seed=shot.h3_seed or random.randint(1, 2**31 - 1),
                estimated_cost=float(estimate["estimated_yuan"]),
                max_attempts=3,
                input_snapshot={
                    "shot_version": shot.version,
                    "storyboard_version": project.storyboard_version,
                    "seedance": {
                        "model_id": model.id,
                        "model_label": model.label,
                        "resolution": resolution,
                        "ratio": project.brief.aspect_ratio,
                        "duration": shot.duration_seconds,
                        "generate_audio": generate_audio,
                        "reference_asset_ids": [asset.id for asset in refs],
                        "prompt": prompt,
                        "continuity_mode": shot.continuity_mode.value,
                        "continuity_source_shot_id": shot.continuity_source_shot_id,
                        "estimate": estimate,
                    },
                },
            )
            jobs.append(self.store.save_job(job))
            shot.video_status = "queued"
            self.store.save_shot(shot)
        if jobs:
            project.status = ProjectStatus.RENDERING
            self.store.save_project(project)
        return jobs

    async def cancel(self, job_id: str) -> RenderJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
        elif job.status in {JobStatus.SUBMITTING, JobStatus.RUNNING}:
            job.status = JobStatus.CANCEL_REQUESTED
            if job.prompt_id:
                if job.provider == "ark_seedance":
                    await self.seedance_client.cancel(job.prompt_id)
                else:
                    await self.client.cancel(job.prompt_id)
        self.store.save_job(job)
        if job.shot_id:
            shot = self.store.get_shot(job.shot_id)
            if shot:
                shot.video_status = "cancelled"
                self.store.save_shot(shot)
        return job

    async def _run(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next_job()
            if job is None:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=1.0)
                except TimeoutError:
                    pass
                continue
            try:
                await self._process(job)
            except Exception as exc:
                logger.exception("Render job %s failed", job.id)
                job = self.store.get_job(job.id) or job
                job.error = str(exc)
                if isinstance(exc, httpx.TransportError) and job.status != JobStatus.CANCEL_REQUESTED:
                    # Keep queued work durable while an on-demand GPU instance is off.
                    # Connectivity outages do not consume a model-generation retry.
                    job.attempt = max(0, job.attempt - 1)
                    job.status = JobStatus.QUEUED
                    service = "Seedance API" if job.provider == "ark_seedance" else "ComfyUI"
                    job.error = f"{service} 暂时不可连接，等待重试: {exc}"
                elif (
                    job.status != JobStatus.CANCEL_REQUESTED
                    and job.attempt < job.max_attempts
                    and not (job.provider == "ark_seedance" and not _is_transient_seedance_error(exc))
                ):
                    job.status = JobStatus.QUEUED
                    if job.provider == "ark_seedance":
                        job.error = f"Seedance 素材通道/方舟瞬时故障，自动重试（{job.attempt}/{job.max_attempts}）: {exc}"
                else:
                    job.status = JobStatus.CANCELLED if job.status == JobStatus.CANCEL_REQUESTED else JobStatus.FAILED
                    job.completed_at = utc_now()
                self.store.save_job(job)
                if job.shot_id:
                    shot = self.store.get_shot(job.shot_id)
                    if shot:
                        shot.video_status = "failed" if job.status == JobStatus.FAILED else job.status.value
                        self.store.save_shot(shot)
                if isinstance(exc, httpx.TransportError) or (
                    job.status == JobStatus.QUEUED and job.provider == "ark_seedance"
                ):
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=15.0)
                    except TimeoutError:
                        pass

    async def _process(self, job: RenderJob) -> None:
        if job.provider == "ark_seedance":
            await self._process_seedance(job)
            return
        if not job.shot_id:
            raise ValueError("Video render job has no shot_id")
        project = self.projects.require_project(job.project_id)
        shot = self.projects.require_shot(job.shot_id)
        h3_parameters = job.input_snapshot.get("h3_parameters", {})
        assets = self.store.list_assets(job.project_id)
        segment_plan = self.projects.h3_segment_plan(project, shot, assets)
        job.input_snapshot["segment_plan"] = segment_plan
        job.input_snapshot.setdefault("quality_guard", {})["segment_count"] = len(segment_plan)

        diffusion_models = h3_parameters.get("diffusion_models")
        if not isinstance(diffusion_models, dict):
            # Backward compatibility for jobs queued before model profiles were
            # introduced.  Resolve once and persist the concrete choice now.
            model_profile = resolve_h3_model_profile(str(h3_parameters.get("model_profile") or shot.h3_model_profile))
            diffusion_models = {
                GenerationMode.I2V.value: h3_diffusion_model(model_profile, GenerationMode.I2V),
                GenerationMode.R2V.value: h3_diffusion_model(model_profile, GenerationMode.R2V),
            }
            h3_parameters["model_profile"] = model_profile
            h3_parameters["diffusion_models"] = diffusion_models
        text_encoder_profile = resolve_h3_text_encoder_profile(
            str(h3_parameters.get("text_encoder_profile") or shot.h3_text_encoder_profile)
        )
        text_encoder_model = str(h3_parameters.get("text_encoder_model") or h3_text_encoder(text_encoder_profile))
        h3_parameters["text_encoder_profile"] = text_encoder_profile
        h3_parameters["text_encoder_model"] = text_encoder_model

        required_modes: set[GenerationMode] = set()
        for segment_index, segment in enumerate(segment_plan):
            if str(segment.get("dialogue") or "").strip():
                required_modes.add(GenerationMode.R2V)
            elif segment_index == 0:
                required_modes.add(job.mode or GenerationMode.I2V)
            else:
                required_modes.add(GenerationMode.I2V)
        selected_models = [text_encoder_model]
        for mode in required_modes:
            selected = diffusion_models.get(mode.value)
            if not isinstance(selected, str) or not selected:
                raise RuntimeError(f"任务快照缺少 {mode.value.upper()} diffusion 模型文件名")
            selected_models.append(selected)
        model_check = await self.client.preflight(
            self.builder.preflight_requirements(
                selected_models,
                require_turbo=bool(h3_parameters.get("turbo", shot.h3_turbo)),
            )
        )
        if not model_check["ok"]:
            missing = [*model_check.get("missing_nodes", []), *model_check.get("missing_models", [])]
            raise RuntimeError(
                "所选 H3 模型/节点尚未安装，已在上传素材和占用 GPU 前停止：" + "、".join(missing)
            )
        job.input_snapshot["model_preflight"] = {
            "required_models": model_check["required_models"],
            "model_profile": h3_parameters.get("model_profile"),
            "text_encoder_profile": h3_parameters.get("text_encoder_profile"),
            "checked_at": utc_now(),
        }
        self.store.save_job(job)

        required = self._required_assets(shot, assets, job.mode or GenerationMode.I2V)
        uploaded = await self.client.upload_assets(required, f"{job.project_id}/{job.id}")
        image_files = [uploaded[a.id] for a in required if a.type == AssetType.IMAGE]
        video_files = [uploaded[a.id] for a in required if a.type == AssetType.VIDEO]
        audio_files = [uploaded[a.id] for a in required if a.type == AssetType.AUDIO]

        if job.mode == GenerationMode.R2V and not image_files and shot.image_path:
            image_path = resolve_media_path(shot.image_path)
            image_files.append(
                await self.client.upload_input(
                    image_path,
                    f"videoworkflow/{job.project_id}/{job.id}/composition{image_path.suffix}",
                )
            )

        first_frame = None
        last_frame = None
        if job.mode == GenerationMode.I2V:
            if shot.keyframe_asset_id:
                first_frame = uploaded.get(shot.keyframe_asset_id)
            if not first_frame and shot.image_path:
                image_path = resolve_media_path(shot.image_path)
                first_frame = await self.client.upload_input(
                    image_path,
                    f"videoworkflow/{job.project_id}/{job.id}/first{image_path.suffix}",
                )
            if shot.last_frame_asset_id:
                last_frame = uploaded.get(shot.last_frame_asset_id)

        output_dir = self.projects.project_dir(job.project_id) / "videos" / shot.id
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{job.id}.mp4"
        dialogue_guides: dict[int, dict[str, object]] = {}
        expected_dialogue_tracks = sum(bool(str(segment.get("dialogue") or "").strip()) for segment in segment_plan)
        if settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE == "clean_tts":
            for segment_index, segment in enumerate(segment_plan):
                dialogue = str(segment.get("dialogue") or "").strip()
                if not dialogue:
                    continue
                segment_shot = shot.model_copy(deep=True)
                segment_shot.dialogue = dialogue
                segment_shot.dialogue_speaker_id = segment.get("speaker_id") or shot.dialogue_speaker_id
                segment_shot.dialogue_turns = []
                raw_speech = output_dir / f"{job.id}.dialogue{segment_index + 1}.mp3"
                delivery_guide = output_dir / f"{job.id}.dialogue{segment_index + 1}.delivery.wav"
                h3_guide = output_dir / f"{job.id}.dialogue{segment_index + 1}.h3.wav"
                metadata = await DialogueAudioService.synthesize(project, segment_shot, raw_speech)
                if not metadata.get("generated"):
                    raise RuntimeError(f"第 {segment_index + 1} 段干净对白生成失败：{metadata.get('reason') or '未知原因'}")
                delivery_duration = float(segment["duration"])
                generation_duration = float(segment.get("generation_duration", delivery_duration))
                local_offset = min(shot.dialogue_start_seconds, max(0.05, delivery_duration * 0.2))
                await asyncio.to_thread(
                    self._prepare_dialogue_guide,
                    raw_speech,
                    delivery_guide,
                    delivery_duration,
                    local_offset,
                )
                await asyncio.to_thread(
                    self._pad_dialogue_guide,
                    delivery_guide,
                    h3_guide,
                    generation_duration,
                )
                raw_speech.unlink(missing_ok=True)
                remote_audio = await self.client.upload_input(
                    h3_guide,
                    f"videoworkflow/{job.project_id}/{job.id}/dialogue{segment_index + 1}.wav",
                )
                metadata.update(
                    {
                        "segment": segment_index + 1,
                        "speaker_id": segment_shot.dialogue_speaker_id,
                        "dialogue_start_seconds": local_offset,
                        "delivery_guide": str(delivery_guide),
                        "h3_reference_guide": str(h3_guide),
                    }
                )
                dialogue_guides[segment_index] = {
                    "delivery_path": delivery_guide,
                    "h3_path": h3_guide,
                    "remote": remote_audio,
                    "metadata": metadata,
                }
            if len(dialogue_guides) != expected_dialogue_tracks:
                raise RuntimeError(
                    f"独立对白生成不完整：需要 {expected_dialogue_tracks} 段，成功 {len(dialogue_guides)} 段；"
                    "已在占用 GPU 前阻止任务"
                )
            job.input_snapshot["dialogue_guides"] = [
                guide["metadata"] for guide in dialogue_guides.values()
            ]
            self.store.save_job(job)
        segment_paths: list[Path] = []
        workflows: list[dict] = []
        continuation_first_frame = first_frame
        started = time.monotonic()
        try:
            for segment_index, segment in enumerate(segment_plan):
                dialogue_guide = dialogue_guides.get(segment_index)
                segment_mode = (
                    GenerationMode.R2V
                    if dialogue_guide
                    else ((job.mode or GenerationMode.I2V) if segment_index == 0 else GenerationMode.I2V)
                )
                segment_reference_images = image_files
                if segment_mode == GenerationMode.R2V and segment_index > 0 and continuation_first_frame:
                    segment_reference_images = list(dict.fromkeys([continuation_first_frame, *image_files]))
                segment_reference_audios = audio_files
                if dialogue_guide:
                    segment_reference_audios = [str(dialogue_guide["remote"]), *audio_files]
                request = H3WorkflowRequest(
                    mode=segment_mode,
                    prompt=str(segment["prompt"]),
                    width=int(h3_parameters.get("width", shot.h3_width or project.brief.width)),
                    height=int(h3_parameters.get("height", shot.h3_height or project.brief.height)),
                    frames=h3_frames_for_seconds(float(segment.get("generation_duration", segment["duration"]))),
                    # A genuine model/QC retry must explore a new sample instead
                    # of deterministically recreating the same bad clip.
                    seed=job.seed + max(0, job.attempt - 1) * 1009 + segment_index,
                    output_prefix=f"video/videoworkflow/{job.project_id}/{shot.ordinal}_{job.id}_part{segment_index + 1}",
                    first_frame_filename=continuation_first_frame if segment_mode == GenerationMode.I2V else None,
                    last_frame_filename=last_frame if segment_mode == GenerationMode.I2V and segment_index == len(segment_plan) - 1 else None,
                    reference_images=segment_reference_images if segment_mode == GenerationMode.R2V else [],
                    reference_videos=video_files if segment_mode == GenerationMode.R2V else [],
                    reference_audios=segment_reference_audios if segment_mode == GenerationMode.R2V else [],
                    ref_image_size=shot.ref_image_size,
                    turbo=bool(h3_parameters.get("turbo", shot.h3_turbo)),
                    steps=int(h3_parameters.get("steps", shot.h3_steps)),
                    scheduler=str(h3_parameters.get("scheduler", shot.h3_scheduler)),
                    denoise=float(h3_parameters.get("denoise", shot.h3_denoise)),
                    lora_strength=float(h3_parameters.get("lora_strength", shot.h3_lora_strength)),
                    low_vram=bool(h3_parameters.get("low_vram", shot.h3_low_vram)),
                    shift_video=float(h3_parameters.get("shift_video", shot.h3_shift_video)),
                    shift_audio=float(h3_parameters.get("shift_audio", shot.h3_shift_audio)),
                    diffusion_model=str(diffusion_models[segment_mode.value]),
                    text_encoder_model=text_encoder_model,
                )
                workflow = self.builder.build(request)
                workflows.append(workflow)
                prompt_id = await self.client.submit(workflow)
                job.prompt_id = prompt_id
                job.status = JobStatus.RUNNING
                job.workflow_snapshot = {"segments": workflows} if len(segment_plan) > 1 else workflow
                self.store.save_job(job)

                async def progress(value: float, message: str, *, active_prompt_id: str = prompt_id, index: int = segment_index) -> None:
                    current = self.store.get_job(job.id)
                    if current is None:
                        return
                    if current.status == JobStatus.CANCEL_REQUESTED:
                        await self.client.cancel(active_prompt_id)
                        raise asyncio.CancelledError()
                    current.progress = min(0.99, (index + value) / len(segment_plan))
                    if message.startswith("queued:"):
                        try:
                            current.queue_position = int(message.split(":", 1)[1])
                        except ValueError:
                            current.queue_position = None
                        current.error = f"高质量续帧 {index + 1}/{len(segment_plan)} · 排队中"
                    elif message in {"queued", "running", "completed"}:
                        current.queue_position = None if message != "queued" else current.queue_position
                        current.error = f"高质量续帧 {index + 1}/{len(segment_plan)} · {message}"
                    else:
                        current.error = message
                    self.store.save_job(current)

                history = await self.client.wait_for_result(prompt_id, progress)
                output = self.client.extract_output(history)
                segment_path = output_dir / f"{job.id}.part{segment_index + 1}{Path(output['filename']).suffix or '.mp4'}"
                await self.client.download_output(output, segment_path)
                segment_paths.append(segment_path)

                if segment_index < len(segment_plan) - 1:
                    continuation_path = output_dir / f"{job.id}.continuation{segment_index + 1}.png"
                    await asyncio.to_thread(self._extract_last_frame, segment_path, continuation_path)
                    continuation_first_frame = await self.client.upload_input(
                        continuation_path,
                        f"videoworkflow/{job.project_id}/{job.id}/continuation{segment_index + 1}.png",
                    )
                    continuation_path.unlink(missing_ok=True)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = utc_now()
            self.store.save_job(job)
            return

        if len(segment_paths) == 1 and abs(
            float(segment_plan[0].get("generation_duration", segment_plan[0]["duration"]))
            - float(segment_plan[0]["duration"])
        ) <= 0.05:
            segment_paths[0].replace(destination)
        else:
            await asyncio.to_thread(
                self._concat_video_segments,
                segment_paths,
                [float(segment["duration"]) for segment in segment_plan],
                destination,
            )
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)
        speech_tracks: list[dict[str, object]] = []
        speech_metadata: dict[str, object] = {"generated": False, "reason": "audio_mode_not_clean_tts", "tracks": []}
        if settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE == "clean_tts":
            elapsed_delivery = 0.0
            metadata_tracks: list[dict[str, object]] = []
            for segment_index, segment in enumerate(segment_plan):
                segment_duration = float(segment["duration"])
                guide = dialogue_guides.get(segment_index)
                if guide:
                    metadata_tracks.append(dict(guide["metadata"]))
                    speech_tracks.append(
                        {
                            "path": guide["delivery_path"],
                            "start_seconds": elapsed_delivery,
                            "available_seconds": segment_duration,
                            "speaker_id": segment.get("speaker_id") or shot.dialogue_speaker_id,
                        }
                    )
                elapsed_delivery += segment_duration
            speech_metadata = {
                "generated": bool(speech_tracks),
                "provider": settings.TTS_PROVIDER,
                "track_count": len(speech_tracks),
                "expected_track_count": expected_dialogue_tracks,
                "tracks": metadata_tracks,
            }
            if len(speech_tracks) != expected_dialogue_tracks:
                for track in speech_tracks:
                    Path(str(track["path"])).unlink(missing_ok=True)
                job.input_snapshot["dialogue_tts"] = speech_metadata
                self.store.save_job(job)
                raise RuntimeError(
                    f"独立对白生成不完整：需要 {expected_dialogue_tracks} 段，成功 {len(speech_tracks)} 段；"
                    "已阻止静音或缺台词视频进入交付"
                )
        audio_delivery = await asyncio.to_thread(
            self._replace_generated_audio,
            destination,
            speech_tracks,
            shot.dialogue_start_seconds,
        )
        for track in speech_tracks:
            Path(str(track["path"])).unlink(missing_ok=True)
        for guide in dialogue_guides.values():
            Path(str(guide["h3_path"])).unlink(missing_ok=True)
        media_qc = await asyncio.to_thread(
            self._validate_rendered_media,
            destination,
            int(h3_parameters.get("width", shot.h3_width or project.brief.width)),
            int(h3_parameters.get("height", shot.h3_height or project.brief.height)),
            float(shot.duration_seconds),
            settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE != "native",
        )
        job.input_snapshot["media_qc"] = media_qc
        if not media_qc["passed"]:
            self.store.save_job(job)
            raise RuntimeError(f"成片媒体质量门禁未通过：{'；'.join(media_qc['issues'])}")
        elapsed = time.monotonic() - started
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.queue_position = None
        job.output_path = str(destination.resolve())
        job.elapsed_seconds = elapsed
        job.estimated_cost = elapsed / 3600 * settings.COMFYUI_HOURLY_RATE
        job.completed_at = utc_now()
        job.error = None
        job.input_snapshot["dialogue_tts"] = speech_metadata
        job.input_snapshot["audio_delivery"] = audio_delivery
        self.store.save_job(job)

        shot.video_path = job.output_path
        shot.video_status = "completed"
        self.store.save_shot(shot)
        all_shots = self.store.list_shots(job.project_id)
        if all(item.video_status == "completed" for item in all_shots):
            project.status = ProjectStatus.CLIPS_REVIEW
            self.store.save_project(project)

    async def _process_seedance(self, job: RenderJob) -> None:
        if not job.shot_id:
            raise ValueError("Seedance 视频任务缺少 shot_id")
        project = self.projects.require_project(job.project_id)
        shot = self.projects.require_shot(job.shot_id)
        config = job.input_snapshot.get("seedance")
        if not isinstance(config, dict):
            raise ValueError("Seedance 任务快照缺少提交参数")
        asset_ids = [str(value) for value in config.get("reference_asset_ids", [])]
        asset_map = {asset.id: asset for asset in self.store.list_assets(job.project_id)}
        refs = [asset_map[asset_id] for asset_id in asset_ids if asset_id in asset_map]
        if len(refs) != len(asset_ids):
            raise ValueError("Seedance 任务引用的素材已被删除，请重新编译并提交")

        # Continuous mode resolves the previous shot at execution time. This
        # lets a batch queue shot N+1 before shot N has produced its tail frame.
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS:
            source = self.projects.continuity_source_shot(project, shot)
            if source is None:
                raise ValueError(f"镜头 {shot.ordinal} 已选择连续续接，但没有可用的上一镜")
            if not source.last_frame_asset_id:
                raise ValueError(
                    f"镜头 {shot.ordinal} 等待镜头 {source.ordinal} 的尾帧；请先完成上一镜再提交本镜"
                )
            assets = self.store.list_assets(job.project_id)
            refs = self.projects.seedance_reference_assets(shot, assets, project)
            config["reference_asset_ids"] = [asset.id for asset in refs]
            config["prompt"] = self.projects.compile_seedance_prompt(project, shot, assets)
            job.input_snapshot["seedance"] = config
            shot.seedance_prompt = str(config["prompt"])
            shot.seedance_prompt_source_revision = shot.content_revision
            self.store.save_shot(shot)

        first_frame_asset_id = self.projects.seedance_first_frame_asset_id(
            project,
            shot,
            self.store.list_assets(job.project_id),
        )

        started = time.monotonic()
        job.status = JobStatus.SUBMITTING
        job.progress = 0.08
        job.error = "正在上传参考素材并创建 Seedance 任务"
        self.store.save_job(job)
        task_id, request_snapshot = await self.seedance_client.submit(
            model_id=str(config["model_id"]),
            prompt=str(config["prompt"]),
            assets=refs,
            ratio=str(config.get("ratio") or project.brief.aspect_ratio),
            duration=float(config.get("duration") or shot.duration_seconds),
            resolution=str(config["resolution"]),
            seed=job.seed,
            generate_audio=bool(config.get("generate_audio", True)),
            first_frame_asset_id=first_frame_asset_id,
        )
        job.prompt_id = task_id
        job.status = JobStatus.RUNNING
        job.progress = 0.2
        job.workflow_snapshot = request_snapshot
        job.error = "Seedance 已受理，正在生成"
        self.store.save_job(job)

        deadline = started + max(60, settings.SEEDANCE_JOB_TIMEOUT_SECONDS)
        result: dict[str, object] = {}
        while True:
            current = self.store.get_job(job.id) or job
            if current.status == JobStatus.CANCEL_REQUESTED:
                await self.seedance_client.cancel(task_id)
                current.status = JobStatus.CANCELLED
                current.completed_at = utc_now()
                current.error = "任务已取消"
                self.store.save_job(current)
                shot.video_status = "cancelled"
                self.store.save_shot(shot)
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"Seedance 任务超过 {settings.SEEDANCE_JOB_TIMEOUT_SECONDS} 秒仍未完成")
            await asyncio.sleep(max(2.0, settings.SEEDANCE_POLL_INTERVAL_SECONDS))
            result = await self.seedance_client.get(task_id)
            status = str(result.get("status") or "").lower()
            if status == "succeeded":
                break
            if status in {"failed", "expired", "cancelled", "canceled"}:
                error = result.get("error") or result.get("failure_reason") or status
                raise RuntimeError(f"Seedance 生成失败: {error}")
            current.progress = 0.55 if status == "running" else 0.3
            current.error = f"Seedance {status or 'queued'} · 任务 {task_id}"
            self.store.save_job(current)

        destination = self.projects.project_dir(job.project_id) / "videos" / shot.id / f"{job.id}.mp4"
        await self.seedance_client.download(self.seedance_client.video_url(result), destination)
        last_frame_url = self.seedance_client.last_frame_url(result)
        if last_frame_url:
            last_frame_path = (
                self.projects.project_dir(job.project_id)
                / "images"
                / "last_frames"
                / f"{shot.id}-{job.id}.png"
            )
            await self.seedance_client.download(last_frame_url, last_frame_path)
            last_frame_asset = self.projects.register_existing_asset(
                job.project_id,
                last_frame_path,
                role=AssetRole.LAST_FRAME,
                name=f"镜头 {shot.ordinal} · Seedance 尾帧",
                description="Seedance return_last_frame 产物，可作为下一镜连续续接的严格首帧",
            )
            shot.last_frame_asset_id = last_frame_asset.id
        elapsed = time.monotonic() - started
        job = self.store.get_job(job.id) or job
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.output_path = str(destination.resolve())
        job.elapsed_seconds = elapsed
        job.completed_at = utc_now()
        job.error = None
        job.input_snapshot["seedance_result"] = {
            "usage": result.get("usage", {}),
            "last_frame_url": last_frame_url,
            "last_frame_asset_id": shot.last_frame_asset_id,
            "completed_at": utc_now(),
        }
        self.store.save_job(job)
        shot.video_path = job.output_path
        shot.video_status = "completed"
        self.store.save_shot(shot)
        if all(item.video_status == "completed" for item in self.store.list_shots(job.project_id)):
            project.status = ProjectStatus.CLIPS_REVIEW
            self.store.save_project(project)

    @staticmethod
    def _required_assets(shot, assets: list[Asset], mode: GenerationMode) -> list[Asset]:
        asset_map = {asset.id: asset for asset in assets}
        ids: list[str] = []
        if mode == GenerationMode.I2V:
            ids.extend(asset_id for asset_id in [shot.keyframe_asset_id, shot.last_frame_asset_id] if asset_id)
        else:
            # Picture 1 is always the approved composition anchor. Character,
            # style, video, and audio references follow in authored order.
            ids.extend(asset_id for asset_id in [shot.keyframe_asset_id, *shot.reference_asset_ids] if asset_id)
        result = [asset_map[asset_id] for asset_id in dict.fromkeys(ids) if asset_id in asset_map]
        if mode == GenerationMode.I2V and not result and not shot.image_path:
            raise ValueError(f"Shot {shot.ordinal} needs an approved first frame before I2V rendering")
        if mode == GenerationMode.R2V and not result and not shot.image_path:
            raise ValueError(f"Shot {shot.ordinal} needs reference media before R2V rendering")
        return result

    @staticmethod
    def _extract_last_frame(source: Path, destination: Path) -> None:
        if not shutil.which(settings.FFMPEG_BIN):
            raise RuntimeError("ffmpeg is required for sequential H3 continuation")
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            settings.FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            "-0.08",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(result.stderr[-1500:] or "Could not extract continuation frame")

    @staticmethod
    def _concat_video_segments(sources: list[Path], durations: list[float], destination: Path) -> None:
        if len(sources) != len(durations) or not sources:
            raise ValueError("Sequential segment sources/durations do not match")
        command = [settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"]
        for source in sources:
            command.extend(["-i", str(source)])
        filters = []
        labels = []
        for index, duration in enumerate(durations):
            label = f"v{index}"
            safe_duration = max(0.25, duration)
            filters.append(
                f"[{index}:v]trim=duration={safe_duration:.6f},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={safe_duration:.6f},"
                f"trim=duration={safe_duration:.6f},setpts=PTS-STARTPTS[{label}]"
            )
            labels.append(f"[{label}]")
        filters.append(f"{''.join(labels)}concat=n={len(sources)}:v=1:a=0[outv]")
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[outv]",
                "-an",
                "-c:v",
                settings.FINAL_VIDEO_CODEC,
                "-preset",
                settings.FINAL_VIDEO_PRESET,
                "-crf",
                str(settings.FINAL_VIDEO_CRF),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(result.stderr[-2000:] or "Could not concatenate sequential H3 segments")

    @staticmethod
    def _media_duration(path: Path) -> float:
        probe = subprocess.run(
            [
                settings.FFPROBE_BIN,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr[-1000:] or "ffprobe failed")
        return max(0.05, float(json.loads(probe.stdout or "{}").get("format", {}).get("duration") or 0))

    @classmethod
    def _prepare_dialogue_guide(
        cls,
        source: Path,
        destination: Path,
        total_duration: float,
        start_seconds: float,
    ) -> None:
        """Create the exact timed dialogue stem used by both H3 and delivery."""
        duration = max(0.3, float(total_duration))
        start = max(0.0, min(float(start_seconds), duration - 0.2))
        available = max(0.25, duration - start - 0.05)
        speech_duration = cls._media_duration(source)
        tempo = max(1.0, speech_duration / available)
        tempo_filters: list[str] = []
        while tempo > 2.0:
            tempo_filters.append("atempo=2.0")
            tempo /= 2.0
        if tempo > 1.005:
            tempo_filters.append(f"atempo={tempo:.5f}")
        processing = [
            f"aresample={settings.FINAL_AUDIO_SAMPLE_RATE}",
            *tempo_filters,
            "highpass=f=60",
            "lowpass=f=15500",
            "loudnorm=I=-20:TP=-3:LRA=7",
            "asetpts=N/SR/TB",
            f"atrim=0:{available:.6f}",
            f"adelay={round(start * 1000)}:all=1",
            "asetpts=N/SR/TB",
            f"apad=pad_dur={duration:.6f}",
            f"atrim=0:{duration:.6f}",
        ]
        command = [
            settings.FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            ",".join(processing),
            "-ar",
            str(settings.FINAL_AUDIO_SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(result.stderr[-1500:] or "Could not prepare timed dialogue guide")

    @staticmethod
    def _pad_dialogue_guide(source: Path, destination: Path, total_duration: float) -> None:
        duration = max(2.0, min(15.0, float(total_duration)))
        command = [
            settings.FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            f"apad=pad_dur={duration:.6f},atrim=0:{duration:.6f}",
            "-ar",
            str(settings.FINAL_AUDIO_SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(result.stderr[-1500:] or "Could not pad H3 dialogue reference")

    @classmethod
    def _replace_generated_audio(
        cls,
        path: Path,
        speech_path: Path | list[dict[str, object]] | None,
        dialogue_start_seconds: float,
    ) -> dict[str, object]:
        """Copy H3 pixels and replace its broadband synthetic noise track."""
        if not settings.H3_POSTPROCESS_AUDIO or settings.H3_AUDIO_MODE == "native":
            return {"applied": False, "mode": "native", "reason": "native_audio_selected"}
        if not shutil.which(settings.FFMPEG_BIN) or not shutil.which(settings.FFPROBE_BIN):
            return {"applied": False, "mode": settings.H3_AUDIO_MODE, "reason": "ffmpeg_missing"}

        try:
            duration = cls._media_duration(path)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Audio replacement probe failed for %s: %s", path, exc)
            return {"applied": False, "mode": settings.H3_AUDIO_MODE, "reason": "probe_failed"}

        target = path.with_name(f"{path.stem}.clean-audio{path.suffix}")
        common_output = [
            "-c:v",
            "copy",
            "-ar",
            str(settings.FINAL_AUDIO_SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            settings.FINAL_AUDIO_CODEC,
            "-b:a",
            settings.H3_POSTPROCESS_AUDIO_BITRATE,
            "-movflags",
            "+faststart",
            "-shortest",
            str(target),
        ]
        command = [settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path)]
        audio_map: list[str]

        delivered = "silence"
        tracks: list[dict[str, object]] = []
        if isinstance(speech_path, Path):
            if speech_path.exists():
                tracks = [{
                    "path": speech_path,
                    "start_seconds": dialogue_start_seconds,
                    "available_seconds": max(0.5, duration - dialogue_start_seconds - 0.1),
                }]
        elif isinstance(speech_path, list):
            tracks = [
                track for track in speech_path
                if Path(str(track.get("path") or "")).exists()
            ]

        if settings.H3_AUDIO_MODE == "clean_tts" and tracks:
            filters: list[str] = []
            voice_labels: list[str] = []
            for index, track in enumerate(tracks, start=1):
                track_path = Path(str(track["path"]))
                start_seconds = max(0.0, float(track.get("start_seconds") or 0.0))
                available = max(0.3, min(duration - start_seconds, float(track.get("available_seconds") or duration)))
                try:
                    speech_duration = cls._media_duration(track_path)
                except (RuntimeError, ValueError, json.JSONDecodeError):
                    speech_duration = available
                tempo = max(1.0, speech_duration / available)
                tempo_filters: list[str] = []
                while tempo > 2.0:
                    tempo_filters.append("atempo=2.0")
                    tempo /= 2.0
                if tempo > 1.005:
                    tempo_filters.append(f"atempo={tempo:.5f}")
                label = f"voice{index}"
                filters.append(
                    f"[{index}:a]aresample={settings.FINAL_AUDIO_SAMPLE_RATE},"
                    f"{','.join([*tempo_filters, 'highpass=f=60', 'lowpass=f=15500', 'loudnorm=I=-20:TP=-3:LRA=7'])},"
                    f"atrim=0:{available:.6f},adelay={round(start_seconds * 1000)}:all=1[{label}]"
                )
                voice_labels.append(f"[{label}]")
                command.extend(["-i", str(track_path)])
            if len(voice_labels) == 1:
                filters.append(f"{voice_labels[0]}apad,atrim=0:{duration:.6f}[clean]")
            else:
                filters.append(
                    f"{''.join(voice_labels)}amix=inputs={len(voice_labels)}:duration=longest:dropout_transition=0:normalize=0,"
                    f"alimiter=limit=0.95,apad,atrim=0:{duration:.6f}[clean]"
                )
            command.extend(["-filter_complex", ";".join(filters)])
            audio_map = ["-map", "[clean]"]
            delivered = "independent_tts" if len(tracks) == 1 else "independent_multi_speaker_tts"
        else:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{duration:.6f}",
                    "-i",
                    f"anullsrc=channel_layout=stereo:sample_rate={settings.FINAL_AUDIO_SAMPLE_RATE}",
                ]
            )
            audio_map = ["-map", "1:a:0"]
        command.extend(["-map", "0:v:0", *audio_map, *common_output])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            logger.warning("Audio replacement failed for %s: %s", path, result.stderr[-1500:])
            return {"applied": False, "mode": settings.H3_AUDIO_MODE, "reason": "ffmpeg_failed"}
        target.replace(path)
        return {
            "applied": True,
            "mode": settings.H3_AUDIO_MODE,
            "delivered": delivered,
            "sample_rate": settings.FINAL_AUDIO_SAMPLE_RATE,
            "bitrate": settings.H3_POSTPROCESS_AUDIO_BITRATE,
            "video_stream": "copied_without_reencoding",
            "native_h3_audio": "discarded",
            "speech_tracks": len(tracks),
        }

    @staticmethod
    def _validate_rendered_media(
        path: Path,
        expected_width: int,
        expected_height: int,
        expected_duration: float,
        require_clean_audio: bool,
    ) -> dict[str, object]:
        """Reject broken media before a shot can be marked completed."""
        probe = subprocess.run(
            [
                settings.FFPROBE_BIN,
                "-v", "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        issues: list[str] = []
        payload: dict[str, object] = {}
        if probe.returncode != 0:
            issues.append(f"ffprobe 读取失败: {probe.stderr[-300:]}")
        else:
            try:
                payload = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError:
                issues.append("ffprobe 返回了无效 JSON")
        streams = payload.get("streams", []) if isinstance(payload, dict) else []
        streams = streams if isinstance(streams, list) else []
        video = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), {})
        audio = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"), {})
        actual_width = int(video.get("width") or 0)
        actual_height = int(video.get("height") or 0)
        if (actual_width, actual_height) != (expected_width, expected_height):
            issues.append(f"分辨率异常 {actual_width}x{actual_height}，预期 {expected_width}x{expected_height}")
        format_payload = payload.get("format", {}) if isinstance(payload, dict) else {}
        duration = float(format_payload.get("duration") or 0.0) if isinstance(format_payload, dict) else 0.0
        duration_tolerance = max(0.75, expected_duration * 0.08)
        if duration <= 0 or abs(duration - expected_duration) > duration_tolerance:
            issues.append(f"时长异常 {duration:.3f}s，预期约 {expected_duration:.3f}s")
        frame_count = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
        if frame_count <= 1:
            issues.append("视频帧不足或不可读")
        sample_rate = int(audio.get("sample_rate") or 0) if audio else 0
        if require_clean_audio and not audio:
            issues.append("缺少干净音频轨")
        elif require_clean_audio and sample_rate != settings.FINAL_AUDIO_SAMPLE_RATE:
            issues.append(f"音频采样率异常 {sample_rate}Hz，预期 {settings.FINAL_AUDIO_SAMPLE_RATE}Hz")

        decode = subprocess.run(
            [settings.FFMPEG_BIN, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        if decode.returncode != 0:
            issues.append(f"完整解码失败: {decode.stderr[-300:]}")
        return {
            "passed": not issues,
            "issues": issues,
            "width": actual_width,
            "height": actual_height,
            "duration_seconds": duration,
            "frames": frame_count,
            "has_audio": bool(audio),
            "audio_sample_rate": sample_rate,
            "decode_passed": decode.returncode == 0,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
