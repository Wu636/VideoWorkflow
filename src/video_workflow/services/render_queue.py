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
    AssetType,
    GenerationMode,
    JobStatus,
    JobType,
    ProjectStatus,
    RenderJob,
    utc_now,
)
from src.video_workflow.integrations.comfyui import ComfyUIClient, H3WorkflowBuilder, H3WorkflowRequest, h3_frames_for_seconds
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.audio import DialogueAudioService
from src.video_workflow.storage import ProjectStore

logger = logging.getLogger(__name__)


class RenderQueue:
    def __init__(self, store: ProjectStore, projects: ProjectService):
        self.store = store
        self.projects = projects
        self.client = ComfyUIClient()
        self.builder = H3WorkflowBuilder()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def reload_settings(self) -> None:
        """Refresh clients after browser-managed runtime settings change."""
        self.client = ComfyUIClient()
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

    async def cancel(self, job_id: str) -> RenderJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
        elif job.status in {JobStatus.SUBMITTING, JobStatus.RUNNING}:
            job.status = JobStatus.CANCEL_REQUESTED
            if job.prompt_id:
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
                    job.error = f"ComfyUI offline; waiting to retry: {exc}"
                elif job.attempt < job.max_attempts and job.status != JobStatus.CANCEL_REQUESTED:
                    job.status = JobStatus.QUEUED
                else:
                    job.status = JobStatus.CANCELLED if job.status == JobStatus.CANCEL_REQUESTED else JobStatus.FAILED
                    job.completed_at = utc_now()
                self.store.save_job(job)
                if job.shot_id:
                    shot = self.store.get_shot(job.shot_id)
                    if shot:
                        shot.video_status = "failed" if job.status == JobStatus.FAILED else job.status.value
                        self.store.save_shot(shot)
                if isinstance(exc, httpx.TransportError):
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=15.0)
                    except TimeoutError:
                        pass

    async def _process(self, job: RenderJob) -> None:
        if not job.shot_id:
            raise ValueError("Video render job has no shot_id")
        project = self.projects.require_project(job.project_id)
        shot = self.projects.require_shot(job.shot_id)
        h3_parameters = job.input_snapshot.get("h3_parameters", {})
        assets = self.store.list_assets(job.project_id)
        required = self._required_assets(shot, assets, job.mode or GenerationMode.I2V)
        uploaded = await self.client.upload_assets(required, f"{job.project_id}/{job.id}")
        image_files = [uploaded[a.id] for a in required if a.type == AssetType.IMAGE]
        video_files = [uploaded[a.id] for a in required if a.type == AssetType.VIDEO]
        audio_files = [uploaded[a.id] for a in required if a.type == AssetType.AUDIO]

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

        segment_plan = self.projects.h3_segment_plan(project, shot, assets)
        job.input_snapshot["segment_plan"] = segment_plan
        job.input_snapshot["quality_guard"]["segment_count"] = len(segment_plan)
        output_dir = self.projects.project_dir(job.project_id) / "videos" / shot.id
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{job.id}.mp4"
        segment_paths: list[Path] = []
        workflows: list[dict] = []
        continuation_first_frame = first_frame
        started = time.monotonic()
        try:
            for segment_index, segment in enumerate(segment_plan):
                segment_mode = (job.mode or GenerationMode.I2V) if segment_index == 0 else GenerationMode.I2V
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
                    reference_images=image_files if segment_mode == GenerationMode.R2V else [],
                    reference_videos=video_files if segment_mode == GenerationMode.R2V else [],
                    reference_audios=audio_files if segment_mode == GenerationMode.R2V else [],
                    ref_image_size=shot.ref_image_size,
                    turbo=bool(h3_parameters.get("turbo", shot.h3_turbo)),
                    steps=int(h3_parameters.get("steps", shot.h3_steps)),
                    scheduler=str(h3_parameters.get("scheduler", shot.h3_scheduler)),
                    denoise=float(h3_parameters.get("denoise", shot.h3_denoise)),
                    lora_strength=float(h3_parameters.get("lora_strength", shot.h3_lora_strength)),
                    low_vram=bool(h3_parameters.get("low_vram", shot.h3_low_vram)),
                    shift_video=float(h3_parameters.get("shift_video", shot.h3_shift_video)),
                    shift_audio=float(h3_parameters.get("shift_audio", shot.h3_shift_audio)),
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
        expected_dialogue_tracks = sum(bool(str(segment.get("dialogue") or "").strip()) for segment in segment_plan)
        speech_metadata: dict[str, object] = {"generated": False, "reason": "audio_mode_not_clean_tts", "tracks": []}
        if settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE == "clean_tts":
            elapsed_delivery = 0.0
            metadata_tracks: list[dict[str, object]] = []
            for segment_index, segment in enumerate(segment_plan):
                segment_duration = float(segment["duration"])
                dialogue = str(segment.get("dialogue") or "").strip()
                if dialogue:
                    segment_shot = shot.model_copy(deep=True)
                    segment_shot.dialogue = dialogue
                    segment_shot.dialogue_speaker_id = segment.get("speaker_id") or shot.dialogue_speaker_id
                    segment_shot.dialogue_turns = []
                    speech_candidate = output_dir / f"{job.id}.dialogue{segment_index + 1}.mp3"
                    try:
                        metadata = await DialogueAudioService.synthesize(project, segment_shot, speech_candidate)
                        metadata["segment"] = segment_index + 1
                        metadata_tracks.append(metadata)
                        if metadata.get("generated"):
                            local_offset = min(shot.dialogue_start_seconds, max(0.05, segment_duration * 0.2))
                            speech_tracks.append(
                                {
                                    "path": speech_candidate,
                                    "start_seconds": elapsed_delivery + local_offset,
                                    "available_seconds": max(0.3, segment_duration - local_offset - 0.05),
                                    "speaker_id": segment_shot.dialogue_speaker_id,
                                }
                            )
                    except Exception as exc:
                        logger.exception("Independent dialogue generation failed for shot %s segment %s", shot.id, segment_index + 1)
                        metadata_tracks.append(
                            {"generated": False, "reason": "tts_failed", "error": str(exc), "segment": segment_index + 1}
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

    @staticmethod
    def _required_assets(shot, assets: list[Asset], mode: GenerationMode) -> list[Asset]:
        asset_map = {asset.id: asset for asset in assets}
        ids: list[str] = []
        if mode == GenerationMode.I2V:
            ids.extend(asset_id for asset_id in [shot.keyframe_asset_id, shot.last_frame_asset_id] if asset_id)
        else:
            ids.extend(shot.reference_asset_ids)
            if not ids and shot.keyframe_asset_id:
                ids.append(shot.keyframe_asset_id)
        result = [asset_map[asset_id] for asset_id in ids if asset_id in asset_map]
        if mode == GenerationMode.I2V and not result and not shot.image_path:
            raise ValueError(f"Shot {shot.ordinal} needs an approved first frame before I2V rendering")
        if mode == GenerationMode.R2V and not result:
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
