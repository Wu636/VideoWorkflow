from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetType, Delivery, FinalizeRequest, ProjectStatus
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore

logger = logging.getLogger(__name__)


class FinalizeError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FinalizeError(f"Command failed: {' '.join(command[:8])}\n{result.stderr[-3000:]}")
    return result


def probe_media(path: Path) -> dict[str, Any]:
    result = _run(
        [
            settings.FFPROBE_BIN,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def _has_audio(probe: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    return next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), {})


class Finalizer:
    def __init__(self, store: ProjectStore, projects: ProjectService):
        self.store = store
        self.projects = projects

    def finalize(self, project_id: str, request: FinalizeRequest) -> Delivery:
        if not shutil.which(settings.FFMPEG_BIN) or not shutil.which(settings.FFPROBE_BIN):
            raise FinalizeError("ffmpeg and ffprobe must be installed")
        project = self.projects.require_project(project_id)
        shots = self.store.list_shots(project_id)
        if request.shot_ids:
            order = {shot_id: idx for idx, shot_id in enumerate(request.shot_ids)}
            shots = sorted((shot for shot in shots if shot.id in order), key=lambda shot: order[shot.id])
        if not shots or any(not shot.video_path for shot in shots):
            missing = [str(shot.ordinal) for shot in shots if not shot.video_path]
            raise FinalizeError(f"All selected shots need video outputs. Missing: {', '.join(missing)}")

        project.status = ProjectStatus.EDITING
        self.store.save_project(project)
        delivery_dir = self.projects.project_dir(project_id) / "deliveries"
        delivery_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(request.output_name).name
        output_path = delivery_dir / safe_name
        preview_path: Path | None = None

        with tempfile.TemporaryDirectory(prefix="videoworkflow-finalize-") as temp_dir:
            work = Path(temp_dir)
            normalized: list[Path] = []
            durations: list[float] = []
            for index, shot in enumerate(shots, start=1):
                source = resolve_media_path(shot.video_path or "")
                if not source.exists():
                    raise FinalizeError(f"Shot {shot.ordinal} output does not exist: {source}")
                target = work / f"{index:04d}.mp4"
                self._normalize_clip(source, target, shot.duration_seconds, project.brief.width, project.brief.height, project.brief.fps, request.normalize_audio)
                normalized.append(target)
                durations.append(shot.duration_seconds)

            assembled = work / "assembled.mp4"
            if request.crossfade_seconds > 0 and len(normalized) > 1:
                self._crossfade(normalized, durations, assembled, request.crossfade_seconds)
            else:
                self._concat(normalized, assembled)

            processed = assembled
            if request.background_music_asset_id:
                asset = self.store.get_asset(request.background_music_asset_id)
                if asset is None or asset.type != AssetType.AUDIO:
                    raise FinalizeError("Background music asset is missing or is not audio")
                mixed = work / "mixed.mp4"
                self._mix_music(processed, resolve_media_path(asset.path), mixed, request.background_music_volume)
                processed = mixed

            if request.burn_subtitles and request.subtitle_asset_id:
                asset = self.store.get_asset(request.subtitle_asset_id)
                if asset is None or asset.type != AssetType.SUBTITLE:
                    raise FinalizeError("Subtitle asset is missing or has the wrong type")
                subtitled = work / "subtitled.mp4"
                self._burn_subtitles(processed, resolve_media_path(asset.path), subtitled)
                processed = subtitled

            shutil.copy2(processed, output_path)
            if request.preview:
                preview_path = delivery_dir / f"{output_path.stem}_preview.mp4"
                self._make_preview(processed, preview_path)

        qc = self.quality_check(output_path, project.brief.width, project.brief.height, project.brief.fps)
        if not qc["passed"]:
            # Final delivery is advisory: preserve the encoded file and its
            # QC details instead of turning a playable result into a failed
            # operation because of an optional stream or metadata mismatch.
            logger.warning(
                "Final delivery QC advisory for project %s (delivery continues): %s",
                project_id,
                ";".join(str(issue) for issue in qc.get("issues", [])),
            )
            qc["delivery_advisory"] = True
        delivery = Delivery(
            project_id=project_id,
            output_path=str(output_path.resolve()),
            preview_path=str(preview_path.resolve()) if preview_path else None,
            subtitle_path=(self.store.get_asset(request.subtitle_asset_id).path if request.subtitle_asset_id and self.store.get_asset(request.subtitle_asset_id) else None),
            duration_seconds=float(qc.get("duration_seconds", 0.0)),
            qc_report=qc,
        )
        self.store.save_delivery(delivery)
        project.status = ProjectStatus.FINAL_REVIEW
        self.store.save_project(project)
        return delivery

    def _normalize_clip(
        self,
        source: Path,
        target: Path,
        duration: float,
        width: int,
        height: int,
        fps: float,
        normalize_audio: bool,
    ) -> None:
        probe = probe_media(source)
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},format=yuv420p"
        )
        command = [settings.FFMPEG_BIN, "-y", "-i", str(source)]
        if _has_audio(probe):
            command += ["-map", "0:v:0", "-map", "0:a:0", "-vf", video_filter]
            audio_filter = f"atrim=0:{duration},apad"
            if normalize_audio:
                audio_filter += ",loudnorm=I=-16:TP=-1.5:LRA=11"
            command += ["-af", audio_filter]
        else:
            command += [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={settings.FINAL_AUDIO_SAMPLE_RATE}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                video_filter,
            ]
        command += [
            "-t",
            f"{duration:.3f}",
            "-c:v",
            settings.FINAL_VIDEO_CODEC,
            "-preset",
            settings.FINAL_VIDEO_PRESET,
            "-crf",
            str(settings.FINAL_VIDEO_CRF),
            "-c:a",
            settings.FINAL_AUDIO_CODEC,
            "-ar",
            str(settings.FINAL_AUDIO_SAMPLE_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(target),
        ]
        _run(command)

    @staticmethod
    def _concat(clips: list[Path], output: Path) -> None:
        list_path = output.with_suffix(".txt")
        list_path.write_text("".join(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in clips), encoding="utf-8")
        try:
            _run([settings.FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)])
        finally:
            list_path.unlink(missing_ok=True)

    @staticmethod
    def _crossfade(clips: list[Path], durations: list[float], output: Path, fade: float) -> None:
        fade = min(fade, min(durations) / 2)
        command = [settings.FFMPEG_BIN, "-y"]
        for clip in clips:
            command += ["-i", str(clip)]
        filters: list[str] = []
        video_prev = "[0:v]"
        audio_prev = "[0:a]"
        cumulative = durations[0]
        for index in range(1, len(clips)):
            offset = cumulative - fade * index
            video_out = f"[v{index}]"
            audio_out = f"[a{index}]"
            filters.append(f"{video_prev}[{index}:v]xfade=transition=fade:duration={fade}:offset={offset}{video_out}")
            filters.append(f"{audio_prev}[{index}:a]acrossfade=d={fade}{audio_out}")
            video_prev, audio_prev = video_out, audio_out
            cumulative += durations[index]
        command += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            video_prev,
            "-map",
            audio_prev,
            "-c:v",
            settings.FINAL_VIDEO_CODEC,
            "-preset",
            settings.FINAL_VIDEO_PRESET,
            "-crf",
            str(settings.FINAL_VIDEO_CRF),
            "-c:a",
            settings.FINAL_AUDIO_CODEC,
            str(output),
        ]
        _run(command)

    @staticmethod
    def _mix_music(video: Path, music: Path, output: Path, volume: float) -> None:
        _run(
            [
                settings.FFMPEG_BIN,
                "-y",
                "-i",
                str(video),
                "-stream_loop",
                "-1",
                "-i",
                str(music),
                "-filter_complex",
                f"[1:a]volume={volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                settings.FINAL_AUDIO_CODEC,
                "-shortest",
                str(output),
            ]
        )

    @staticmethod
    def _burn_subtitles(video: Path, subtitles: Path, output: Path) -> None:
        escaped = str(subtitles).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        _run(
            [
                settings.FFMPEG_BIN,
                "-y",
                "-i",
                str(video),
                "-vf",
                f"subtitles='{escaped}'",
                "-c:v",
                settings.FINAL_VIDEO_CODEC,
                "-crf",
                "18",
                "-c:a",
                "copy",
                str(output),
            ]
        )

    @staticmethod
    def _make_preview(video: Path, output: Path) -> None:
        _run(
            [
                settings.FFMPEG_BIN,
                "-y",
                "-i",
                str(video),
                "-c:v",
                settings.FINAL_VIDEO_CODEC,
                "-preset",
                "fast",
                "-crf",
                "28",
                "-c:a",
                settings.FINAL_AUDIO_CODEC,
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )

    @staticmethod
    def quality_check(path: Path, width: int, height: int, fps: float) -> dict[str, Any]:
        probe = probe_media(path)
        video = _video_stream(probe)
        duration = float(probe.get("format", {}).get("duration") or 0.0)
        actual_width = int(video.get("width") or 0)
        actual_height = int(video.get("height") or 0)
        rate = video.get("avg_frame_rate", "0/1")
        numerator, denominator = (rate.split("/") + ["1"])[:2]
        actual_fps = float(numerator) / max(float(denominator), 1.0)
        issues: list[str] = []
        if (actual_width, actual_height) != (width, height):
            issues.append(f"resolution mismatch: {actual_width}x{actual_height}")
        if abs(actual_fps - fps) > 0.1:
            issues.append(f"fps mismatch: {actual_fps:.3f}")
        if duration <= 0:
            issues.append("invalid duration")
        if not _has_audio(probe):
            issues.append("missing audio stream")
        return {
            "passed": not issues,
            "issues": issues,
            "duration_seconds": duration,
            "width": actual_width,
            "height": actual_height,
            "fps": actual_fps,
            "has_audio": _has_audio(probe),
            "size_bytes": path.stat().st_size,
        }
