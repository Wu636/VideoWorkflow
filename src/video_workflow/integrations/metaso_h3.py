"""MetaSo hosted MiniMax H3 v2 client.

The gateway follows MiniMax's multimodal ``content`` request format and returns
an asynchronous task id.  This client deliberately keeps Context IR opt-in so
ordinary renders do not silently incur its separate per-request fee.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from src.video_workflow.config import settings

METASO_H3_MODEL_ID = "MiniMax-H3"
METASO_H3_MAX_DURATION_SECONDS = 15.0
METASO_H3_MAX_REQUEST_BYTES = 60 * 1024 * 1024
METASO_H3_MAX_IMAGES = 9
METASO_H3_MAX_VIDEOS = 3
METASO_H3_MAX_AUDIOS = 3
METASO_H3_MAX_MEDIA = 12

METASO_H3_RESOLUTIONS = {"768P", "2K"}
METASO_H3_RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


def metaso_duration(seconds: float) -> int:
    return int(max(4, min(15, round(float(seconds)))))


def metaso_ratio(project_ratio: str, configured_ratio: str = "adaptive") -> str:
    """Use the project's authored canvas ratio for the MetaSo request.

    ``adaptive`` used to be copied from the global model setting even when a
    project had an explicit 9:16 canvas.  That leaves the hosted model free to
    return a landscape clip.  A supported project ratio is the more specific
    instruction; the global setting remains a fallback for legacy/custom
    canvases whose ratio is not part of MiniMax's accepted enum.
    """
    authored = str(project_ratio or "").strip()
    if authored in METASO_H3_RATIOS - {"adaptive"}:
        return authored
    configured = str(configured_ratio or "").strip()
    return configured if configured in METASO_H3_RATIOS else "adaptive"


def metaso_request_ratio(
    project_ratio: str,
    shot_ratio: str = "project",
    configured_ratio: str = "adaptive",
) -> str:
    """Resolve a per-shot API ratio without losing the project default."""
    requested = str(shot_ratio or "project").strip()
    if requested == "project":
        return metaso_ratio(project_ratio, configured_ratio)
    if requested in METASO_H3_RATIOS:
        return requested
    return metaso_ratio(project_ratio, configured_ratio)


def metaso_resolution(shot_resolution: str = "default", configured_resolution: str = "768P") -> str:
    requested = str(shot_resolution or "default").strip()
    if requested in METASO_H3_RESOLUTIONS:
        return requested
    configured = str(configured_resolution or "").strip()
    return configured if configured in METASO_H3_RESOLUTIONS else "768P"


def media_data_url(path: Path) -> str:
    """Encode a small local reference for the v2 API without a second upload."""
    if not path.is_file():
        raise FileNotFoundError(f"MetaSo H3 本地素材不存在: {path.name}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class MetaSoH3Client:
    def __init__(self) -> None:
        self.base_url = (settings.METASO_H3_BASE_URL or "https://metaso.cn/api/minimax").rstrip("/")
        self.api_key = settings.METASO_H3_API_KEY or ""

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("未配置 MetaSo H3 API Key：请在模型设置的 H3 分组填写后保存")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _api_message(payload: object, fallback: str) -> str:
        if not isinstance(payload, dict):
            return fallback
        return str(payload.get("message") or payload.get("msg") or payload.get("error") or fallback)

    @staticmethod
    def _content_item(media_type: str, url: str, role: str) -> dict[str, Any]:
        field = f"{media_type}_url"
        return {"type": field, field: {"url": url}, "role": role}

    async def submit(
        self,
        *,
        prompt: str,
        image_urls: list[str] | None = None,
        video_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        first_frame_url: str | None = None,
        last_frame_url: str | None = None,
        resolution: str,
        duration: int,
        ratio: str,
        context_ir_enabled: bool | None = None,
    ) -> str:
        image_urls = list(dict.fromkeys(image_urls or []))
        video_urls = list(dict.fromkeys(video_urls or []))
        audio_urls = list(dict.fromkeys(audio_urls or []))
        frame_urls = [url for url in (first_frame_url, last_frame_url) if url]
        if frame_urls and (image_urls or video_urls or audio_urls):
            raise ValueError("MetaSo H3 帧模式与全能参考模式需要分开提交")
        image_count = len(image_urls) + len(frame_urls)
        media_count = image_count + len(video_urls) + len(audio_urls)
        if image_count > METASO_H3_MAX_IMAGES:
            raise ValueError(f"MetaSo H3 单次最多 {METASO_H3_MAX_IMAGES} 张图片，当前 {image_count} 张")
        if len(video_urls) > METASO_H3_MAX_VIDEOS:
            raise ValueError(f"MetaSo H3 单次最多 {METASO_H3_MAX_VIDEOS} 个视频，当前 {len(video_urls)} 个")
        if len(audio_urls) > METASO_H3_MAX_AUDIOS:
            raise ValueError(f"MetaSo H3 单次最多 {METASO_H3_MAX_AUDIOS} 个音频，当前 {len(audio_urls)} 个")
        if media_count > METASO_H3_MAX_MEDIA:
            raise ValueError(f"MetaSo H3 单次最多 {METASO_H3_MAX_MEDIA} 项素材，当前 {media_count} 项")
        if not frame_urls and not image_urls and not video_urls and not audio_urls:
            raise ValueError("MetaSo H3 至少需要一项首帧、尾帧或全能参考素材")
        if resolution not in METASO_H3_RESOLUTIONS:
            raise ValueError(f"MetaSo H3 分辨率仅支持 768P 或 2K，收到 {resolution}")
        if ratio not in METASO_H3_RATIOS:
            raise ValueError(f"MetaSo H3 不支持画面比例 {ratio}")
        if int(duration) < 4 or int(duration) > 15:
            raise ValueError("MetaSo H3 单段时长范围为 4–15 秒")

        content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt).strip()}]
        if first_frame_url:
            content.append(self._content_item("image", first_frame_url, "first_frame"))
        if last_frame_url:
            content.append(self._content_item("image", last_frame_url, "last_frame"))
        content.extend(self._content_item("image", url, "reference_image") for url in image_urls)
        content.extend(self._content_item("video", url, "reference_video") for url in video_urls)
        content.extend(self._content_item("audio", url, "reference_audio") for url in audio_urls)
        payload: dict[str, Any] = {
            "model": METASO_H3_MODEL_ID,
            "content": content,
            "resolution": resolution,
            "duration": int(duration),
            "ratio": ratio,
            "context_ir_enabled": (
                bool(settings.METASO_H3_CONTEXT_IR_ENABLED)
                if context_ir_enabled is None
                else bool(context_ir_enabled)
            ),
        }
        payload_size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if payload_size > METASO_H3_MAX_REQUEST_BYTES:
            raise ValueError(
                f"MetaSo H3 请求体过大（{payload_size / 1024 / 1024:.1f}MB）；"
                "请减少或压缩内联参考素材，或配置素材公网地址"
            )

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self.base_url}/v2/video_generation",
                headers=self._headers(),
                json=payload,
            )
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {}
        if response.status_code >= 400:
            message = self._api_message(response_payload, response.text[:300])
            raise RuntimeError(f"MetaSo H3 提交失败 ({response.status_code}): {message}")
        data = response_payload.get("data") if isinstance(response_payload, dict) else None
        task_id = str(
            (response_payload.get("task_id") if isinstance(response_payload, dict) else None)
            or (data.get("task_id") if isinstance(data, dict) else None)
            or ""
        )
        if not task_id:
            raise RuntimeError(f"MetaSo H3 提交响应缺少 task_id: {response.text[:300]}")
        return task_id

    async def status(self, task_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{self.base_url}/v2/query/video_generation/{task_id}",
                headers=self._headers(),
            )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code >= 400:
            message = self._api_message(payload, response.text[:300])
            raise RuntimeError(f"MetaSo H3 查询失败 ({response.status_code}): {message}")
        data = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        task = data.get("task") if isinstance(data, dict) and isinstance(data.get("task"), dict) else data
        return task if isinstance(task, dict) else {}

    @staticmethod
    def _result_url(task: dict[str, Any]) -> str:
        content = task.get("content")
        candidates: list[object] = [task.get("url"), task.get("video_url"), task.get("output_url")]
        if isinstance(content, dict):
            candidates.extend([content.get("url"), content.get("video_url"), content.get("output_url")])
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    nested = item.get("video_url")
                    candidates.extend([item.get("url"), nested.get("url") if isinstance(nested, dict) else nested])
        return next((str(value) for value in candidates if str(value or "").startswith(("http://", "https://"))), "")

    async def wait_for_result(
        self,
        task_id: str,
        progress: Callable[[float, str], Awaitable[None]],
    ) -> str:
        deadline = time.monotonic() + max(60, int(settings.METASO_H3_JOB_TIMEOUT_SECONDS))
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"MetaSo H3 任务超过 {settings.METASO_H3_JOB_TIMEOUT_SECONDS} 秒仍未完成")
            await asyncio.sleep(max(1.0, float(settings.METASO_H3_POLL_INTERVAL_SECONDS)))
            task = await self.status(task_id)
            status = str(task.get("status") or task.get("state") or "").lower()
            if status in {"success", "succeeded", "completed", "done"}:
                url = self._result_url(task)
                if not url:
                    raise RuntimeError("MetaSo H3 任务完成但响应中没有视频地址")
                return url
            if status in {"failed", "failure", "error", "cancelled", "canceled"}:
                detail = task.get("error") or task.get("message") or task.get("fail_reason") or "未知错误"
                raise RuntimeError(f"MetaSo H3 生成失败: {detail}")
            await progress(0.4 if status in {"processing", "running"} else 0.15, f"MetaSo {status or 'queued'} · 任务 {task_id}")

    async def preflight(self) -> dict[str, Any]:
        """Validate gateway authentication with a read-only query."""
        if not self.api_key:
            return {
                "online": False,
                "ok": False,
                "provider": "metaso_h3",
                "message": "MetaSo H3 API Key 尚未配置",
            }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/v2/query/video_generation/__videoworkflow_preflight__",
                headers=self._headers(),
            )
        # An unknown-task response proves the gateway accepted authentication;
        # only auth/server failures make this preflight fail.
        ok = response.status_code not in {401, 403} and response.status_code < 500
        return {
            "online": response.status_code < 500,
            "ok": ok,
            "provider": "metaso_h3",
            "model": METASO_H3_MODEL_ID,
            "resolution": settings.METASO_H3_RESOLUTION,
            "ratio": settings.METASO_H3_RATIO,
            "context_ir_enabled": bool(settings.METASO_H3_CONTEXT_IR_ENABLED),
            "message": "MetaSo H3 鉴权与查询接口已就绪" if ok else f"MetaSo H3 配置检查失败 ({response.status_code})",
        }

    async def download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            destination.write_bytes(response.content)
