"""Atlas Cloud hosted MiniMax H3 Developer reference-to-video client.

This is the paid online alternative to the self-hosted ComfyUI H3 stack:
upload local references -> submit -> poll -> download.  The reference-to-video
API expects public URLs, so local media is uploaded through Atlas ``uploadMedia``
instead of being inlined into the generation payload.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from src.video_workflow.config import settings

ATLAS_MODEL_ID = "minimax/h3-developer/reference-to-video"
# The hosted model renders up to 15 continuous seconds per call; segment plans
# for this provider must use it as the split cap instead of the local 7s cap.
ATLAS_MAX_DURATION_SECONDS = 15.0
# Keep the request comfortably below MySQL TEXT-size storage used by some
# hosted workers. Normal URL-based requests are only a few KB; this guard also
# prevents a future regression that reintroduces inline media.
ATLAS_MAX_REQUEST_BYTES = 60 * 1024

_RATIO_VALUES = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
}


def atlas_resolution(width: int, height: int) -> str:
    """Map project pixel size onto the API's two resolution tiers."""
    return "768P" if min(int(width), int(height)) <= 800 else "1080P"


def atlas_ratio(width: int, height: int) -> str:
    return min(_RATIO_VALUES, key=lambda key: abs(_RATIO_VALUES[key] - int(width) / max(1, int(height))))


def atlas_duration(seconds: float) -> int:
    return int(max(4, min(15, round(float(seconds)))))


class AtlasH3Client:
    def __init__(self) -> None:
        self.base_url = (settings.ATLASCLOUD_BASE_URL or "https://api.atlascloud.ai").rstrip("/")
        self.api_key = settings.ATLASCLOUD_API_KEY or ""

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("未配置 Atlas Cloud API Key：请在模型设置的 H3 分组填写后保存")
        return {"Authorization": f"Bearer {self.api_key}"}

    async def upload_media(self, path: Path) -> str:
        """Upload a local reference and return Atlas' temporary HTTPS URL."""
        if not path.is_file():
            raise FileNotFoundError(f"Atlas H3 上传素材不存在: {path.name}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            with path.open("rb") as handle:
                response = await client.post(
                    f"{self.base_url}/api/v1/model/uploadMedia",
                    headers=self._headers(),
                    files={"file": (path.name, handle, mime_type)},
                )
        if response.status_code >= 400:
            raise RuntimeError(f"Atlas H3 素材上传失败 ({response.status_code}): {response.text[:300]}")
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        candidates = [
            payload.get("url") if isinstance(payload, dict) else None,
            payload.get("file_url") if isinstance(payload, dict) else None,
            payload.get("download_url") if isinstance(payload, dict) else None,
            data.get("url") if isinstance(data, dict) else None,
            data.get("file_url") if isinstance(data, dict) else None,
            data.get("download_url") if isinstance(data, dict) else None,
        ]
        url = next((str(item) for item in candidates if item), "")
        if not url.startswith(("http://", "https://")):
            raise RuntimeError(f"Atlas H3 素材上传响应缺少 HTTP(S) URL: {response.text[:300]}")
        return url

    async def submit(
        self,
        *,
        prompt: str,
        refers: list[dict[str, str]],
        resolution: str,
        duration: int,
        ratio: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": ATLAS_MODEL_ID,
            "prompt": prompt,
            "refers": refers,
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
            "prompt_expansion": False,
        }
        if not refers:
            raise ValueError("Atlas H3 reference-to-video 至少需要一项图片或视频参考")
        for index, item in enumerate(refers, start=1):
            url = str(item.get("url") or "")
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Atlas H3 第 {index} 项参考素材必须使用 HTTP(S) URL")
        if not any(str(item.get("type") or "") in {"image", "video"} for item in refers):
            raise ValueError("Atlas H3 reference-to-video 至少需要一项图片或视频参考")
        payload_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if payload_bytes > ATLAS_MAX_REQUEST_BYTES:
            raise ValueError(
                f"Atlas H3 请求参数过大（{payload_bytes} bytes）；"
                "请减少参考素材数量或提示词长度"
            )
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/model/generateVideo",
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Atlas H3 提交失败 ({response.status_code}): {response.text[:300]}")
            data = response.json().get("data") or {}
            prediction_id = str(data.get("id") or "")
            if not prediction_id:
                raise RuntimeError(f"Atlas H3 提交失败：响应缺少任务 id {response.text[:300]}")
            return prediction_id

    async def status(self, prediction_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/model/prediction/{prediction_id}",
                headers=self._headers(),
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Atlas H3 查询失败 ({response.status_code}): {response.text[:300]}")
            payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, dict) else {}

    async def wait_for_result(
        self,
        prediction_id: str,
        progress: Callable[[float, str], Awaitable[None]],
    ) -> list[str]:
        deadline = time.monotonic() + max(60, int(settings.ATLASCLOUD_JOB_TIMEOUT_SECONDS))
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Atlas H3 任务超过 {settings.ATLASCLOUD_JOB_TIMEOUT_SECONDS} 秒仍未完成")
            await asyncio.sleep(max(1.0, settings.ATLASCLOUD_POLL_INTERVAL_SECONDS))
            data = await self.status(prediction_id)
            status = str(data.get("status") or "").lower()
            if status in {"completed", "succeeded"}:
                outputs = [str(item) for item in data.get("outputs") or [] if item]
                if not outputs:
                    raise RuntimeError("Atlas H3 任务完成但没有返回视频地址")
                return outputs
            if status == "failed":
                raise RuntimeError(f"Atlas H3 生成失败: {data.get('error') or '未知错误'}")
            await progress(0.4 if status == "processing" else 0.15, f"Atlas {status or 'queued'} · 任务 {prediction_id}")

    async def download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            destination.write_bytes(response.content)
