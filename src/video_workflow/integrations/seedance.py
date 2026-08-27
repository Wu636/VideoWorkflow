from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import Asset, AssetType
from src.video_workflow.media_paths import resolve_media_path


@dataclass(frozen=True)
class SeedanceModel:
    id: str
    label: str
    description: str
    price_per_million_tokens: float
    video_input_price_per_million_tokens: float
    resolutions: tuple[str, ...]


SEEDANCE_MODELS: dict[str, SeedanceModel] = {
    "doubao-seedance-2-0-260128": SeedanceModel(
        id="doubao-seedance-2-0-260128",
        label="Seedance 2.0",
        description="质量优先；支持 1080p 与 4K，适合最终交付。",
        price_per_million_tokens=46.0,
        video_input_price_per_million_tokens=28.0,
        resolutions=("480p", "720p", "1080p", "4k"),
    ),
    "doubao-seedance-2-0-fast-260128": SeedanceModel(
        id="doubao-seedance-2-0-fast-260128",
        label="Seedance 2.0 Fast",
        description="速度、质量与成本均衡；最高 720p。",
        price_per_million_tokens=37.0,
        video_input_price_per_million_tokens=22.0,
        resolutions=("480p", "720p"),
    ),
    "doubao-seedance-2-0-mini-260615": SeedanceModel(
        id="doubao-seedance-2-0-mini-260615",
        label="Seedance 2.0 Mini",
        description="成本优先；适合批量初稿与预算敏感项目，最高 720p。",
        price_per_million_tokens=23.0,
        video_input_price_per_million_tokens=14.0,
        resolutions=("480p", "720p"),
    ),
}

RESOLUTION_SHORT_EDGE = {"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}
SUPPORTED_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive"}
INLINE_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def _asset_signing_secret() -> bytes:
    # Reuse the already-secret Ark key as local HMAC material.  The key itself
    # is never included in the URL or response body.
    material = settings.ARK_API_KEY or "videoworkflow-local-asset-signing"
    return hashlib.sha256(material.encode("utf-8")).digest()


def sign_seedance_asset(asset_id: str, expires: int) -> str:
    payload = f"{asset_id}:{expires}".encode("utf-8")
    return hmac.new(_asset_signing_secret(), payload, hashlib.sha256).hexdigest()


def verify_seedance_asset_signature(asset_id: str, expires: int, signature: str) -> bool:
    now = int(time.time())
    max_ttl = max(300, int(settings.SEEDANCE_ASSET_URL_TTL_SECONDS))
    if expires < now or expires > now + max_ttl + 300:
        return False
    return hmac.compare_digest(sign_seedance_asset(asset_id, expires), signature)


def _is_dns_failure(exc: BaseException) -> bool:
    """Detect hostname resolution failures, the typical symptom of an expired tunnel domain."""
    current: BaseException | None = exc
    for _ in range(8):
        if current is None:
            break
        if isinstance(current, socket.gaierror):
            return True
        current = current.__cause__ or current.__context__
    text = str(exc).lower()
    return (
        "name or service not known" in text
        or "nodename nor servname" in text
        or "getaddrinfo failed" in text
    )


def _dns_failure_hint() -> str:
    return (
        "（域名无法解析，Seedance 隧道可能已过期；请运行 scripts/restart-seedance-tunnel.sh "
        "刷新隧道并回写配置，或在设置中更新“Seedance 素材公网地址”）"
    )


def resolve_seedance_model(model_id: str | None) -> SeedanceModel:
    resolved = (model_id or settings.SEEDANCE_DEFAULT_MODEL).strip()
    try:
        return SEEDANCE_MODELS[resolved]
    except KeyError as exc:
        raise ValueError(f"不支持的 Seedance 2.0 模型: {resolved}") from exc


def validate_seedance_resolution(model_id: str, resolution: str | None) -> str:
    model = resolve_seedance_model(model_id)
    resolved = (resolution or settings.SEEDANCE_DEFAULT_RESOLUTION).strip().lower()
    if resolved not in model.resolutions:
        raise ValueError(f"{model.label} 不支持 {resolved}；可选：{', '.join(model.resolutions)}")
    return resolved


def seedance_dimensions(resolution: str, ratio: str) -> tuple[int, int]:
    short_edge = RESOLUTION_SHORT_EDGE[resolution]
    normalized = ratio if ratio in SUPPORTED_RATIOS and ratio != "adaptive" else "16:9"
    left, right = (int(value) for value in normalized.split(":"))
    if left >= right:
        height = short_edge
        width = round(short_edge * left / right)
    else:
        width = short_edge
        height = round(short_edge * right / left)
    # Video encoders require even dimensions.
    return width + width % 2, height + height % 2


def estimate_seedance_cost(
    model_id: str,
    resolution: str,
    ratio: str,
    durations: list[float],
    *,
    has_video_input: bool = False,
) -> dict[str, Any]:
    model = resolve_seedance_model(model_id)
    resolution = validate_seedance_resolution(model.id, resolution)
    width, height = seedance_dimensions(resolution, ratio)
    billed_durations = [max(4, min(15, math.ceil(float(value)))) for value in durations]
    tokens = sum(width * height * (24 * duration + 1) / 1024 for duration in billed_durations)
    unit_price = (
        model.video_input_price_per_million_tokens
        if has_video_input
        else model.price_per_million_tokens
    )
    cost = tokens / 1_000_000 * unit_price
    per_second = width * height * 24 / 1024 / 1_000_000 * unit_price
    per_task = width * height / 1024 / 1_000_000 * unit_price
    return {
        "model_id": model.id,
        "resolution": resolution,
        "ratio": ratio,
        "width": width,
        "height": height,
        "shot_count": len(billed_durations),
        "requested_duration_seconds": round(sum(float(value) for value in durations), 3),
        "billed_duration_seconds": sum(billed_durations),
        "estimated_tokens": round(tokens),
        "unit_price_per_million_tokens": unit_price,
        "estimated_yuan": round(cost, 4),
        "estimated_yuan_per_second": round(per_second, 4),
        "estimated_yuan_per_task": round(per_task, 4),
        "has_video_input": has_video_input,
        "note": "按官方 token 单价和输出规格预估；最终以方舟任务 usage 与账单为准。",
    }


def seedance_catalog() -> dict[str, Any]:
    return {
        "default_model": settings.SEEDANCE_DEFAULT_MODEL,
        "default_resolution": settings.SEEDANCE_DEFAULT_RESOLUTION,
        "models": [
            {
                "id": model.id,
                "label": model.label,
                "description": model.description,
                "price_per_million_tokens": model.price_per_million_tokens,
                "video_input_price_per_million_tokens": model.video_input_price_per_million_tokens,
                "resolutions": list(model.resolutions),
                "example_720p_yuan_per_second": estimate_seedance_cost(
                    model.id,
                    "720p",
                    "16:9",
                    [5],
                )["estimated_yuan_per_second"],
                "example_720p_yuan_per_task": estimate_seedance_cost(
                    model.id,
                    "720p",
                    "16:9",
                    [5],
                )["estimated_yuan_per_task"],
            }
            for model in SEEDANCE_MODELS.values()
        ],
        "ratios": sorted(SUPPORTED_RATIOS - {"adaptive"}),
        "configured": bool(settings.ARK_API_KEY),
        "pricing_source": "https://www.volcengine.com/product/ark",
        "prompt_guide": "https://www.volcengine.com/docs/82379/2222480?lang=zh",
    }


class SeedanceClient:
    def __init__(self) -> None:
        self.base_url = settings.ARK_BASE_URL.rstrip("/")
        self.api_key = settings.ARK_API_KEY
        self.timeout = httpx.Timeout(connect=30.0, read=120.0, write=180.0, pool=60.0)

    async def preflight(self, model_id: str | None = None, resolution: str | None = None) -> dict[str, Any]:
        model = resolve_seedance_model(model_id)
        resolved_resolution = validate_seedance_resolution(model.id, resolution)
        has_asset_origin = bool((settings.SEEDANCE_PUBLIC_ASSET_BASE_URL or "").strip())
        configured = bool(self.api_key)
        # The probe request may loop back through the tunnel into this same
        # server; keep the event loop free by running it in a thread.
        probe = await asyncio.to_thread(self.probe_asset_origin) if has_asset_origin else {
            "ok": False,
            "error": "未配置 Seedance 素材公网地址",
        }
        origin_ok = bool(probe.get("ok"))
        ok = configured and origin_ok
        if not configured:
            message = "请先在模型设置中填写火山方舟 API Key。"
        elif not has_asset_origin:
            message = "方舟 API Key 已配置，但还需要 Seedance 素材公网地址。"
        elif not origin_ok:
            message = f"方舟 API Key 已配置，但素材公网通道不可用：{probe.get('error')}"
        else:
            message = "方舟 API Key 与素材公网通道已配置且可达；检查未创建任务、未产生视频费用。"
        return {
            "online": bool(self.api_key),
            "ok": ok,
            "configured": configured,
            "asset_origin_configured": has_asset_origin,
            "asset_origin_ok": origin_ok,
            "asset_origin_error": probe.get("error"),
            "asset_origin": (settings.SEEDANCE_PUBLIC_ASSET_BASE_URL or "").strip(),
            "base_url": self.base_url,
            "model_id": model.id,
            "model_label": model.label,
            "resolution": resolved_resolution,
            "message": message,
        }

    def probe_asset_origin(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Verify the public asset origin actually routes back to this backend.

        Quick-tunnel hostnames are session-scoped: once the tunnel session dies
        the hostname stops resolving and every Seedance submit would later fail
        the pre-submit readability check.  Requesting the signed asset endpoint
        with a bogus signature proves end-to-end reachability without exposing
        any media: any HTTP response (typically 403) means the tunnel is alive.
        """
        origin = (settings.SEEDANCE_PUBLIC_ASSET_BASE_URL or "").strip().rstrip("/")
        if not origin:
            return {"ok": False, "error": "未配置 Seedance 素材公网地址"}
        url = f"{origin}/api/projects/seedance-assets/__probe__?expires=0&signature=probe"
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
                status_code = client.get(url).status_code
        except Exception as exc:
            if _is_dns_failure(exc):
                return {
                    "ok": False,
                    "error": (
                        f"隧道域名无法解析（{exc}），可能已过期；请运行 "
                        "scripts/restart-seedance-tunnel.sh 刷新隧道并回写配置，"
                        "或在设置中手动更新“Seedance 素材公网地址”。"
                    ),
                }
            return {"ok": False, "error": f"素材公网通道不可达：{exc}"}
        return {"ok": True, "error": None, "status_code": status_code}

    @staticmethod
    def _asset_url(asset: Asset) -> str:
        if asset.path.startswith(("http://", "https://", "data:")):
            return asset.path
        path = resolve_media_path(asset.path)
        if not path.is_file():
            raise FileNotFoundError(f"素材文件不存在: {asset.name}")
        limit = max(1, settings.SEEDANCE_INLINE_ASSET_MAX_MB) * 1024 * 1024
        if asset.type == AssetType.IMAGE and path.stat().st_size <= limit:
            # Ark Seedance accepts base64 data URLs for images.  Inlining keeps
            # keyframe/character references independent from the public tunnel,
            # whose latency occasionally trips Ark's fetch timeout.
            mime = INLINE_IMAGE_MIME.get(path.suffix.lower(), "image/png")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        public_base = (settings.SEEDANCE_PUBLIC_ASSET_BASE_URL or "").strip().rstrip("/")
        if public_base:
            expires = int(time.time()) + max(300, int(settings.SEEDANCE_ASSET_URL_TTL_SECONDS))
            signature = sign_seedance_asset(asset.id, expires)
            return f"{public_base}/api/projects/seedance-assets/{asset.id}?expires={expires}&signature={signature}"
        raise ValueError(
            f"素材 {asset.name} 为本地文件且无法内联（仅 {settings.SEEDANCE_INLINE_ASSET_MAX_MB}MB 内图片支持 base64）。"
            "方舟 Seedance 需要公网可读 HTTP(S) URL，请先在模型设置中配置 Seedance 素材公网地址"
        )

    async def _validate_asset_urls(self, content: list[dict[str, Any]]) -> None:
        urls: list[tuple[str, str]] = []
        for index, item in enumerate(content[1:], start=1):
            field = str(item.get("type") or "")
            nested = item.get(field)
            url = nested.get("url") if isinstance(nested, dict) else None
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append((f"content[{index}].{field}", url))

        async def check(label: str, url: str) -> None:
            last: Exception | None = None
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0),
                        follow_redirects=True,
                    ) as client:
                        async with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes():
                                if chunk:
                                    break
                    return
                except Exception as exc:
                    last = exc if isinstance(exc, Exception) else Exception(str(exc))
                    if _is_dns_failure(exc):
                        break
                    if attempt < 2:
                        await asyncio.sleep(2.0 * (attempt + 1))
            hint = _dns_failure_hint() if _is_dns_failure(last) else "（公网通道瞬时抖动，已自动重试 3 次仍失败）"
            raise ValueError(
                f"{label} 的素材公网地址不可读取{hint}：{last}"
            ) from last

        await asyncio.gather(*(check(label, url) for label, url in urls))

    def build_content(
        self,
        prompt: str,
        assets: list[Asset],
        first_frame_asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        supported_assets = [
            asset
            for asset in assets
            if asset.type in {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
        ]
        # Ark treats strict first/last-frame generation and all-modal reference
        # generation as mutually exclusive request modes.  When additional
        # reference media is present, keep every item as reference media and let
        # the prompt ask 图片1 to be used as the opening composition.  A strict
        # first_frame role is emitted only for a single-image I2V request.
        strict_first_frame_asset_id = (
            first_frame_asset_id
            if len(supported_assets) == 1
            and supported_assets[0].type == AssetType.IMAGE
            and supported_assets[0].id == first_frame_asset_id
            else None
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        counts = {AssetType.IMAGE: 0, AssetType.VIDEO: 0, AssetType.AUDIO: 0}
        for asset in supported_assets:
            counts[asset.type] += 1
            limits = {AssetType.IMAGE: 9, AssetType.VIDEO: 3, AssetType.AUDIO: 3}
            if counts[asset.type] > limits[asset.type]:
                continue
            field = {
                AssetType.IMAGE: "image_url",
                AssetType.VIDEO: "video_url",
                AssetType.AUDIO: "audio_url",
            }[asset.type]
            content.append(
                {
                    "type": field,
                    field: {"url": self._asset_url(asset)},
                    "role": {
                        AssetType.IMAGE: (
                            "first_frame" if asset.id == strict_first_frame_asset_id else "reference_image"
                        ),
                        AssetType.VIDEO: "reference_video",
                        AssetType.AUDIO: "reference_audio",
                    }[asset.type],
                }
            )
        return content

    async def submit(
        self,
        *,
        model_id: str,
        prompt: str,
        assets: list[Asset],
        ratio: str,
        duration: float,
        resolution: str,
        seed: int,
        generate_audio: bool,
        first_frame_asset_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not self.api_key:
            raise ValueError("请先在模型设置中填写火山方舟 API Key")
        model = resolve_seedance_model(model_id)
        resolution = validate_seedance_resolution(model.id, resolution)
        ratio = ratio if ratio in SUPPORTED_RATIOS else "adaptive"
        billed_duration = max(4, min(15, math.ceil(duration)))
        content = await asyncio.to_thread(
            self.build_content,
            prompt,
            assets,
            first_frame_asset_id,
        )
        await self._validate_asset_urls(content)
        payload = {
            "model": model.id,
            "content": content,
            "generate_audio": generate_audio,
            "return_last_frame": True,
            "resolution": resolution,
            "ratio": ratio,
            "duration": billed_duration,
            "seed": seed,
            "watermark": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
            response = await client.post("/contents/generations/tasks", json=payload)
            if response.is_error:
                raise RuntimeError(f"Seedance 提交失败 ({response.status_code}): {response.text[:500]}")
            body = response.json()
        task_id = str(body.get("id") or "")
        if not task_id:
            raise RuntimeError(f"Seedance 提交响应缺少任务 ID: {body}")
        snapshot = {
            "model": model.id,
            "resolution": resolution,
            "ratio": ratio,
            "duration": billed_duration,
            "generate_audio": generate_audio,
            "first_frame_asset_id": first_frame_asset_id,
            "reference_counts": {
                "images": sum(asset.type == AssetType.IMAGE for asset in assets),
                "videos": sum(asset.type == AssetType.VIDEO for asset in assets),
                "audios": sum(asset.type == AssetType.AUDIO for asset in assets),
            },
        }
        return task_id, snapshot

    @staticmethod
    def last_frame_url(payload: dict[str, Any]) -> str | None:
        content = payload.get("content")
        if isinstance(content, dict):
            for key in ("last_frame_url", "last_frame"):
                value = content.get(key)
                if isinstance(value, str) and value:
                    return value
                if isinstance(value, dict) and value.get("url"):
                    return str(value["url"])
        for key in ("last_frame_url", "last_frame"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and value.get("url"):
                return str(value["url"])
        return None

    async def get(self, task_id: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("火山方舟 API Key 未配置")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
            response = await client.get(f"/contents/generations/tasks/{task_id}")
            if response.is_error:
                raise RuntimeError(f"Seedance 查询失败 ({response.status_code}): {response.text[:500]}")
            return response.json()

    async def cancel(self, task_id: str) -> None:
        if not self.api_key:
            return
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
            response = await client.delete(f"/contents/generations/tasks/{task_id}")
            if response.status_code not in {200, 204, 404}:
                raise RuntimeError(f"Seedance 取消失败 ({response.status_code}): {response.text[:500]}")

    @staticmethod
    def video_url(payload: dict[str, Any]) -> str:
        content = payload.get("content")
        if isinstance(content, dict):
            value = content.get("video_url") or content.get("url")
            if isinstance(value, str) and value:
                return value
        for key in ("video_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.lower().split("?")[0].endswith(".mp4"):
                return value
        raise RuntimeError(f"Seedance 成功响应中没有视频地址: {payload}")

    async def download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            await asyncio.to_thread(destination.write_bytes, response.content)
