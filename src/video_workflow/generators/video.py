import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from volcenginesdkarkruntime import Ark

from src.video_workflow.config import settings
from src.video_workflow.core.video_task_registry import register_video_task, update_video_task
from src.video_workflow.generators.base import VideoGenerator
from src.video_workflow.types import Scene

logger = logging.getLogger(__name__)


def _compact_payload(payload: Any, max_len: int = 400) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _resolve_motion_prompt(scene: Scene) -> str:
    prompt = (scene.motion_prompt or "").strip().rstrip("。")
    duration = getattr(scene, "duration", 6)
    try:
        duration = int(duration)
    except (TypeError, ValueError):
        duration = 6
    duration = max(5, min(10, duration))

    if not prompt:
        prompt = "主体先观察环境后迅速执行核心动作"

    if len(prompt) >= 90 and ("镜头" in prompt or "运镜" in prompt):
        if "时长" in prompt or "秒" in prompt:
            return prompt
        return f"总时长约{duration}秒。{prompt}。"

    first_end = max(1, round(duration * 0.3))
    second_end = max(first_end + 1, round(duration * 0.75))
    second_end = min(second_end, duration - 1) if duration > 2 else second_end

    return (
        f"总时长约{duration}秒。"
        f"第1段(0-{first_end}秒)：{prompt}，先给出明确起势与主体站位，镜头中景稳定进入。"
        f"第2段({first_end}-{second_end}秒)：动作强度逐步提升并出现转折，加入前移或侧向跟拍运镜，突出关键动作细节。"
        f"第3段({second_end}-{duration}秒)：动作完成并自然收束，镜头轻微拉近后停留0.5秒，给到结果态与情绪余韵。"
    )

GRSAI_VIDEO_MODELS: list[dict[str, str]] = [
    {"id": "veo3.1-fast", "label": "veo3.1-fast", "description": "速度优先，适合快速迭代"},
    {"id": "veo3.1-pro", "label": "veo3.1-pro", "description": "质量优先，适合最终成片"},
]


def resolve_video_provider(provider: str | None = None) -> str:
    resolved_provider = (provider or settings.VIDEO_PROVIDER or "grsai").strip().lower()
    if resolved_provider not in {"grsai", "ark"}:
        raise ValueError(f"Unsupported video provider: {resolved_provider}")
    return resolved_provider


def resolve_video_model(provider: str | None = None, model: str | None = None) -> str:
    resolved_provider = resolve_video_provider(provider)
    if model and model.strip():
        return model.strip()

    if resolved_provider == "grsai":
        return settings.GRSAI_VIDEO_MODEL

    return settings.ARK_VIDEO_MODEL


def resolve_video_aspect_ratio(aspect_ratio: str | None = None) -> str:
    candidate = (aspect_ratio or settings.GRSAI_VIDEO_ASPECT_RATIO or settings.IMAGE_ASPECT_RATIO or "16:9").strip()
    if candidate not in {"16:9", "9:16", "1:1"}:
        return "16:9"
    return candidate


def get_video_provider_catalog() -> list[dict[str, Any]]:
    return [
        {
            "provider": "grsai",
            "label": "GRSAI Veo3",
            "description": "使用 /v1/video/veo 接口生成视频片段。",
            "default_model": settings.GRSAI_VIDEO_MODEL,
            "default_aspect_ratio": resolve_video_aspect_ratio(),
            "api_key_env": "GRSAI_API_KEY",
            "supported_aspect_ratios": ["16:9", "9:16", "1:1"],
            "models": GRSAI_VIDEO_MODELS,
        },
        {
            "provider": "ark",
            "label": "Volcengine Ark",
            "description": "使用方舟视频能力生成视频片段。",
            "default_model": settings.ARK_VIDEO_MODEL,
            "default_aspect_ratio": "16:9",
            "api_key_env": "ARK_API_KEY",
            "supported_aspect_ratios": ["16:9"],
            "models": [
                {
                    "id": settings.ARK_VIDEO_MODEL,
                    "label": settings.ARK_VIDEO_MODEL,
                    "description": "当前通过 ARK_VIDEO_MODEL 配置的视频模型。",
                }
            ],
        },
    ]


def create_video_generator(
    video_provider: str | None = None,
    model: str | None = None,
    aspect_ratio: str | None = None,
    webhook: str | None = None,
    shut_progress: bool | None = None,
) -> VideoGenerator:
    resolved_provider = resolve_video_provider(video_provider)
    resolved_model = resolve_video_model(resolved_provider, model)

    if resolved_provider == "grsai":
        return GrsaiVeoVideoGenerator(
            model=resolved_model,
            aspect_ratio=resolve_video_aspect_ratio(aspect_ratio),
            webhook=webhook,
            shut_progress=shut_progress,
        )

    return ArkVideoGenerator(model=resolved_model)


class GrsaiVeoVideoGenerator(VideoGenerator):
    def __init__(
        self,
        model: str | None = None,
        aspect_ratio: str | None = None,
        webhook: str | None = None,
        shut_progress: bool | None = None,
    ):
        if not settings.GRSAI_API_KEY:
            raise ValueError("GRSAI_API_KEY is not configured")

        self.api_key = settings.GRSAI_API_KEY
        self.base_url = settings.GRSAI_BASE_URL.rstrip("/")
        self.model = model or settings.GRSAI_VIDEO_MODEL
        self.aspect_ratio = resolve_video_aspect_ratio(aspect_ratio)
        self.webhook = settings.GRSAI_VIDEO_WEBHOOK if webhook is None else webhook
        self.shut_progress = settings.GRSAI_VIDEO_SHUT_PROGRESS if shut_progress is None else shut_progress

    async def generate_video(self, scene: Scene, image_path: str, output_dir: str) -> str:
        first_frame_url = await self._resolve_first_frame_url(image_path)
        motion_prompt = _resolve_motion_prompt(scene)
        payload = {
            "model": self.model,
            "prompt": motion_prompt,
            "firstFrameUrl": first_frame_url,
            "aspectRatio": self.aspect_ratio,
            "webHook": self.webhook,
            "shutProgress": self.shut_progress,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=60.0)

        video_url = None
        last_error: Exception | None = None
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout) as client:
                try:
                    submit_response = await client.post("/v1/video/veo", json=payload)
                    submit_response.raise_for_status()
                    submit_payload = submit_response.json()
                except Exception as exc:
                    last_error = RuntimeError(f"GRSAI veo submit failed: {exc}")
                    if attempt >= max_attempts:
                        raise last_error
                    await asyncio.sleep(1.5 * attempt)
                    continue

                task_state = self._extract_task_state(submit_payload)
                task_id = task_state.get("id")
                logger.info("GRSAI video submit accepted for scene=%s task_id=%s", scene.id, task_id)
                if task_id:
                    register_video_task(
                        output_dir=settings.OUTPUT_DIR,
                        session_dir=Path(output_dir).parent,
                        task_id=str(task_id),
                        scene_id=scene.id,
                        provider="grsai",
                    )

                try:
                    result_data = await self._wait_for_result(client, task_state, Path(output_dir).parent)
                    video_url = self._extract_video_url(result_data)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_attempts or not self._is_retryable_failure(exc):
                        raise
                    logger.warning(
                        "GRSAI video generation failed for scene %s (attempt %s/%s), retrying once: %s",
                        scene.id,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    await asyncio.sleep(2.0 * attempt)

        if not video_url:
            if last_error:
                raise RuntimeError(str(last_error))
            raise RuntimeError("GRSAI veo returned no video url")

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=180.0, write=60.0, pool=60.0)) as client:
                download_response = await client.get(video_url)
                download_response.raise_for_status()
                video_content = download_response.content
        except Exception as exc:
            raise RuntimeError(f"Failed to download generated video: {exc}") from exc

        output_path = Path(output_dir) / f"{scene.id}_video.mp4"
        async with aiofiles.open(output_path, "wb") as file_handle:
            await file_handle.write(video_content)

        return str(output_path)

    async def _resolve_first_frame_url(self, image_path: str) -> str:
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path

        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"First frame image does not exist: {image_path}")

        suffix = path.suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg"
        async with aiofiles.open(path, "rb") as file_handle:
            image_bytes = await file_handle.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{image_b64}"

    def _extract_task_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected GRSAI veo response: {payload}")

        code = payload.get("code")
        if code not in {0, None}:
            raise RuntimeError(payload.get("msg") or payload.get("error") or "GRSAI veo submit failed")

        data = payload.get("data") if isinstance(payload.get("data"), dict) else None
        if data:
            if data.get("status") == "succeeded" and data.get("url"):
                return data
            if data.get("id"):
                return data

        if payload.get("status") == "succeeded" and payload.get("url"):
            return payload
        if payload.get("id"):
            return payload

        raise RuntimeError(f"Missing task id in GRSAI veo response: {payload}")

    async def _wait_for_result(
        self,
        client: httpx.AsyncClient,
        task_state: dict[str, Any],
        session_dir: Path,
    ) -> dict[str, Any]:
        status = task_state.get("status")
        if status == "succeeded":
            return task_state
        if status == "failed":
            raise RuntimeError(self._format_failure_message(task_state))

        task_id = task_state.get("id")
        if not task_id:
            raise RuntimeError(f"Missing task id in state: {task_state}")

        max_wait = max(settings.VIDEO_GENERATION_TIMEOUT_SECONDS, 1)
        start = asyncio.get_running_loop().time()

        while True:
            if asyncio.get_running_loop().time() - start > max_wait:
                raise TimeoutError(f"Video generation timeout after {max_wait}s (task={task_id})")

            await asyncio.sleep(settings.GRSAI_RESULT_POLL_INTERVAL_SECONDS)
            poll_payload = {
                "id": task_id,
                "shutProgress": self.shut_progress,
            }

            try:
                result_response = await client.post("/v1/draw/result", json=poll_payload)
                result_response.raise_for_status()
                result_payload = result_response.json()
            except Exception as exc:
                raise RuntimeError(f"GRSAI veo polling failed: {exc}") from exc

            result_code = result_payload.get("code") if isinstance(result_payload, dict) else None
            if result_code == -22:
                logger.info("GRSAI veo task %s is still processing", task_id)
                continue
            if result_code not in {0, None}:
                raise RuntimeError(
                    "GRSAI veo result query failed "
                    f"(task_id={task_id}, code={result_code}, payload={_compact_payload(result_payload)})"
                )

            result_data = result_payload.get("data") if isinstance(result_payload.get("data"), dict) else result_payload
            if not isinstance(result_data, dict):
                raise RuntimeError(f"Unexpected GRSAI veo result payload: {result_payload}")

            status = result_data.get("status")
            if session_dir.exists():
                update_video_task(
                    session_dir=session_dir,
                    task_id=str(task_id),
                    status=str(status) if status else None,
                    progress=result_data.get("progress") if isinstance(result_data.get("progress"), int) else None,
                    url=result_data.get("url") if isinstance(result_data.get("url"), str) else None,
                    fail_reason=result_data.get("fail_reason") if isinstance(result_data.get("fail_reason"), str) else None,
                )

            if status == "succeeded" or (result_data.get("url") and status is None):
                return result_data
            if status == "failed":
                raise RuntimeError(
                    self._format_failure_message(
                        result_data,
                        task_id=str(task_id),
                        raw_payload=result_payload,
                    )
                )

    def _extract_video_url(self, result_data: dict[str, Any]) -> str:
        video_url = result_data.get("url")
        if video_url:
            return str(video_url)

        results = result_data.get("results") or []
        if results and isinstance(results[0], dict) and results[0].get("url"):
            return str(results[0]["url"])

        raise RuntimeError(f"GRSAI veo returned no video url: {result_data}")

    def _format_failure_message(
        self,
        result_data: dict[str, Any],
        task_id: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> str:
        failure_reason = result_data.get("fail_reason") or result_data.get("failure_reason") or "unknown"
        status = result_data.get("status")
        progress = result_data.get("progress")
        resolved_task_id = task_id or result_data.get("id")
        msg = result_data.get("msg") or result_data.get("error")

        details: list[str] = []
        if resolved_task_id:
            details.append(f"id={resolved_task_id}")
        if status:
            details.append(f"status={status}")
        if progress is not None:
            details.append(f"progress={progress}")
        if msg:
            details.append(f"msg={msg}")
        if raw_payload is not None:
            details.append(f"payload={_compact_payload(raw_payload)}")

        details_str = f" ({', '.join(details)})" if details else ""
        return f"GRSAI veo generation failed: {failure_reason}{details_str}"

    def _is_retryable_failure(self, error: Exception) -> bool:
        text = str(error).lower()
        return "failed: error" in text or "failed: unknown" in text or "timeout" in text


class ArkVideoGenerator(VideoGenerator):
    def __init__(self, model: str | None = None):
        if not settings.ARK_API_KEY:
            raise ValueError("ARK_API_KEY is not configured")
        self.client = Ark(
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL,
        )
        self.model = model or settings.ARK_VIDEO_MODEL

    async def generate_video(self, scene: Scene, image_path: str, output_dir: str) -> str:
        async with aiofiles.open(image_path, "rb") as file_handle:
            img_data = await file_handle.read()
            img_b64 = base64.b64encode(img_data).decode("utf-8")

        ext = Path(image_path).suffix.lstrip(".") or "png"
        img_data_uri = f"data:image/{ext};base64,{img_b64}"
        loop = asyncio.get_running_loop()

        def _submit_task():
            duration = getattr(scene, "duration", 5)
            motion_prompt = _resolve_motion_prompt(scene)
            prompt_with_duration = f"{motion_prompt} --duration {duration}"
            return self.client.content_generation.tasks.create(
                model=self.model,
                content=[
                    {
                        "type": "text",
                        "text": prompt_with_duration,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_uri},
                    },
                ],
                extra_body={"watermark": False},
            )

        try:
            create_result = await loop.run_in_executor(None, _submit_task)
        except Exception as exc:
            raise RuntimeError(f"Ark API submit failed: {exc}") from exc

        task_id = create_result.id
        max_wait_time = max(settings.VIDEO_GENERATION_TIMEOUT_SECONDS, 1)
        start = asyncio.get_running_loop().time()

        while True:
            if asyncio.get_running_loop().time() - start > max_wait_time:
                raise TimeoutError(f"Ark video generation timeout after {max_wait_time}s")

            await asyncio.sleep(10)

            def _get_status():
                return self.client.content_generation.tasks.get(task_id=task_id)

            try:
                result = await loop.run_in_executor(None, _get_status)
            except Exception:
                continue

            if result.status == "succeeded":
                try:
                    video_url = result.content.video_url
                except Exception as exc:
                    raise RuntimeError(f"Failed to extract Ark video url: {exc}") from exc
                break
            if result.status == "failed":
                error_msg = getattr(result, "error", "unknown error")
                raise RuntimeError(f"Ark video generation failed: {error_msg}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            download_response = await client.get(video_url)
            download_response.raise_for_status()
            video_content = download_response.content

        output_path = Path(output_dir) / f"{scene.id}_video.mp4"
        async with aiofiles.open(output_path, "wb") as file_handle:
            await file_handle.write(video_content)

        return str(output_path)
