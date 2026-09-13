import asyncio
import base64
import json
import logging
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import httpx
from volcenginesdkarkruntime import Ark

from src.video_workflow.config import settings
from src.video_workflow.generators.base import ImageGenerator
from src.video_workflow.types import Scene

logger = logging.getLogger(__name__)


class ImageDownloadError(RuntimeError):
    """The provider already generated an image; retry retrieval, not generation."""


class ImageGenerationError(RuntimeError):
    """The provider explicitly reported terminal generation failure."""


async def download_generated_image(url: str, output_dir: str, scene_id: int = 1) -> str:
    """Download with bounded retries, redirects and a direct-route fallback.

    Only the unauthenticated result download uses the fallback. API credentials
    are never attached to CDN requests and TLS verification remains enabled.
    """
    if urlparse(url).scheme not in {"http", "https"}:
        raise ImageDownloadError("图像结果地址格式错误")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    last_error = ""
    timeout = httpx.Timeout(connect=10, read=60, write=20, pool=20)
    for attempt, trust_env in enumerate((True, False, False)):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=trust_env) as client:
                response = await client.get(url)
                response.raise_for_status()
            content = response.content
            if not (content.startswith(b"\x89PNG\r\n\x1a\n") or content.startswith(b"\xff\xd8\xff")
                    or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")):
                raise ValueError("返回内容不是有效的 PNG/JPEG/WebP 图像")
            path = folder / f"{scene_id}_keyframe{_generated_image_suffix(content)}"
            temporary = path.with_suffix(path.suffix + ".part")
            async with aiofiles.open(temporary, "wb") as handle:
                await handle.write(content)
            temporary.replace(path)
            return str(path)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                last_error += f" HTTP {exc.response.status_code}"
            logger.warning("Image download attempt %s failed: host=%s error=%s", attempt + 1, urlparse(url).hostname, last_error)
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise ImageDownloadError(f"服务商已出图，但图片下载失败（{last_error}）；结果地址已保留，请重试下载，无需重新生成")

GRSAI_MODELS: list[dict[str, str]] = [
    {"id": "gpt-image-2", "label": "gpt-image-2", "description": "GPT Image 2，适合中文提示词与角色构图"},
    {"id": "gpt-image-2-vip", "label": "gpt-image-2-vip", "description": "GPT Image 2 高质量版本，支持更高分辨率"},
    {"id": "gpt-image-2.5", "label": "gpt-image-2.5", "description": "GPT Image 2.5，1K 版本，适合批量分镜"},
    {"id": "gpt-image-2.5-sunburst", "label": "gpt-image-2.5-sunburst", "description": "GPT Image 2.5 Sunburst，支持 1K/2K/4K 与质量参数"},
    {"id": "gpt-image-2.5-flare", "label": "gpt-image-2.5-flare", "description": "GPT Image 2.5 Flare，支持 1K/2K/4K 与质量参数"},
    {"id": "nano-banana-fast", "label": "nano-banana-fast", "description": "速度优先，适合批量分镜快速出图"},
    {"id": "nano-banana", "label": "nano-banana", "description": "平衡质量与速度的标准模型"},
    {"id": "nano-banana-2", "label": "nano-banana-2", "description": "新版本 Nano Banana，支持更高规格参数"},
    {"id": "nano-banana-pro", "label": "nano-banana-pro", "description": "质量优先，适合关键画面"},
    {"id": "nano-banana-pro-vt", "label": "nano-banana-pro-vt", "description": "高质量 Pro 变体"},
    {"id": "nano-banana-pro-cl", "label": "nano-banana-pro-cl", "description": "高质量 Pro 变体"},
    {"id": "nano-banana-pro-vip", "label": "nano-banana-pro-vip", "description": "支持更高规格输出的 VIP 版本"},
    {"id": "nano-banana-pro-4k-vip", "label": "nano-banana-pro-4k-vip", "description": "4K 输出专用 VIP 版本"},
]

GRSAI_IMAGE_SIZE_SUPPORTED_MODELS = {
    "nano-banana-2",
    "nano-banana-pro",
    "nano-banana-pro-vt",
    "nano-banana-pro-cl",
    "nano-banana-pro-vip",
    "nano-banana-pro-4k-vip",
}

GRSAI_GPT_IMAGE_25_MODELS = {
    "gpt-image-2.5",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
}
GRSAI_GPT_IMAGE_MODELS = {
    "gpt-image-2",
    "gpt-image-2-vip",
    *GRSAI_GPT_IMAGE_25_MODELS,
}


def _is_grsai_gpt_image_model(model: str) -> bool:
    """Keep manually entered future ``gpt-image-*`` IDs on the GPT route."""
    normalized = model.strip().lower()
    return normalized in GRSAI_GPT_IMAGE_MODELS or normalized.startswith("gpt-image-")


GRSAI_SUCCESS_STATUSES = {"success", "succeeded", "completed", "complete", "done", "finished"}
GRSAI_FAILURE_STATUSES = {"failed", "failure", "error", "cancelled", "canceled"}

GRSAI_GPT_IMAGE_ASPECT_RATIOS = {
    "1:1": "1024x1024",
    "16:9": "1672x941",
    "9:16": "941x1672",
    "4:3": "1443x1090",
    "3:4": "1090x1443",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "5:4": "1408x1120",
    "4:5": "1120x1408",
    "21:9": "1920x832",
}

# GRSAI's 2.5 endpoints use concrete pixel values for the 1K/2K/4K variants.
# Keep the legacy GPT Image 2 mapping above unchanged for existing projects.
GRSAI_GPT_IMAGE_25_ASPECT_RATIOS = {
    "1K": {
        "auto": "1024x1024",
        "1:1": "1024x1024",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "4:3": "1152x864",
        "3:4": "864x1152",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
        "5:4": "1120x896",
        "4:5": "896x1120",
        "21:9": "1456x624",
        "9:21": "624x1456",
        "1:2": "768x1536",
        "2:1": "1536x768",
        "1:3": "688x2048",
        "3:1": "2048x688",
    },
    "2K": {
        "auto": "2048x2048",
        "1:1": "2048x2048",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "4:3": "2304x1728",
        "3:4": "1728x2304",
        "3:2": "2048x1360",
        "2:3": "1360x2048",
        "5:4": "2240x1792",
        "4:5": "1792x2240",
        "21:9": "2912x1248",
        "9:21": "1248x2912",
        "1:2": "1536x3072",
        "2:1": "3072x1536",
    },
    "4K": {
        "auto": "2880x2880",
        "1:1": "2880x2880",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "4:3": "3264x2448",
        "3:4": "2448x3264",
        "3:2": "3504x2336",
        "2:3": "2336x3504",
        "5:4": "3200x2560",
        "4:5": "2560x3200",
        "21:9": "3840x1648",
        "9:21": "1648x3840",
        "1:2": "1920x3840",
        "2:1": "3840x1920",
        "1:3": "1280x3840",
        "3:1": "3840x1280",
    },
}


def grsai_image_endpoint(model: str) -> str:
    """Use GRSAI's current GPT endpoint while preserving the Nano route."""
    return "/v1/api/generate" if _is_grsai_gpt_image_model(model) else "/v1/draw/nano-banana"


def grsai_image_aspect_ratio(model: str, aspect_ratio: str | None) -> str:
    resolved = aspect_ratio or settings.IMAGE_ASPECT_RATIO or "auto"
    normalized_model = model.strip().lower()
    if normalized_model in GRSAI_GPT_IMAGE_25_MODELS:
        # The base 2.5 model is 1K-only.  Sunburst/Flare honour the configured
        # size, while custom pixel values remain available for advanced users.
        configured_size = settings.GRSAI_IMAGE_SIZE.upper().strip()
        size = "1K" if normalized_model == "gpt-image-2.5" else configured_size
        size_map = GRSAI_GPT_IMAGE_25_ASPECT_RATIOS.get(
            size,
            GRSAI_GPT_IMAGE_25_ASPECT_RATIOS["1K"],
        )
        return size_map.get(resolved, resolved)
    if _is_grsai_gpt_image_model(model):
        return GRSAI_GPT_IMAGE_ASPECT_RATIOS.get(resolved, resolved)
    return resolved


def resolve_image_provider(provider: str | None = None) -> str:
    resolved_provider = (provider or settings.IMAGE_PROVIDER or "ark").strip().lower()
    if resolved_provider not in {"ark", "grsai"}:
        raise ValueError(f"Unsupported image provider: {resolved_provider}")
    return resolved_provider


def resolve_image_model(provider: str | None = None, model: str | None = None) -> str:
    resolved_provider = resolve_image_provider(provider)
    if model and model.strip():
        return model.strip()

    if resolved_provider == "grsai":
        return settings.GRSAI_IMAGE_MODEL

    return settings.ARK_IMAGE_MODEL


def get_image_provider_catalog() -> list[dict[str, Any]]:
    return [
        {
            "provider": "ark",
            "label": "Volcengine Ark",
            "description": "使用方舟图像能力生成分镜首帧。",
            "default_model": settings.ARK_IMAGE_MODEL,
            "api_key_env": "ARK_API_KEY",
            "models": [
                {
                    "id": settings.ARK_IMAGE_MODEL,
                    "label": settings.ARK_IMAGE_MODEL,
                    "description": "当前通过 ARK_IMAGE_MODEL 配置的方舟图像模型。",
                }
            ],
        },
        {
            "provider": "grsai",
            "label": "GRSAI 图像模型",
            "description": "使用 GRSAI 的 GPT Image 与 Nano Banana 系列接口生成图像。",
            "default_model": settings.GRSAI_IMAGE_MODEL,
            "api_key_env": "GRSAI_API_KEY",
            "models": GRSAI_MODELS,
        },
    ]


def create_image_generator(image_provider: str | None = None, model: str | None = None) -> ImageGenerator:
    resolved_provider = resolve_image_provider(image_provider)
    resolved_model = resolve_image_model(resolved_provider, model)

    if resolved_provider == "grsai":
        return GrsaiImageGenerator(model=resolved_model)

    return ArkImageGenerator(model=resolved_model)


def _build_image_prompt(
    scene: Scene,
    character_description: str | None = None,
    image_style: str | None = None,
) -> str:
    prompt = scene.visual_prompt
    # ``None`` means "use the process-level default" while an explicit empty
    # string means "this shot has no characters".  Using ``or`` here made
    # title cards and empty establishing shots unexpectedly inherit the global
    # character bible.
    resolved_character_description = (
        settings.CHARACTER_DESCRIPTION
        if character_description is None
        else character_description
    )
    resolved_image_style = image_style if image_style is not None else settings.IMAGE_STYLE

    if resolved_character_description:
        prompt = f"【角色特征】{resolved_character_description.rstrip('。 ')}。\n【场景描述】{prompt}"

    if resolved_image_style:
        prompt = f"{prompt.rstrip('。 ')}。\n【画面风格】{resolved_image_style}"

    return prompt


def _collect_reference_sources(reference_image_path: str | None) -> list[str]:
    if not reference_image_path:
        return []

    collected: list[str] = []
    raw_sources = [segment.strip() for segment in reference_image_path.split(",") if segment.strip()]

    for source in raw_sources:
        if source.startswith(("http://", "https://", "data:")):
            collected.append(source)
            continue

        path = Path(source)
        if path.is_dir():
            image_files = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg"))
            for image_file in image_files:
                collected.append(str(image_file))
                if len(collected) >= 10:
                    return collected
            continue

        if path.exists():
            collected.append(str(path))
            if len(collected) >= 10:
                return collected
            continue

        logger.warning("Reference image path does not exist: %s", source)

    return collected[:10]


def _reference_groups(content_path: str | None, style_path: str | None) -> tuple[list[str], list[str]]:
    """Keep content/identity first and reserve slots for genuine style images."""
    content = list(dict.fromkeys(_collect_reference_sources(content_path)))
    style = [source for source in dict.fromkeys(_collect_reference_sources(style_path)) if source not in content][:3]
    content = content[: 10 - len(style)]
    return content, style


def _style_reference_instruction(content_count: int, style_count: int) -> str:
    if not style_count:
        return ""
    start = content_count + 1
    end = content_count + style_count
    label = f"第 {start} 张" if start == end else f"第 {start}–{end} 张"
    content_rule = f"前 {content_count} 张用于主体身份、场景内容或布局；" if content_count else ""
    return (
        f"\n【输入图片分工】{content_rule}{label}仅作为画风参考。"
        "继承其绘画媒介、笔触线条、上色方式、配色关系和材质感；"
        "不要复制风格图里的人物身份、服装、道具、场景布局或构图。"
        "生成内容以本条场景与角色描述为准。"
    )


def _encode_local_image(path: Path) -> tuple[str | None, str | None]:
    try:
        with open(path, "rb") as file_handle:
            image_bytes = file_handle.read()
    except Exception as exc:
        logger.warning("Failed to load reference image %s: %s", path, exc)
        return None, None

    suffix = path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return mime_type, image_b64


def _generated_image_suffix(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def _normalized_grsai_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _decode_grsai_response(response: httpx.Response, operation: str) -> dict[str, Any]:
    """Decode regular JSON plus GRSAI event-stream/JSONL responses."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, ValueError):
        pass

    body = response.text.lstrip("\ufeff").strip()
    if not body:
        content_type = response.headers.get("content-type", "unknown")
        raise RuntimeError(
            f"GRSAI {operation} returned an empty response "
            f"(status={response.status_code}, content_type={content_type})"
        )

    if body.startswith(("http://", "https://")):
        return {"status": "succeeded", "results": [{"url": body}]}

    # The GPT Image endpoint may emit multiple ``data: {...}`` events. Decode
    # each top-level JSON value and prefer the newest event containing a URL.
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(body):
        object_start = min(
            (index for index in (body.find("{", cursor), body.find("[", cursor)) if index >= 0),
            default=-1,
        )
        if object_start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(body[object_start:])
        except json.JSONDecodeError:
            cursor = object_start + 1
            continue
        if isinstance(value, dict):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.append({"status": "succeeded", "results": value})
        cursor = object_start + consumed

    if candidates:
        for candidate in reversed(candidates):
            if GrsaiImageGenerator._extract_result_url(candidate, required=False):
                return candidate
        return candidates[-1]

    content_type = response.headers.get("content-type", "unknown")
    preview = body[:240].replace("\n", " ")
    raise RuntimeError(
        f"GRSAI {operation} returned an unrecognized response "
        f"(status={response.status_code}, content_type={content_type}, body={preview})"
    )


class ArkImageGenerator(ImageGenerator):
    def __init__(self, model: str | None = None):
        if not settings.ARK_API_KEY:
            raise ValueError("ARK_API_KEY is not configured")
        self.client = Ark(
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL,
        )
        self.model = model or settings.ARK_IMAGE_MODEL

        seed_str = settings.IMAGE_SEED
        if seed_str and seed_str.strip():
            self._session_seed = int(seed_str.strip())
        else:
            self._session_seed = random.randint(1, 999999999)
        logger.info("Image seed initialized: %s", self._session_seed)

        self.aspect_ratio_sizes = {
            "1:1": "2048x2048",
            "16:9": "2560x1440",
            "9:16": "1440x2560",
            "4:3": "2304x1728",
            "3:4": "1728x2304",
        }

    async def generate_image(
        self,
        scene: Scene,
        output_dir: str,
        reference_image_path: str | None = None,
        seed: int | None = None,
        character_description: str | None = None,
        image_style: str | None = None,
        aspect_ratio: str | None = None,
        style_reference_image_path: str | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        size = self.aspect_ratio_sizes.get(aspect_ratio or settings.IMAGE_ASPECT_RATIO, "2048x1800")
        content_sources, style_sources = _reference_groups(reference_image_path, style_reference_image_path)
        prompt = _build_image_prompt(scene, character_description, image_style)
        prompt += _style_reference_instruction(len(content_sources), len(style_sources))

        def _generate():
            params = {
                "model": self.model,
                "prompt": prompt,
                "size": size,
                "response_format": "b64_json",
            }

            current_seed = seed if seed is not None else self._session_seed
            extra_body = {
                "seed": current_seed,
                "watermark": False,
            }

            if content_sources or style_sources:
                ref_images = self._load_reference_images(",".join([*content_sources, *style_sources]))
                if ref_images:
                    extra_body["reference_images"] = ref_images
                    extra_body["reference_weight"] = settings.IMAGE_STYLE_WEIGHT
                    if content_sources:
                        extra_body["reference_mode"] = "character"
                    logger.info(
                        "Using %s reference images with weight %.2f for Ark generation",
                        len(ref_images),
                        settings.IMAGE_STYLE_WEIGHT,
                    )

            params["extra_body"] = extra_body
            return self.client.images.generate(**params)

        try:
            response = await loop.run_in_executor(None, _generate)
        except Exception as exc:
            raise RuntimeError(f"Ark image generation failed: {exc}") from exc

        try:
            b64_data = response.data[0].b64_json
            image_content = base64.b64decode(b64_data)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ark image response: {exc}") from exc

        filepath = Path(output_dir) / f"{scene.id}_keyframe{_generated_image_suffix(image_content)}"
        async with aiofiles.open(filepath, "wb") as file_handle:
            await file_handle.write(image_content)

        return str(filepath)

    def _load_reference_images(self, reference_image_path: str) -> list[dict[str, str]]:
        ref_images: list[dict[str, str]] = []

        for source in _collect_reference_sources(reference_image_path):
            if source.startswith(("http://", "https://", "data:")):
                ref_images.append({"url": source, "role": "character"})
                continue

            mime_type, image_b64 = _encode_local_image(Path(source))
            if image_b64:
                ref_images.append(
                    {
                        "url": f"data:{mime_type};base64,{image_b64}",
                        "role": "character",
                    }
                )

        return ref_images


class GrsaiImageGenerator(ImageGenerator):
    def __init__(self, model: str | None = None):
        if not settings.GRSAI_API_KEY:
            raise ValueError("GRSAI_API_KEY is not configured")

        self.api_key = settings.GRSAI_API_KEY
        self.base_url = settings.GRSAI_BASE_URL.rstrip("/")
        self.model = (model or settings.GRSAI_IMAGE_MODEL).strip()

    async def generate_image(
        self,
        scene: Scene,
        output_dir: str,
        reference_image_path: str | None = None,
        seed: int | None = None,
        character_description: str | None = None,
        image_style: str | None = None,
        aspect_ratio: str | None = None,
        style_reference_image_path: str | None = None,
    ) -> str:
        del seed

        resolved_aspect_ratio = grsai_image_aspect_ratio(self.model, aspect_ratio)
        content_sources, style_sources = _reference_groups(reference_image_path, style_reference_image_path)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": _build_image_prompt(scene, character_description, image_style)
            + _style_reference_instruction(len(content_sources), len(style_sources)),
            "aspectRatio": resolved_aspect_ratio,
        }

        is_gpt_image = _is_grsai_gpt_image_model(self.model)
        if is_gpt_image:
            payload["replyType"] = "json"
        else:
            payload["shutProgress"] = True

        image_size = self._resolve_image_size() if not _is_grsai_gpt_image_model(self.model) else None
        if image_size:
            payload["imageSize"] = image_size

        reference_urls = self._load_reference_inputs(",".join([*content_sources, *style_sources]))
        if reference_urls:
            payload["images" if is_gpt_image else "urls"] = reference_urls

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = grsai_image_endpoint(self.model)
        timeout = httpx.Timeout(
            connect=20.0,
            read=float(settings.IMAGE_GENERATION_TIMEOUT_SECONDS),
            write=60.0,
            pool=60.0,
        )
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout) as client:
            try:
                logger.info(
                    "GRSAI image request: model=%s endpoint=%s aspect_ratio=%s references=%s",
                    self.model,
                    endpoint,
                    resolved_aspect_ratio,
                    len(reference_urls),
                )
                submit_response = await client.post(endpoint, json=payload)
                submit_response.raise_for_status()
                submit_data = _decode_grsai_response(submit_response, "image submit")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                body_preview = ""
                if exc.response is not None:
                    response_text = exc.response.text or ""
                    if response_text:
                        body_preview = response_text[:300]
                details = f"status={status_code}"
                if body_preview:
                    details += f", body={body_preview}"
                raise RuntimeError(f"GRSAI image submit failed ({details})") from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    f"GRSAI image submit failed (timeout after {settings.IMAGE_GENERATION_TIMEOUT_SECONDS}s): {type(exc).__name__}"
                ) from exc
            except httpx.RequestError as exc:
                request_url = str(exc.request.url) if exc.request is not None else self.base_url
                raise RuntimeError(
                    f"GRSAI image submit failed (request_error={type(exc).__name__}, url={request_url}): {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"GRSAI image submit failed ({type(exc).__name__}): {exc}") from exc

            task_state = self._extract_task_state(submit_data)
            self._save_result_checkpoint(output_dir, {"model": self.model, "scene_id": scene.id, "task_state": task_state})
            result_data = await self._wait_for_result(client, task_state)
            result_url = self._extract_result_url(result_data)
            self._save_result_checkpoint(output_dir, {"model": self.model, "scene_id": scene.id, "task_state": task_state, "result_url": result_url})
        return await download_generated_image(result_url, output_dir, scene.id)

    @staticmethod
    def _save_result_checkpoint(output_dir: str, data: dict[str, Any]) -> None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "grsai_result.json.part"
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(folder / "grsai_result.json")

    @staticmethod
    async def resume_image(output_dir: str) -> str:
        """Resume a recorded result; this method never submits a generation."""
        checkpoint = Path(output_dir) / "grsai_result.json"
        if not checkpoint.is_file():
            raise ValueError("没有已保存的服务商结果，请从服务商后台下载图片后上传绑定")
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        result_url = data.get("result_url")
        if not result_url:
            generator = GrsaiImageGenerator(model=data.get("model"))
            async with httpx.AsyncClient(
                base_url=generator.base_url,
                headers={"Authorization": f"Bearer {generator.api_key}"},
                timeout=60,
            ) as client:
                result = await generator._wait_for_result(client, data["task_state"])
            result_url = generator._extract_result_url(result)
            data["result_url"] = result_url
            generator._save_result_checkpoint(output_dir, data)
        return await download_generated_image(result_url, output_dir, int(data.get("scene_id", 1)))

    def _resolve_image_size(self) -> str | None:
        if self.model not in GRSAI_IMAGE_SIZE_SUPPORTED_MODELS:
            return None

        configured_size = settings.GRSAI_IMAGE_SIZE.upper().strip()
        if self.model == "nano-banana-pro-4k-vip":
            return "4K"

        if self.model == "nano-banana-pro-vip" and configured_size not in {"1K", "2K"}:
            return "1K"

        if configured_size not in {"1K", "2K", "4K"}:
            return "1K"

        return configured_size

    def _load_reference_inputs(self, reference_image_path: str | None) -> list[str]:
        reference_inputs: list[str] = []
        for source in _collect_reference_sources(reference_image_path):
            if source.startswith(("http://", "https://", "data:")):
                reference_inputs.append(source)
                continue

            _, image_b64 = _encode_local_image(Path(source))
            if image_b64:
                reference_inputs.append(image_b64)

        return reference_inputs

    def _extract_task_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "code" in payload and payload.get("code") != 0:
            raise RuntimeError(payload.get("error") or payload.get("msg") or "GRSAI request failed")

        if "status" in payload:
            return payload

        if self._extract_result_url(payload, required=False):
            return {**payload, "status": "succeeded"}

        data = payload.get("data")
        if isinstance(data, dict):
            if data.get("status") or data.get("id") or data.get("task_id"):
                return data

        raise RuntimeError(f"Unexpected GRSAI response: {payload}")

    async def _wait_for_result(self, client: httpx.AsyncClient, task_state: dict[str, Any]) -> dict[str, Any]:
        status = _normalized_grsai_status(task_state.get("status"))
        if self._extract_result_url(task_state, required=False) or status in GRSAI_SUCCESS_STATUSES:
            return task_state
        if status in GRSAI_FAILURE_STATUSES:
            raise ImageGenerationError(self._format_failure_message(task_state))

        task_id = task_state.get("task_id") or task_state.get("id")
        if not task_id:
            raise RuntimeError(f"Missing GRSAI task id in response: {task_state}")

        deadline = asyncio.get_running_loop().time() + settings.IMAGE_GENERATION_TIMEOUT_SECONDS
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("GRSAI 结果查询超时；任务记录已保留，请稍后取回结果，不必重新生图")
            await asyncio.sleep(settings.GRSAI_RESULT_POLL_INTERVAL_SECONDS)

            try:
                if _is_grsai_gpt_image_model(self.model):
                    result_response = await client.get("/v1/api/result", params={"id": task_id})
                else:
                    result_response = await client.post("/v1/draw/result", json={"id": task_id})
                result_response.raise_for_status()
                result_payload = _decode_grsai_response(result_response, "result polling")
            except Exception as exc:
                raise RuntimeError(f"GRSAI result polling failed: {exc}") from exc

            result_code = result_payload.get("code")
            if result_code == -22:
                logger.info("GRSAI task %s is not ready yet, retrying...", task_id)
                continue
            if "code" in result_payload and result_code not in {0, None}:
                raise RuntimeError(result_payload.get("msg") or "GRSAI result query failed")

            result_data = result_payload.get("data") if isinstance(result_payload.get("data"), dict) else result_payload
            status = _normalized_grsai_status(result_data.get("status"))
            if self._extract_result_url(result_data, required=False) or status in GRSAI_SUCCESS_STATUSES:
                return result_data
            if status in GRSAI_FAILURE_STATUSES:
                raise ImageGenerationError(self._format_failure_message(result_data))

    @staticmethod
    def _extract_result_url(result_data: dict[str, Any], required: bool = True) -> str:
        if isinstance(result_data.get("url"), str) and result_data["url"]:
            return result_data["url"]
        for key in ("results", "result", "output", "data"):
            nested = result_data.get(key)
            if isinstance(nested, dict):
                nested_url = GrsaiImageGenerator._extract_result_url(nested, required=False)
                if nested_url:
                    return nested_url
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        nested_url = GrsaiImageGenerator._extract_result_url(item, required=False)
                        if nested_url:
                            return nested_url
                    elif isinstance(item, str) and item.startswith(("http://", "https://")):
                        return item
        if required:
            raise RuntimeError(f"GRSAI returned no image url: {result_data}")
        return ""

    def _format_failure_message(self, result_data: dict[str, Any]) -> str:
        failure_reason = result_data.get("failure_reason") or "error"
        error_detail = result_data.get("error") or "Unknown error"
        return f"GRSAI image generation failed ({failure_reason}): {error_detail}"
