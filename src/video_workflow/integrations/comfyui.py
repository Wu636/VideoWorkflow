from __future__ import annotations

import asyncio
import copy
import json
import math
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode, urlparse
from uuid import uuid4

import httpx

from src.video_workflow.config import settings
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.domain import Asset, AssetType, GenerationMode

ProgressCallback = Callable[[float, str], Awaitable[None]]

H3_SCHEDULERS = {
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
}


class ComfyUIError(RuntimeError):
    pass


@dataclass(slots=True)
class H3WorkflowRequest:
    mode: GenerationMode
    prompt: str
    width: int
    height: int
    frames: int
    seed: int
    output_prefix: str
    first_frame_filename: str | None = None
    last_frame_filename: str | None = None
    reference_images: list[str] = field(default_factory=list)
    reference_videos: list[str] = field(default_factory=list)
    reference_audios: list[str] = field(default_factory=list)
    ref_image_size: str = "match"
    turbo: bool = True
    steps: int = 4
    scheduler: str = "simple"
    denoise: float = 1.0
    lora_strength: float = 1.0
    low_vram: bool = False
    shift_video: float = 12.0
    shift_audio: float = 3.0


def h3_frames_for_seconds(seconds: float) -> int:
    """Return a trained-range H3 frame count on the 17k+5 grid at 24fps."""
    requested = max(5.0, min(15.0, float(seconds))) * 24.0
    frames = 5 + 17 * math.ceil((requested - 5) / 17)
    return max(124, min(362, frames))


def seconds_for_h3_frames(frames: int) -> float:
    return frames / 24.0


class H3WorkflowBuilder:
    def __init__(self, i2v_path: Path | None = None, r2v_path: Path | None = None):
        bundled_dir = Path(__file__).resolve().parents[1] / "workflows"
        self.i2v_path = i2v_path or settings.COMFYUI_H3_I2V_WORKFLOW or bundled_dir / "minimax_h3_i2v_turbo_api.json"
        self.r2v_path = r2v_path or settings.COMFYUI_H3_R2V_WORKFLOW or bundled_dir / "minimax_h3_r2v_turbo_api.json"

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        resolved = Path(path)
        if not resolved.exists():
            raise ComfyUIError(f"H3 workflow not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ComfyUIError(f"Invalid H3 workflow: {resolved}")
        return data

    def build(self, request: H3WorkflowRequest) -> dict[str, Any]:
        if request.mode == GenerationMode.I2V:
            workflow = copy.deepcopy(self._load(Path(self.i2v_path)))
            if not request.first_frame_filename:
                raise ComfyUIError("I2V requires an approved first-frame image")
            workflow["7"]["inputs"]["image"] = request.first_frame_filename
            if request.last_frame_filename:
                node_id = "18"
                workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": request.last_frame_filename}}
                workflow["8"]["inputs"]["last_frame"] = [node_id, 0]
        elif request.mode == GenerationMode.R2V:
            workflow = copy.deepcopy(self._load(Path(self.r2v_path)))
            self._attach_references(workflow, request)
        else:
            raise ComfyUIError(f"Unsupported H3 generation mode: {request.mode}")

        workflow["8"]["inputs"].update(
            prompt=request.prompt,
            width=max(32, round(request.width / 32) * 32),
            height=max(32, round(request.height / 32) * 32),
            length=request.frames,
        )
        workflow["11"]["inputs"]["noise_seed"] = request.seed
        # Match the current official ComfyUI H3 templates: both native and
        # Lightning/Turbo sampling use res_multistep.  The previous automation
        # used custom Turbo sampler + sigma-shift nodes, which did not match the
        # official canvas and produced noticeably less stable faces and limbs.
        model_source = ["1", 0]
        workflow["6"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
        workflow.pop("5", None)
        if request.turbo:
            workflow["17"]["class_type"] = "LoraLoaderModelOnly"
            workflow["17"]["inputs"] = {
                "model": ["1", 0],
                "lora_name": workflow["17"]["inputs"]["lora_name"],
                "strength_model": max(-10.0, min(10.0, float(request.lora_strength))),
            }
            model_source = ["17", 0]
        else:
            # Native 20-step is the official quality path.  It bypasses the
            # acceleration LoRA entirely and needs no additional model file.
            workflow.pop("17", None)
        workflow["9"]["inputs"]["model"] = model_source
        workflow["10"]["inputs"].update(
            model=model_source,
            scheduler=request.scheduler if request.scheduler in H3_SCHEDULERS else "simple",
            steps=max(1, min(100, int(request.steps))),
            denoise=max(0.0, min(1.0, float(request.denoise))),
        )
        workflow["16"]["inputs"]["filename_prefix"] = request.output_prefix
        return workflow

    @staticmethod
    def _attach_references(workflow: dict[str, Any], request: H3WorkflowRequest) -> None:
        inputs = workflow["8"]["inputs"]
        for key in list(inputs):
            if key != "ref_image_size" and key.startswith(("ref_image_", "ref_video_", "ref_audio_")):
                inputs.pop(key, None)
        inputs["ref_image_size"] = request.ref_image_size if request.ref_image_size in {"match", "max"} else "match"
        workflow.pop("7", None)

        for index, filename in enumerate(request.reference_images[:9], start=1):
            node_id = str(100 + index)
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            inputs[f"ref_image_{index}"] = [node_id, 0]

        for index, filename in enumerate(request.reference_videos[:3], start=1):
            load_id = str(120 + index * 2)
            split_id = str(121 + index * 2)
            workflow[load_id] = {"class_type": "LoadVideo", "inputs": {"file": filename}}
            workflow[split_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
            inputs[f"ref_video_{index}"] = [split_id, 0]
            inputs[f"ref_video_audio_{index}"] = [split_id, 1]

        for index, filename in enumerate(request.reference_audios[:3], start=1):
            node_id = str(140 + index)
            workflow[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": filename}}
            inputs[f"ref_audio_{index}"] = [node_id, 0]

        if not request.reference_images and not request.reference_videos and not request.reference_audios:
            raise ComfyUIError("R2V requires at least one reference image, video, or audio")

    def preflight_requirements(self) -> dict[str, list[str]]:
        workflows = [self._load(Path(self.i2v_path)), self._load(Path(self.r2v_path))]
        node_types = sorted({node["class_type"] for workflow in workflows for node in workflow.values()})
        # These nodes are injected only when an R2V shot contains video/audio refs,
        # so they are not present in the static workflow JSON.
        node_types = sorted(set(node_types) | {"LoadImage", "LoadVideo", "GetVideoComponents", "LoadAudio"})
        model_names: list[str] = []
        optional_model_names: list[str] = []
        for workflow in workflows:
            for node in workflow.values():
                inputs = node.get("inputs", {})
                for key in ("unet_name", "clip_name", "vae_name"):
                    value = inputs.get(key)
                    if isinstance(value, str):
                        model_names.append(value)
                # The acceleration LoRA is needed only by the optional Turbo
                # preview profile.  Native 20-step final rendering deliberately
                # bypasses it, so a missing LoRA must not make the server look
                # unusable for the quality path.
                lora_name = inputs.get("lora_name")
                if isinstance(lora_name, str):
                    optional_model_names.append(lora_name)
        return {
            "nodes": node_types,
            "models": sorted(set(model_names)),
            "optional_models": sorted(set(optional_model_names)),
        }


class ComfyUIClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.COMFYUI_BASE_URL).rstrip("/")
        self.client_id = uuid4().hex

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if settings.COMFYUI_API_TOKEN:
            headers["Authorization"] = f"Bearer {settings.COMFYUI_API_TOKEN}"
        return headers

    def _timeout(self, read: float | None = None) -> httpx.Timeout:
        base = settings.COMFYUI_REQUEST_TIMEOUT_SECONDS
        return httpx.Timeout(connect=min(base, 30.0), read=read or base, write=base, pool=base)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            response = await client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()

    async def preflight(self, requirements: dict[str, list[str]]) -> dict[str, Any]:
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            response = await client.get(f"{self.base_url}/object_info")
            response.raise_for_status()
            object_info = response.json()
        available_nodes = set(object_info)
        missing_nodes = [name for name in requirements.get("nodes", []) if name not in available_nodes]
        available_strings: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, str):
                available_strings.add(value)
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(object_info)
        required_models = requirements.get("models", [])
        optional_models = requirements.get("optional_models", [])
        missing_models = [name for name in required_models if name not in available_strings]
        missing_optional_models = [name for name in optional_models if name not in available_strings]
        return {
            "ok": not missing_nodes and not missing_models,
            "missing_nodes": missing_nodes,
            "missing_models": missing_models,
            "required_models": required_models,
            "missing_optional_models": missing_optional_models,
            "optional_models": optional_models,
        }

    async def upload_input(self, path: Path, remote_name: str | None = None) -> str:
        source = Path(path)
        if not source.exists():
            raise ComfyUIError(f"Input asset not found: {source}")
        requested = Path(remote_name or source.name)
        filename = requested.name
        subfolder = requested.parent.as_posix() if requested.parent.as_posix() != "." else ""
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(read=300.0), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            with source.open("rb") as handle:
                response = await client.post(
                    f"{self.base_url}/upload/image",
                    files={settings.COMFYUI_UPLOAD_FIELD: (filename, handle, mime_type)},
                    data={"type": "input", "subfolder": subfolder, "overwrite": "true"},
                )
            if response.is_error:
                raise ComfyUIError(f"ComfyUI upload failed ({response.status_code}): {response.text[:500]}")
            payload = response.json()
        name = payload.get("name") or filename
        subfolder = payload.get("subfolder") or ""
        return f"{subfolder}/{name}".lstrip("/")

    async def upload_assets(self, assets: list[Asset], namespace: str) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for asset in assets:
            asset_path = resolve_media_path(asset.path)
            suffix = asset_path.suffix
            remote_name = f"videoworkflow/{namespace}/{asset.id}{suffix}"
            uploaded[asset.id] = await self.upload_input(asset_path, remote_name)
        return uploaded

    async def submit(self, workflow: dict[str, Any]) -> str:
        # ComfyUI 0.32+ validates custom prompt ids as canonical UUID strings.
        # Keeping our own id makes queue recovery deterministic while remaining
        # compatible with older servers that simply echoed arbitrary strings.
        prompt_id = str(uuid4())
        payload = {"prompt": workflow, "client_id": self.client_id, "prompt_id": prompt_id}
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            response = await client.post(f"{self.base_url}/prompt", json=payload)
            if response.is_error:
                raise ComfyUIError(f"ComfyUI prompt rejected ({response.status_code}): {response.text[:1000]}")
            result = response.json()
        if result.get("error") or result.get("node_errors"):
            raise ComfyUIError(f"ComfyUI workflow validation failed: {json.dumps(result, ensure_ascii=False)[:2000]}")
        return str(result.get("prompt_id") or prompt_id)

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            data = response.json()
        entry = data.get(prompt_id) if isinstance(data, dict) else None
        return entry if isinstance(entry, dict) else None

    async def get_queue(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            response = await client.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()

    async def wait_for_result(
        self,
        prompt_id: str,
        on_progress: ProgressCallback | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        timeout = timeout_seconds or settings.COMFYUI_JOB_TIMEOUT_SECONDS
        last_message = "queued"
        while time.monotonic() - started < timeout:
            history = await self.get_history(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    messages = status.get("messages") or []
                    raise ComfyUIError(f"ComfyUI execution failed: {json.dumps(messages, ensure_ascii=False)[-2000:]}")
                outputs = history.get("outputs")
                if isinstance(outputs, dict) and outputs:
                    if on_progress:
                        await on_progress(1.0, "completed")
                    return history

            try:
                queue = await self.get_queue()
                running = queue.get("queue_running") or []
                pending = queue.get("queue_pending") or []
                if any(prompt_id in json.dumps(item) for item in running):
                    last_message = "running"
                else:
                    position = next(
                        (index for index, item in enumerate(pending, start=1) if prompt_id in json.dumps(item)),
                        None,
                    )
                    if position is not None:
                        last_message = f"queued:{position}"
                if on_progress:
                    elapsed_ratio = min(0.95, (time.monotonic() - started) / max(timeout, 1))
                    await on_progress(max(0.02, elapsed_ratio), last_message)
            except Exception:
                pass
            await asyncio.sleep(settings.COMFYUI_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"ComfyUI job timed out after {timeout}s: {prompt_id}")

    async def cancel(self, prompt_id: str) -> None:
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
        ) as client:
            await client.post(f"{self.base_url}/queue", json={"delete": [prompt_id]})
            queue = await self.get_queue()
            if any(prompt_id in json.dumps(item) for item in queue.get("queue_running") or []):
                await client.post(f"{self.base_url}/interrupt")

    @staticmethod
    def extract_output(history: dict[str, Any]) -> dict[str, str]:
        candidates: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("filename"), str):
                    candidates.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(history.get("outputs", {}))
        if not candidates:
            raise ComfyUIError("ComfyUI completed but returned no downloadable output")
        video = next((item for item in candidates if Path(item["filename"]).suffix.lower() in {".mp4", ".webm", ".mov"}), candidates[-1])
        return {
            "filename": video["filename"],
            "subfolder": video.get("subfolder", ""),
            "type": video.get("type", "output"),
        }

    async def download_output(self, output: dict[str, str], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        query = urlencode({"filename": output["filename"], "subfolder": output.get("subfolder", ""), "type": output.get("type", "output")})
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=self._timeout(read=600.0), verify=settings.COMFYUI_VERIFY_TLS,
            trust_env=settings.COMFYUI_TRUST_ENV,
            follow_redirects=True,
        ) as client:
            response = await client.get(f"{self.base_url}/view?{query}")
            response.raise_for_status()
            destination.write_bytes(response.content)
        return destination

    def websocket_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/ws?clientId={self.client_id}"
