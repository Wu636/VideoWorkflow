from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from src.video_workflow.config import settings


@dataclass(frozen=True)
class RuntimeField:
    key: str
    group: Literal["llm", "image", "video", "comfyui", "audio"]
    label: str
    description: str = ""
    secret: bool = False
    options: tuple[str, ...] = ()


RUNTIME_FIELDS = (
    RuntimeField("LLM_PROVIDER", "llm", "分镜生成模型", "用于生成完整分镜表。", options=("deepseek", "glm", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("BRIEF_ANALYSIS_PROVIDER", "llm", "剧本分析模型", "auto 表示沿用分镜模型，遇到参考图时切到视觉模型。", options=("auto", "deepseek", "glm", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("SHOT_COUNT_PROVIDER", "llm", "镜头数建议模型", "auto 表示沿用剧本分析模型。", options=("auto", "deepseek", "glm", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("REFERENCE_ANALYSIS_PROVIDER", "llm", "参考图理解模型", "auto 按方舟视觉 → GLM → 剧本模型选择已配置服务。", options=("auto", "glm", "ark", "ark_doubao")),
    RuntimeField("DEEPSEEK_API_KEY", "llm", "DeepSeek API Key", secret=True),
    RuntimeField("DEEPSEEK_BASE_URL", "llm", "DeepSeek Base URL"),
    RuntimeField("DEEPSEEK_MODEL", "llm", "DeepSeek 模型"),
    RuntimeField("GLM_API_KEY", "llm", "智谱 GLM API Key", secret=True),
    RuntimeField("GLM_MODEL", "llm", "GLM 视觉模型"),
    RuntimeField("ARK_API_KEY", "llm", "火山方舟 API Key", secret=True),
    RuntimeField("ARK_BASE_URL", "llm", "火山方舟 Base URL"),
    RuntimeField("ARK_LLM_MODEL", "llm", "方舟文本模型"),
    RuntimeField("ARK_VISION_MODEL", "llm", "方舟视觉模型"),
    RuntimeField("IMAGE_PROVIDER", "image", "默认分镜图服务", options=("grsai", "ark")),
    RuntimeField("ARK_IMAGE_MODEL", "image", "方舟图像模型"),
    RuntimeField("GRSAI_API_KEY", "image", "GRSAI API Key", secret=True),
    RuntimeField("GRSAI_BASE_URL", "image", "GRSAI Base URL"),
    RuntimeField("GRSAI_IMAGE_MODEL", "image", "GRSAI 图像模型"),
    RuntimeField("GRSAI_IMAGE_SIZE", "image", "GRSAI 图像尺寸", options=("1K", "2K", "4K")),
    RuntimeField("VIDEO_PROVIDER", "video", "默认外部视频服务", options=("grsai", "ark")),
    RuntimeField("ARK_VIDEO_MODEL", "video", "方舟视频模型"),
    RuntimeField("GRSAI_VIDEO_MODEL", "video", "GRSAI 视频模型"),
    RuntimeField("COMFYUI_BASE_URL", "comfyui", "ComfyUI 地址", "MiniMax H3 服务器的外网或内网地址。"),
    RuntimeField("COMFYUI_API_TOKEN", "comfyui", "ComfyUI API Token", secret=True),
    RuntimeField("COMFYUI_REQUEST_TIMEOUT_SECONDS", "comfyui", "请求超时（秒）"),
    RuntimeField("COMFYUI_POLL_INTERVAL_SECONDS", "comfyui", "轮询间隔（秒）"),
    RuntimeField("COMFYUI_JOB_TIMEOUT_SECONDS", "comfyui", "单任务超时（秒）"),
    RuntimeField("COMFYUI_VERIFY_TLS", "comfyui", "验证 TLS 证书"),
    RuntimeField("COMFYUI_TRUST_ENV", "comfyui", "继承系统代理"),
    RuntimeField("COMFYUI_HOURLY_RATE", "comfyui", "GPU 每小时价格（元）"),
    RuntimeField("H3_AUTO_SEGMENT_COMPLEX_SHOTS", "comfyui", "复杂长镜头自动续帧拆段", "开启后，超过稳定时长或包含多动作的镜头会拆成短段连续生成，再自动拼回一条视频。"),
    RuntimeField("H3_MAX_SEGMENT_SECONDS", "comfyui", "单段高稳定时长（秒）", "建议 6–8 秒；越短越稳，但会增加生成次数。"),
    RuntimeField("H3_AUDIO_MODE", "audio", "成片音频策略", "clean_tts 会丢弃 H3 波浪噪声音轨并覆盖独立对白；mute 输出干净静音；native 保留 H3 原声。", options=("clean_tts", "mute", "native")),
    RuntimeField("TTS_PROVIDER", "audio", "对白语音服务", "edge 为免密钥的在线普通话语音；disabled 仅输出静音。", options=("edge", "disabled")),
    RuntimeField("TTS_DEFAULT_FEMALE_VOICE", "audio", "默认女声 Voice ID"),
    RuntimeField("TTS_DEFAULT_MALE_VOICE", "audio", "默认男声 Voice ID"),
    RuntimeField("TTS_DEFAULT_MATURE_FEMALE_VOICE", "audio", "默认中年女声 Voice ID"),
    RuntimeField("TTS_DEFAULT_MATURE_MALE_VOICE", "audio", "默认中年男声 Voice ID"),
    RuntimeField("H3_POSTPROCESS_AUDIO_BITRATE", "audio", "成片对白码率"),
)

FIELD_MAP = {field.key: field for field in RUNTIME_FIELDS}
GROUP_LABELS = {"llm": "剧本与分镜大模型", "image": "分镜图生成", "video": "外部视频模型", "comfyui": "MiniMax H3 / ComfyUI", "audio": "对白与音频净化"}


class RuntimeSettingsManager:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return Path(settings.OUTPUT_DIR) / "runtime_settings.json"

    @staticmethod
    def _coerce(key: str, value: Any) -> Any:
        annotation = type(settings).model_fields[key].annotation
        return TypeAdapter(annotation).validate_python(value)

    def load(self) -> None:
        path = self.path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        valid: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in FIELD_MAP:
                continue
            try:
                valid[key] = self._coerce(key, value)
            except (TypeError, ValueError):
                continue
        self._values = valid
        self._apply(valid)

    @staticmethod
    def _apply(values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(settings, key, value)

    def update(self, values: dict[str, Any], clear_keys: list[str] | None = None) -> None:
        merged = dict(self._values)
        for key in clear_keys or []:
            if key in FIELD_MAP and FIELD_MAP[key].secret:
                merged[key] = None
        for key, value in values.items():
            field = FIELD_MAP.get(key)
            if field is None:
                raise ValueError(f"不支持的配置项: {key}")
            if field.secret and (value is None or value == ""):
                continue
            merged[key] = self._coerce(key, value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)
        self._values = merged
        self._apply(merged)

    def public_payload(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        for group in ("llm", "image", "video", "comfyui", "audio"):
            fields = []
            for definition in (item for item in RUNTIME_FIELDS if item.group == group):
                value = getattr(settings, definition.key)
                item: dict[str, Any] = {
                    "key": definition.key,
                    "label": definition.label,
                    "description": definition.description,
                    "secret": definition.secret,
                    "options": list(definition.options),
                    "value": "" if definition.secret else value,
                }
                if definition.secret:
                    item["configured"] = bool(value)
                    item["masked"] = f"••••{str(value)[-4:]}" if value else "未配置"
                fields.append(item)
            groups.append({"id": group, "label": GROUP_LABELS[group], "fields": fields})
        return {
            "routes": self._routing_payload(),
            "groups": groups,
            "path": str(self.path),
            "restart_required": False,
        }

    @staticmethod
    def _llm_info(provider: str, *, vision: bool = False) -> tuple[str, str, bool]:
        if provider == "glm":
            return "智谱 GLM", settings.GLM_MODEL, bool(settings.GLM_API_KEY)
        if provider in {"ark", "ark_doubao", "ark_deepseek"}:
            return "火山方舟", settings.ARK_VISION_MODEL if vision else settings.ARK_LLM_MODEL, bool(settings.ARK_API_KEY)
        return "DeepSeek", settings.DEEPSEEK_MODEL, bool(settings.DEEPSEEK_API_KEY)

    @staticmethod
    def _resolve_llm(selected: str, fallback: str | None = None) -> str:
        if selected == "auto":
            selected = fallback or settings.LLM_PROVIDER
        return selected

    def _routing_payload(self) -> list[dict[str, Any]]:
        provider_options = [
            {"value": "auto", "label": "自动选择"},
            {"value": "deepseek", "label": "DeepSeek"},
            {"value": "glm", "label": "智谱 GLM"},
            {"value": "ark", "label": "火山方舟（默认端点）"},
            {"value": "ark_doubao", "label": "火山方舟（豆包/通用端点）"},
            {"value": "ark_deepseek", "label": "火山方舟（DeepSeek 端点）"},
        ]
        vision_options = [
            {"value": "auto", "label": "自动选择"},
            {"value": "glm", "label": "智谱 GLM"},
            {"value": "ark", "label": "火山方舟视觉（默认端点）"},
            {"value": "ark_doubao", "label": "火山方舟视觉"},
        ]

        brief_selected = settings.BRIEF_ANALYSIS_PROVIDER
        brief_effective = self._resolve_llm(brief_selected, settings.LLM_PROVIDER)
        brief_label, brief_model, brief_ready = self._llm_info(brief_effective)
        vision_selected = settings.REFERENCE_ANALYSIS_PROVIDER
        if vision_selected == "auto":
            vision_effective = "ark_doubao" if settings.ARK_API_KEY else "glm" if settings.GLM_API_KEY else brief_effective
        else:
            vision_effective = vision_selected
        vision_label, vision_model, vision_ready = self._llm_info(vision_effective, vision=True)
        count_selected = settings.SHOT_COUNT_PROVIDER
        count_effective = self._resolve_llm(count_selected, brief_effective)
        count_label, count_model, count_ready = self._llm_info(count_effective)
        storyboard_label, storyboard_model, storyboard_ready = self._llm_info(settings.LLM_PROVIDER)

        image_provider = settings.IMAGE_PROVIDER
        if image_provider == "ark":
            image_label, image_model, image_ready = "火山方舟图像", settings.ARK_IMAGE_MODEL, bool(settings.ARK_API_KEY)
        else:
            image_label, image_model, image_ready = "GRSAI", settings.GRSAI_IMAGE_MODEL, bool(settings.GRSAI_API_KEY)
        video_provider = settings.VIDEO_PROVIDER
        if video_provider == "ark":
            video_label, video_model, video_ready = "火山方舟视频", settings.ARK_VIDEO_MODEL, bool(settings.ARK_API_KEY)
        else:
            video_label, video_model, video_ready = "GRSAI", settings.GRSAI_VIDEO_MODEL, bool(settings.GRSAI_API_KEY)

        return [
            {
                "id": "brief_analysis", "label": "剧本分析、扩写与需求回填", "setting_key": "BRIEF_ANALYSIS_PROVIDER",
                "selected": brief_selected, "options": provider_options, "effective_label": brief_label,
                "model": brief_model, "configured": brief_ready,
                "priority": [f"普通剧本：{brief_label}", f"绑定参考图：{vision_label}"],
                "description": "按目标时长扩写/缩写剧本，并填写视觉风格、节奏、受众、风格圣经与角色草稿。",
            },
            {
                "id": "shot_count", "label": "AI 镜头数建议", "setting_key": "SHOT_COUNT_PROVIDER",
                "selected": count_selected, "options": provider_options, "effective_label": count_label,
                "model": count_model, "configured": count_ready, "priority": [count_label],
                "description": "根据时长、剧情转折、场景和对白节奏决定镜头数量。",
            },
            {
                "id": "storyboard", "label": "详细分镜生成", "setting_key": "LLM_PROVIDER",
                "selected": settings.LLM_PROVIDER, "options": provider_options[1:], "effective_label": storyboard_label,
                "model": storyboard_model, "configured": storyboard_ready, "priority": [storyboard_label],
                "description": "生成逐镜剧情、画面、动作、镜头、声音和 H3 Prompt。",
            },
            {
                "id": "reference_vision", "label": "角色参考图理解", "setting_key": "REFERENCE_ANALYSIS_PROVIDER",
                "selected": vision_selected, "options": vision_options, "effective_label": vision_label,
                "model": vision_model, "configured": vision_ready,
                "priority": ["火山方舟视觉（已配置时）", "智谱 GLM（备用）", "剧本模型文本描述（兜底）"],
                "description": "读取角色参考图外貌、服饰与固定视觉特征。",
            },
            {
                "id": "keyframe", "label": "分镜首帧生成", "setting_key": "IMAGE_PROVIDER",
                "selected": image_provider, "options": [{"value": "ark", "label": "火山方舟图像"}, {"value": "grsai", "label": "GRSAI"}],
                "effective_label": image_label, "model": image_model, "configured": image_ready,
                "priority": [image_label], "description": "为每个镜头生成 I2V 首帧或构图参考图。",
            },
            {
                "id": "h3_video", "label": "项目视频片段生成", "setting_key": None,
                "selected": "comfyui_h3", "options": [], "effective_label": "MiniMax H3 / ComfyUI",
                "model": settings.COMFYUI_BASE_URL, "configured": bool(settings.COMFYUI_BASE_URL),
                "priority": ["MiniMax H3 I2V/R2V", "本地持久队列"],
                "description": "当前项目工作台固定走自部署 H3；逐镜参数在 H3 生成页控制。",
            },
            {
                "id": "dialogue_audio", "label": "对白生成与 H3 杂音替换", "setting_key": "H3_AUDIO_MODE",
                "selected": settings.H3_AUDIO_MODE,
                "options": [
                    {"value": "clean_tts", "label": "独立 TTS 干净对白（推荐）"},
                    {"value": "mute", "label": "移除原声并静音"},
                    {"value": "native", "label": "保留 H3 原声"},
                ],
                "effective_label": "Edge TTS + FFmpeg" if settings.H3_AUDIO_MODE == "clean_tts" else "FFmpeg 静音轨" if settings.H3_AUDIO_MODE == "mute" else "MiniMax H3 原声",
                "model": settings.TTS_PROVIDER if settings.H3_AUDIO_MODE == "clean_tts" else settings.H3_AUDIO_MODE,
                "configured": settings.H3_AUDIO_MODE != "clean_tts" or settings.TTS_PROVIDER in {"edge", "disabled"},
                "priority": ["角色专属 voice ID", "按声音描述选择男/女默认音色", "任一对白失败即阻止交付并换 Seed 重试"],
                "description": "默认丢弃 H3 宽带波浪噪声音轨，用明确发言者的独立普通话对白覆盖；画面流直接复制，不二次压画质。",
            },
            {
                "id": "legacy_video", "label": "旧版工作台视频生成", "setting_key": "VIDEO_PROVIDER",
                "selected": video_provider, "options": [{"value": "ark", "label": "火山方舟视频"}, {"value": "grsai", "label": "GRSAI"}],
                "effective_label": video_label, "model": video_model, "configured": video_ready,
                "priority": [video_label], "description": "仅影响保留的旧版工作台，项目生产仍优先使用 H3。",
            },
            {
                "id": "finalize", "label": "剪辑、字幕与最终成片", "setting_key": None,
                "selected": "ffmpeg", "options": [], "effective_label": "本机 FFmpeg",
                "model": settings.FFMPEG_BIN, "configured": True, "priority": ["本机 FFmpeg"],
                "description": "拼接镜头、转场、混音、字幕、预览版和 QC，不调用大模型。",
            },
        ]


runtime_settings = RuntimeSettingsManager()
