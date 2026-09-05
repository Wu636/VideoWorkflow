from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from src.video_workflow.config import settings
from src.video_workflow.integrations.atlas_h3 import ATLAS_MODEL_ID as ATLAS_H3_MODEL_LABEL
from src.video_workflow.integrations.metaso_h3 import METASO_H3_MODEL_ID


@dataclass(frozen=True)
class RuntimeField:
    key: str
    group: Literal["llm", "image", "video", "comfyui", "audio"]
    label: str
    description: str = ""
    secret: bool = False
    options: tuple[str, ...] = ()


RUNTIME_FIELDS = (
    RuntimeField("LLM_PROVIDER", "llm", "分镜生成模型", "用于生成完整分镜表。", options=("deepseek", "glm", "openlux", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("BRIEF_ANALYSIS_PROVIDER", "llm", "剧本分析模型", "auto 表示沿用分镜模型，遇到参考图时优先使用同一多模态服务。", options=("auto", "deepseek", "glm", "openlux", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("SHOT_COUNT_PROVIDER", "llm", "镜头数建议模型", "auto 表示沿用剧本分析模型。", options=("auto", "deepseek", "glm", "openlux", "ark", "ark_doubao", "ark_deepseek")),
    RuntimeField("REFERENCE_ANALYSIS_PROVIDER", "llm", "参考图理解模型", "auto 优先沿用已选择的 OpenLux 多模态模型，再按方舟视觉 → GLM 选择。", options=("auto", "openlux", "glm", "ark", "ark_doubao")),
    RuntimeField("DEEPSEEK_API_KEY", "llm", "DeepSeek API Key", secret=True),
    RuntimeField("DEEPSEEK_BASE_URL", "llm", "DeepSeek Base URL"),
    RuntimeField("DEEPSEEK_MODEL", "llm", "DeepSeek 模型"),
    RuntimeField("GLM_API_KEY", "llm", "智谱 GLM API Key", secret=True),
    RuntimeField("GLM_MODEL", "llm", "GLM 视觉模型"),
    RuntimeField("OPENLUX_API_KEY", "llm", "OpenLux API Key", "在 OpenLux 控制台创建；用于 GPT、Claude、Gemini、DeepSeek 等模型。", secret=True),
    RuntimeField("OPENLUX_BASE_URL", "llm", "OpenLux Base URL", "官方 OpenAI 兼容地址，默认 https://api.openlux.ai/v1。"),
    RuntimeField("OPENLUX_MODEL", "llm", "OpenLux 文本/分镜模型", "可填 OpenLux 模型广场中的模型 ID，例如 claude-sonnet-5、claude-opus-5、gpt-5.6-sol、gpt-5.5。"),
    RuntimeField("OPENLUX_VISION_MODEL", "llm", "OpenLux 多模态模型", "用于参考图分析、人物形象补全和带图分镜；默认 gpt-5.6-sol。"),
    RuntimeField("OPENLUX_REQUEST_TIMEOUT_SECONDS", "llm", "OpenLux 单次请求超时（秒）", "旗舰模型可能需要数分钟；默认 900 秒。程序不会自动产生第二次付费请求。"),
    RuntimeField("OPENLUX_STREAM", "llm", "OpenLux 流式长连接", "建议开启。持续接收增量结果，降低代理或网关因长时间无数据而断开连接的概率。"),
    RuntimeField("OPENLUX_SAVE_RAW_RESPONSES", "llm", "保存 OpenLux 原始响应", "建议开启。模型一旦返回就先落盘，后续 JSON 校验失败仍可恢复已经付费的结果。"),
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
    RuntimeField(
        "SEEDANCE_DEFAULT_MODEL", "video", "Seedance 默认模型",
        "项目页仍可逐批切换 2.0 / Fast / Mini。",
        options=("doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128", "doubao-seedance-2-0-mini-260615"),
    ),
    RuntimeField("SEEDANCE_DEFAULT_RESOLUTION", "video", "Seedance 默认分辨率", options=("480p", "720p", "1080p", "4k")),
    RuntimeField(
        "SEEDANCE_PUBLIC_ASSET_BASE_URL", "video", "Seedance 素材公网地址",
        "本地分镜图/角色图通过此地址生成带时效签名的下载链接；留空时本地素材会在提交前被拦截。"
        "隧道域名过期后可运行 scripts/restart-seedance-tunnel.sh 一键刷新并回写。",
    ),
    RuntimeField("SEEDANCE_POLL_INTERVAL_SECONDS", "video", "Seedance 轮询间隔（秒）"),
    RuntimeField("SEEDANCE_JOB_TIMEOUT_SECONDS", "video", "Seedance 单任务超时（秒）"),
    RuntimeField("ARK_VIDEO_MODEL", "video", "旧版方舟视频模型"),
    RuntimeField("GRSAI_VIDEO_MODEL", "video", "GRSAI 视频模型"),
    RuntimeField("COMFYUI_BASE_URL", "comfyui", "ComfyUI 地址", "MiniMax H3 服务器的外网或内网地址。"),
    RuntimeField(
        "H3_PROVIDER", "comfyui", "H3 生成通道",
        "comfyui_h3 走自部署 ComfyUI；atlas_h3 与 metaso_h3 分别走两条线上 MiniMax H3 API。",
        options=("comfyui_h3", "metaso_h3", "atlas_h3"),
    ),
    RuntimeField("ATLASCLOUD_API_KEY", "comfyui", "Atlas Cloud API Key", "https://console.atlascloud.ai 获取；仅 atlas_h3 通道需要。", secret=True),
    RuntimeField("ATLASCLOUD_BASE_URL", "comfyui", "Atlas Cloud Base URL"),
    RuntimeField(
        "H3_ATLAS_RESOLUTION", "comfyui", "Atlas H3 分辨率",
        "Atlas 通道提交时使用的分辨率，手动选择，不随项目尺寸自动映射。",
        options=("768P", "1080P"),
    ),
    RuntimeField(
        "H3_ATLAS_RATIO", "comfyui", "Atlas H3 画面比例",
        "adaptive 表示让模型自行决定；成片会缩放回项目分辨率。",
        options=("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
    ),
    RuntimeField("ATLASCLOUD_JOB_TIMEOUT_SECONDS", "comfyui", "Atlas 单任务超时（秒）"),
    RuntimeField("METASO_H3_API_KEY", "comfyui", "MetaSo H3 API Key", "MetaSo 控制台 H3 API 密钥；仅 metaso_h3 通道使用。", secret=True),
    RuntimeField("METASO_H3_BASE_URL", "comfyui", "MetaSo H3 Base URL", "默认 https://metaso.cn/api/minimax。"),
    RuntimeField("METASO_H3_RESOLUTION", "comfyui", "MetaSo H3 分辨率", "768P 是默认基础生成档；需要更高细节时可选择 2K。", options=("768P", "2K")),
    RuntimeField(
        "METASO_H3_RATIO", "comfyui", "MetaSo H3 画面比例",
        "adaptive 表示由模型结合参考素材决定；成片会缩放回项目分辨率。",
        options=("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
    ),
    RuntimeField("METASO_H3_CONTEXT_IR_ENABLED", "comfyui", "MetaSo Context IR", "默认关闭；开启后网关会按次收取额外的素材理解费用。"),
    RuntimeField("METASO_H3_POLL_INTERVAL_SECONDS", "comfyui", "MetaSo 轮询间隔（秒）"),
    RuntimeField("METASO_H3_JOB_TIMEOUT_SECONDS", "comfyui", "MetaSo 单任务超时（秒）"),
    RuntimeField("COMFYUI_API_TOKEN", "comfyui", "ComfyUI API Token", secret=True),
    RuntimeField("COMFYUI_REQUEST_TIMEOUT_SECONDS", "comfyui", "请求超时（秒）"),
    RuntimeField("COMFYUI_POLL_INTERVAL_SECONDS", "comfyui", "轮询间隔（秒）"),
    RuntimeField("COMFYUI_JOB_TIMEOUT_SECONDS", "comfyui", "单任务超时（秒）"),
    RuntimeField("COMFYUI_VERIFY_TLS", "comfyui", "验证 TLS 证书"),
    RuntimeField("COMFYUI_TRUST_ENV", "comfyui", "继承系统代理"),
    RuntimeField("COMFYUI_HOURLY_RATE", "comfyui", "GPU 每小时价格（元）"),
    RuntimeField(
        "H3_MODEL_PROFILE", "comfyui", "默认 H3 diffusion 权重",
        "pruned_int8 是当前 21GB 基线；full_int8 保留完整模型能力；BF16 需要 CPU offload 或更大显存。",
        options=("pruned_int8", "pruned_fp8", "full_int8", "pruned_bf16", "full_bf16"),
    ),
    RuntimeField(
        "H3_TEXT_ENCODER_PROFILE", "comfyui", "默认 H3 文本编码器",
        "NVFP4 最省显存；INT8/BF16 可能改善复杂指令和多人关系理解，但显存/内存开销更高。",
        options=("nvfp4", "int8", "bf16"),
    ),
    RuntimeField("H3_AUTO_SEGMENT_COMPLEX_SHOTS", "comfyui", "复杂长镜头自动续帧拆段", "开启后，超过稳定时长或包含多动作的镜头会拆成短段连续生成，再自动拼回一条视频。"),
    RuntimeField("H3_MAX_SEGMENT_SECONDS", "comfyui", "单段高稳定时长（秒）", "建议 6–8 秒；越短越稳，但会增加生成次数。"),
    RuntimeField("H3_AUDIO_MODE", "audio", "成片音频策略", "native 默认保留 H3 原声；需要独立配音版时再主动选择 clean_tts；mute 输出静音版。", options=("native", "clean_tts", "mute")),
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
        if provider == "openlux":
            model = settings.OPENLUX_VISION_MODEL if vision else settings.OPENLUX_MODEL
            return "OpenLux", model, bool(settings.OPENLUX_API_KEY)
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
            {"value": "openlux", "label": "OpenLux（GPT / Claude / Gemini）"},
            {"value": "ark", "label": "火山方舟（默认端点）"},
            {"value": "ark_doubao", "label": "火山方舟（豆包/通用端点）"},
            {"value": "ark_deepseek", "label": "火山方舟（DeepSeek 端点）"},
        ]
        vision_options = [
            {"value": "auto", "label": "自动选择"},
            {"value": "openlux", "label": "OpenLux 多模态"},
            {"value": "glm", "label": "智谱 GLM"},
            {"value": "ark", "label": "火山方舟视觉（默认端点）"},
            {"value": "ark_doubao", "label": "火山方舟视觉"},
        ]

        brief_selected = settings.BRIEF_ANALYSIS_PROVIDER
        brief_effective = self._resolve_llm(brief_selected, settings.LLM_PROVIDER)
        brief_label, brief_model, brief_ready = self._llm_info(brief_effective)
        vision_selected = settings.REFERENCE_ANALYSIS_PROVIDER
        if vision_selected == "auto":
            if brief_effective == "openlux" and settings.OPENLUX_API_KEY:
                vision_effective = "openlux"
            elif settings.ARK_API_KEY:
                vision_effective = "ark_doubao"
            elif settings.GLM_API_KEY:
                vision_effective = "glm"
            elif settings.OPENLUX_API_KEY:
                vision_effective = "openlux"
            else:
                vision_effective = brief_effective
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

        h3_selected = settings.H3_PROVIDER if settings.H3_PROVIDER in {"comfyui_h3", "metaso_h3", "atlas_h3"} else "comfyui_h3"
        if h3_selected == "metaso_h3":
            h3_label = "MetaSo MiniMax H3 API"
            h3_model = f"{METASO_H3_MODEL_ID} · {settings.METASO_H3_RESOLUTION} · {settings.METASO_H3_RATIO}"
            h3_ready = bool(settings.METASO_H3_API_KEY)
            h3_priority = ["MetaSo 线上 API（选中时）", f"Context IR：{'开启' if settings.METASO_H3_CONTEXT_IR_ENABLED else '关闭'}"]
        elif h3_selected == "atlas_h3":
            h3_label = "Atlas Cloud MiniMax H3 API"
            h3_model = f"{ATLAS_H3_MODEL_LABEL} · {settings.H3_ATLAS_RESOLUTION} · {settings.H3_ATLAS_RATIO}"
            h3_ready = bool(settings.ATLASCLOUD_API_KEY)
            h3_priority = ["Atlas Cloud 线上 API（选中时）"]
        else:
            h3_label = "MiniMax H3 / ComfyUI"
            h3_model = f"{settings.H3_MODEL_PROFILE} + {settings.H3_TEXT_ENCODER_PROFILE}"
            h3_ready = bool(settings.COMFYUI_BASE_URL)
            h3_priority = [
                "自部署 ComfyUI（选中时）",
                f"Diffusion：{settings.H3_MODEL_PROFILE}",
                f"文本编码器：{settings.H3_TEXT_ENCODER_PROFILE}",
            ]

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
                "priority": ["OpenLux（分镜模型已选且配置时）", "火山方舟视觉（备用）", "智谱 GLM（备用）", "剧本模型文本描述（兜底）"],
                "description": "读取角色参考图外貌、服饰与固定视觉特征。",
            },
            {
                "id": "keyframe", "label": "分镜首帧生成", "setting_key": "IMAGE_PROVIDER",
                "selected": image_provider, "options": [{"value": "ark", "label": "火山方舟图像"}, {"value": "grsai", "label": "GRSAI"}],
                "effective_label": image_label, "model": image_model, "configured": image_ready,
                "priority": [image_label], "description": "为每个镜头生成 I2V 首帧或构图参考图。",
            },
            {
                "id": "seedance_video", "label": "Seedance 付费分镜视频", "setting_key": "SEEDANCE_DEFAULT_MODEL",
                "selected": settings.SEEDANCE_DEFAULT_MODEL,
                "options": [
                    {"value": "doubao-seedance-2-0-260128", "label": "Seedance 2.0（质量优先）"},
                    {"value": "doubao-seedance-2-0-fast-260128", "label": "Seedance 2.0 Fast（速度/成本均衡）"},
                    {"value": "doubao-seedance-2-0-mini-260615", "label": "Seedance 2.0 Mini（成本优先）"},
                ],
                "effective_label": "火山方舟 Seedance 2.0 系列", "model": settings.SEEDANCE_DEFAULT_MODEL,
                "configured": bool(settings.ARK_API_KEY),
                "priority": [settings.SEEDANCE_DEFAULT_MODEL, f"默认 {settings.SEEDANCE_DEFAULT_RESOLUTION}", "项目页可逐批覆盖"],
                "description": "走官方异步视频 API；按项目所选镜头生成，使用独立 Seedance 全模态 Prompt。",
            },
            {
                "id": "h3_video", "label": "项目视频片段生成", "setting_key": "H3_PROVIDER",
                "selected": h3_selected,
                "options": [
                    {"value": "comfyui_h3", "label": "自部署 ComfyUI（本地 GPU）"},
                    {"value": "metaso_h3", "label": "MetaSo MiniMax H3 API（按量付费）"},
                    {"value": "atlas_h3", "label": "Atlas Cloud 线上 API（按量付费）"},
                ],
                "effective_label": h3_label,
                "model": h3_model,
                "configured": h3_ready,
                "priority": h3_priority,
                "description": "三条通道共用分镜、对白、原声保留与断点续轮询管线；MetaSo/Atlas 在 15 秒内整段生成，超过才续帧分段，成片自动缩放回项目分辨率。",
            },
            {
                "id": "dialogue_audio", "label": "H3 原声与可选配音替换", "setting_key": "H3_AUDIO_MODE",
                "selected": settings.H3_AUDIO_MODE,
                "options": [
                    {"value": "native", "label": "保留 H3 原声（默认）"},
                    {"value": "clean_tts", "label": "主动替换为独立 TTS"},
                    {"value": "mute", "label": "移除原声并静音"},
                ],
                "effective_label": "Edge TTS + FFmpeg" if settings.H3_AUDIO_MODE == "clean_tts" else "FFmpeg 静音轨" if settings.H3_AUDIO_MODE == "mute" else "MiniMax H3 原声",
                "model": settings.TTS_PROVIDER if settings.H3_AUDIO_MODE == "clean_tts" else settings.H3_AUDIO_MODE,
                "configured": settings.H3_AUDIO_MODE != "clean_tts" or settings.TTS_PROVIDER in {"edge", "disabled"},
                "priority": (
                    ["保留 H3 自带人声、环境音与动作音效", "需要配音版时再手动选择 clean_tts"]
                    if settings.H3_AUDIO_MODE == "native"
                    else ["保留一份 H3 原声备份", "按明确发言者生成独立普通话音轨", "画面流直接复制"]
                    if settings.H3_AUDIO_MODE == "clean_tts"
                    else ["保留一份 H3 原声备份", "输出静音交付版"]
                ),
                "description": (
                    "默认直接交付 MiniMax H3 原声音轨，不执行 TTS 覆盖。"
                    if settings.H3_AUDIO_MODE == "native"
                    else "仅在主动选择后生成独立 TTS 替换版；原始 H3 音轨会保存在同目录备份中。"
                    if settings.H3_AUDIO_MODE == "clean_tts"
                    else "主动选择后输出静音版；原始 H3 音轨会保存在同目录备份中。"
                ),
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
