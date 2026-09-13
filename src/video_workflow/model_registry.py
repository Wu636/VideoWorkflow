"""Provider connections, model discovery and function-level model routing.

The registry is deliberately small and file-backed for this single-user studio.
Secrets never leave this module in public payloads; custom connection data is
stored in a 0600 JSON file and can later be moved to the OS keychain without
changing the API shape.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from src.video_workflow.config import settings
from src.video_workflow.integrations.atlas_h3 import ATLAS_MODEL_ID as ATLAS_H3_MODEL_ID
from src.video_workflow.integrations.metaso_h3 import METASO_H3_MODEL_ID


Protocol = Literal["openai_chat", "ark", "anthropic_messages", "gemini", "local"]


class ProviderConnection(BaseModel):
    id: str
    name: str
    protocol: Protocol = "openai_chat"
    base_url: str
    api_key: str | None = None
    models_url: str | None = None
    api_key_header: str = "Authorization"
    enabled: bool = True
    builtin: bool = False
    hidden: bool = False
    last_checked_at: str | None = None
    last_status: str = "unknown"
    last_error: str = ""
    # Discovery and connectivity are separate operations.  Keep their status
    # independently so a 404 from an optional /models endpoint does not look
    # like a failed chat request in the UI.
    last_discovery_at: str | None = None
    last_discovery_status: str = "unknown"
    last_discovery_error: str = ""
    last_discovery_count: int | None = None
    last_test_at: str | None = None
    last_test_status: str = "unknown"
    last_test_error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProviderModel(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    connection_id: str
    model_id: str
    label: str = ""
    capabilities: list[str] = Field(default_factory=lambda: ["text", "json"])
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_price_per_million: float | None = None
    currency: str = "CNY"
    enabled: bool = True
    source: Literal["builtin", "discovered", "manual"] = "manual"
    last_seen_at: str | None = None


ROUTE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {"id": "brief_analysis", "label": "剧本、人物与场景分析", "capability": "text", "setting_key": "BRIEF_ANALYSIS_PROVIDER", "description": "分析剧本结构、人物关系、场景设定和制作需求。"},
    {"id": "shot_count", "label": "镜头数量建议", "capability": "text", "setting_key": "SHOT_COUNT_PROVIDER", "description": "根据时长、节奏、动作和对白密度建议镜头数。"},
    {"id": "storyboard", "label": "详细分镜与 Prompt", "capability": "text", "setting_key": "LLM_PROVIDER", "description": "生成逐镜剧情、镜头语言、动作弧线和 Seedance/H3 Prompt。"},
    {"id": "reference_vision", "label": "角色、场景与风格图理解", "capability": "vision", "setting_key": "REFERENCE_ANALYSIS_PROVIDER", "description": "读取人物、场景和用户上传风格图，提取可复用的视觉特征。"},
    {"id": "image_generation", "label": "生图：角色、场景、分镜与封面", "capability": "image", "setting_key": "IMAGE_PROVIDER", "description": "统一控制角色参考图、场景母版、分镜首帧和项目封面的图像服务。"},
    {"id": "seedance_video", "label": "Seedance 视频生成", "capability": "video", "setting_key": "SEEDANCE_DEFAULT_MODEL", "description": "提交并轮询 Seedance 异步视频任务；实际并发受账号配额限制。"},
    {"id": "h3_video", "label": "MiniMax H3 视频生成", "capability": "video", "setting_key": "H3_PROVIDER", "description": "选择 MetaSo、Atlas Cloud 或本地 ComfyUI H3 通道。"},
    {"id": "dialogue_audio", "label": "对白语音生成", "capability": "audio", "setting_key": "TTS_PROVIDER", "description": "为对白事件生成独立音轨；也可以选择静音以保留 H3 原声。"},
)

ROUTE_ALLOWED_CONNECTIONS: dict[str, set[str]] = {
    # These generators have provider-specific request formats.  Keeping the
    # picker scoped here prevents a manually-added model from being selected
    # for a route whose runtime adapter cannot execute it yet.
    "image_generation": {"grsai", "ark"},
    "seedance_video": {"ark"},
    "h3_video": {"metaso_h3", "atlas_h3", "comfyui_h3"},
    "dialogue_audio": {"edge_tts", "disabled_audio"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _grsai_base_url() -> str:
    base = settings.GRSAI_BASE_URL.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _builtin_connections() -> dict[str, ProviderConnection]:
    return {
        "grsai": ProviderConnection(
            id="grsai", name="GRSAI", protocol="openai_chat", base_url=_grsai_base_url(),
            api_key=settings.GRSAI_API_KEY, builtin=True,
        ),
        "openlux": ProviderConnection(
            id="openlux", name="OpenLux", protocol="openai_chat", base_url=settings.OPENLUX_BASE_URL,
            api_key=settings.OPENLUX_API_KEY, builtin=True,
        ),
        "claude": ProviderConnection(
            id="claude", name="Claude 中转", protocol="openai_chat", base_url=settings.CLAUDE_BASE_URL,
            api_key=settings.CLAUDE_API_KEY, builtin=True,
        ),
        "ark": ProviderConnection(
            id="ark", name="火山方舟", protocol="ark", base_url=settings.ARK_BASE_URL,
            api_key=settings.ARK_API_KEY, builtin=True,
        ),
        "deepseek": ProviderConnection(
            id="deepseek", name="DeepSeek 官方", protocol="openai_chat", base_url=settings.DEEPSEEK_BASE_URL,
            api_key=settings.DEEPSEEK_API_KEY, builtin=True,
        ),
        "glm": ProviderConnection(
            id="glm", name="智谱 GLM", protocol="openai_chat", base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=settings.GLM_API_KEY, builtin=True,
        ),
        "metaso_h3": ProviderConnection(
            id="metaso_h3", name="MetaSo MiniMax H3", protocol="local", base_url=settings.METASO_H3_BASE_URL,
            api_key=settings.METASO_H3_API_KEY, builtin=True, hidden=True,
        ),
        "atlas_h3": ProviderConnection(
            id="atlas_h3", name="Atlas Cloud MiniMax H3", protocol="local", base_url=settings.ATLASCLOUD_BASE_URL,
            api_key=settings.ATLASCLOUD_API_KEY, builtin=True, hidden=True,
        ),
        "comfyui_h3": ProviderConnection(
            id="comfyui_h3", name="ComfyUI H3", protocol="local", base_url=settings.COMFYUI_BASE_URL,
            api_key=settings.COMFYUI_API_TOKEN, builtin=True, hidden=True,
        ),
        "edge_tts": ProviderConnection(
            id="edge_tts", name="Edge TTS", protocol="local", base_url="local://edge-tts", builtin=True, hidden=True,
        ),
        "disabled_audio": ProviderConnection(
            id="disabled_audio", name="静音音轨", protocol="local", base_url="local://mute", builtin=True, hidden=True,
        ),
    }


def _builtin_models() -> list[ProviderModel]:
    models: list[ProviderModel] = []
    grsai_prices = {
        "gpt-6-astra": (4.0, 20.0),
        "gpt-5.6-terra": (0.9, 5.2),
        "gpt-5.6-sol": (2.2, 13.0),
        "gpt-5.5": (2.2, 13.5),
    }
    for model_id, (input_price, output_price) in grsai_prices.items():
        models.append(ProviderModel(
            connection_id="grsai", model_id=model_id, label=model_id,
            capabilities=["text", "json", "stream", "vision"],
            input_price_per_million=input_price, output_price_per_million=output_price,
            source="builtin",
        ))
    for model_id in ("claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-5"):
        for connection_id in ("openlux", "claude"):
            models.append(ProviderModel(
                connection_id=connection_id, model_id=model_id, label=model_id,
                capabilities=["text", "json", "stream", "vision"], source="builtin",
            ))
    models.extend([
        ProviderModel(connection_id="ark", model_id="deepseek-v4-pro-ga-260813", label="DeepSeek V4 Pro（火山）", capabilities=["text", "json"], source="builtin"),
        ProviderModel(connection_id="ark", model_id="deepseek-v4-flash-ga-260731", label="DeepSeek V4 Flash（火山）", capabilities=["text", "json"], source="builtin"),
        ProviderModel(connection_id="ark", model_id=settings.ARK_LLM_MODEL, label=settings.ARK_LLM_MODEL, capabilities=["text", "json"], source="builtin"),
        ProviderModel(connection_id="ark", model_id=settings.ARK_VISION_MODEL, label=settings.ARK_VISION_MODEL, capabilities=["text", "json", "vision"], source="builtin"),
        ProviderModel(connection_id="deepseek", model_id=settings.DEEPSEEK_MODEL, label=settings.DEEPSEEK_MODEL, capabilities=["text", "json"], source="builtin"),
        ProviderModel(connection_id="glm", model_id=settings.GLM_MODEL, label=settings.GLM_MODEL, capabilities=["text", "json", "vision"], source="builtin"),
        ProviderModel(connection_id="grsai", model_id=settings.GRSAI_IMAGE_MODEL, label=f"{settings.GRSAI_IMAGE_MODEL}（生图）", capabilities=["image"], source="builtin"),
        ProviderModel(connection_id="ark", model_id=settings.ARK_IMAGE_MODEL, label=f"{settings.ARK_IMAGE_MODEL}（生图）", capabilities=["image"], source="builtin"),
        ProviderModel(connection_id="ark", model_id=settings.SEEDANCE_DEFAULT_MODEL, label=f"{settings.SEEDANCE_DEFAULT_MODEL}（Seedance）", capabilities=["video"], source="builtin"),
        ProviderModel(connection_id="ark", model_id=settings.ARK_VIDEO_MODEL, label=f"{settings.ARK_VIDEO_MODEL}（视频）", capabilities=["video"], source="builtin"),
        ProviderModel(connection_id="metaso_h3", model_id=METASO_H3_MODEL_ID, label="MiniMax-H3 · MetaSo", capabilities=["video"], source="builtin"),
        ProviderModel(connection_id="atlas_h3", model_id=ATLAS_H3_MODEL_ID, label="MiniMax-H3 · Atlas Cloud", capabilities=["video"], source="builtin"),
        ProviderModel(connection_id="comfyui_h3", model_id="h3-comfyui-local", label="MiniMax H3 · ComfyUI", capabilities=["video"], source="builtin"),
        ProviderModel(connection_id="edge_tts", model_id="edge-tts", label="Edge TTS 普通话", capabilities=["audio"], source="builtin"),
        ProviderModel(connection_id="disabled_audio", model_id="disabled", label="静音音轨", capabilities=["audio"], source="builtin"),
    ])
    return models


class ModelRegistry:
    def __init__(self) -> None:
        self._loaded = False
        self._custom_connections: dict[str, ProviderConnection] = {}
        self._custom_models: list[ProviderModel] = []
        self._routes: dict[str, dict[str, str]] = {}

    @property
    def path(self) -> Path:
        return Path(settings.OUTPUT_DIR) / "model_registry.json"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for raw in payload.get("connections", []) if isinstance(payload, dict) else []:
            try:
                item = ProviderConnection.model_validate(raw)
            except (TypeError, ValueError):
                continue
            if not item.builtin:
                self._custom_connections[item.id] = item
            else:
                # Persisted overrides for built-ins are applied on top of the
                # current runtime settings while keeping missing secrets intact.
                self._custom_connections[item.id] = item
        for raw in payload.get("models", []) if isinstance(payload, dict) else []:
            try:
                self._custom_models.append(ProviderModel.model_validate(raw))
            except (TypeError, ValueError):
                continue
        routes = payload.get("routes", {}) if isinstance(payload, dict) else {}
        if isinstance(routes, dict):
            self._routes = {
                str(route_id): {"connection_id": str(value.get("connection_id", "")), "model_id": str(value.get("model_id", ""))}
                for route_id, value in routes.items() if isinstance(value, dict)
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Built-in secrets already live in the runtime settings/.env. Never
        # duplicate them into the registry file when a built-in route is saved.
        persisted_connections = []
        for item in self._custom_connections.values():
            persisted = item.model_copy(update={"api_key": None}) if item.builtin else item
            persisted_connections.append(persisted.model_dump(mode="json"))
        payload = {
            "connections": persisted_connections,
            "models": [item.model_dump(mode="json") for item in self._custom_models if item.source != "builtin"],
            "routes": self._routes,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)

    def connections(self) -> list[ProviderConnection]:
        self._load()
        result = _builtin_connections()
        for connection_id, override in self._custom_connections.items():
            if connection_id in result:
                current = result[connection_id]
                values = override.model_dump(exclude_unset=True)
                if not override.api_key:
                    values["api_key"] = current.api_key
                result[connection_id] = current.model_copy(update=values)
            else:
                result[connection_id] = override
        return list(result.values())

    def connection(self, connection_id: str) -> ProviderConnection:
        item = next((item for item in self.connections() if item.id == connection_id), None)
        if item is None:
            raise ValueError(f"接口不存在: {connection_id}")
        return item

    def models(self) -> list[ProviderModel]:
        self._load()
        result = _builtin_models()
        builtin_ids = {(item.connection_id, item.model_id) for item in result}
        for item in self._custom_models:
            if (item.connection_id, item.model_id) in builtin_ids:
                result = [current for current in result if (current.connection_id, current.model_id) != (item.connection_id, item.model_id)]
            result.append(item)
        return [item for item in result if self._connection_enabled(item.connection_id)]

    def _connection_enabled(self, connection_id: str) -> bool:
        try:
            return self.connection(connection_id).enabled
        except ValueError:
            return False

    def models_for(self, connection_id: str, capability: str | None = None) -> list[ProviderModel]:
        return [
            item for item in self.models()
            if item.connection_id == connection_id and (capability is None or capability in item.capabilities)
        ]

    @staticmethod
    def _route_model_allowed(route_id: str, connection_id: str) -> bool:
        allowed = ROUTE_ALLOWED_CONNECTIONS.get(route_id)
        return allowed is None or connection_id in allowed

    def route_selection(self, route_id: str) -> tuple[str, str]:
        self._load()
        saved = self._routes.get(route_id)
        if saved and saved.get("connection_id") and saved.get("model_id"):
            connection_id = saved["connection_id"]
            if self._connection_enabled(connection_id) and any(
                item.connection_id == connection_id
                and item.model_id == saved["model_id"]
                and self._route_model_allowed(route_id, item.connection_id)
                for item in self.models()
            ):
                return connection_id, saved["model_id"]
        setting_key = next((item["setting_key"] for item in ROUTE_DEFINITIONS if item["id"] == route_id), "LLM_PROVIDER")
        if route_id == "image_generation":
            provider = str(settings.IMAGE_PROVIDER or "grsai")
            return self._connection_for_provider(provider), self._model_for_provider(provider, route_id)
        if route_id == "seedance_video":
            return "ark", settings.SEEDANCE_DEFAULT_MODEL
        if route_id == "h3_video":
            provider = str(settings.H3_PROVIDER or "metaso_h3")
            return self._connection_for_provider(provider), self._model_for_provider(provider, route_id)
        if route_id == "dialogue_audio":
            provider = str(settings.TTS_PROVIDER or "edge")
            connection_id = "edge_tts" if provider == "edge" else "disabled_audio"
            return connection_id, "edge-tts" if provider == "edge" else "disabled"
        provider = str(getattr(settings, setting_key, settings.LLM_PROVIDER) or "deepseek")
        if provider == "auto":
            if route_id == "reference_vision":
                provider = "claude" if settings.CLAUDE_API_KEY else str(settings.LLM_PROVIDER or "deepseek")
            elif route_id == "shot_count":
                provider = str(settings.BRIEF_ANALYSIS_PROVIDER or settings.LLM_PROVIDER)
                if provider == "auto":
                    provider = str(settings.LLM_PROVIDER or "deepseek")
            else:
                provider = str(settings.LLM_PROVIDER or "deepseek")
        if provider.startswith("custom:"):
            parts = provider.split(":", 2)
            if len(parts) == 3:
                return parts[1], parts[2]
        model = self._model_for_provider(provider, route_id)
        return self._connection_for_provider(provider), model

    @staticmethod
    def _connection_for_provider(provider: str) -> str:
        return {
            "ark_deepseek": "ark", "ark_doubao": "ark", "ark": "ark",
            "grsai": "grsai", "openlux": "openlux", "claude": "claude",
            "deepseek": "deepseek", "glm": "glm",
            "metaso_h3": "metaso_h3", "atlas_h3": "atlas_h3", "comfyui_h3": "comfyui_h3",
            "edge": "edge_tts", "disabled": "disabled_audio",
        }.get(provider, provider)

    @staticmethod
    def _model_for_provider(provider: str, route_id: str) -> str:
        # Capability-specific routes must resolve their dedicated model before
        # the generic LLM fallbacks below.  Otherwise a GRSAI/Ark image route
        # could incorrectly display the text model as its selected value.
        if route_id == "image_generation":
            if provider == "grsai":
                return settings.GRSAI_IMAGE_MODEL
            if provider in {"ark", "ark_doubao", "ark_deepseek"}:
                return settings.ARK_IMAGE_MODEL
        if route_id == "seedance_video" and provider in {"ark", "ark_doubao", "ark_deepseek"}:
            return settings.SEEDANCE_DEFAULT_MODEL
        if provider == "grsai":
            return settings.GRSAI_LLM_VISION_MODEL if route_id == "reference_vision" else settings.GRSAI_LLM_MODEL
        if provider == "openlux":
            return settings.OPENLUX_VISION_MODEL if route_id == "reference_vision" else settings.OPENLUX_MODEL
        if provider == "claude":
            return settings.CLAUDE_VISION_MODEL if route_id == "reference_vision" else settings.CLAUDE_MODEL
        if provider == "ark_deepseek":
            return settings.ARK_VISION_MODEL if route_id == "reference_vision" else settings.ARK_DEEPSEEK_MODEL
        if provider in {"ark", "ark_doubao"}:
            return settings.ARK_VISION_MODEL if route_id == "reference_vision" else settings.ARK_LLM_MODEL
        if provider == "glm":
            return settings.GLM_MODEL
        if provider == "metaso_h3":
            return METASO_H3_MODEL_ID
        if provider == "atlas_h3":
            return ATLAS_H3_MODEL_ID
        if provider == "comfyui_h3":
            return "h3-comfyui-local"
        return settings.DEEPSEEK_MODEL

    def set_route(self, route_id: str, connection_id: str, model_id: str) -> None:
        definition = next((item for item in ROUTE_DEFINITIONS if item["id"] == route_id), None)
        if definition is None:
            raise ValueError(f"不支持的功能路由: {route_id}")
        connection = self.connection(connection_id)
        model = next((item for item in self.models() if item.connection_id == connection_id and item.model_id == model_id), None)
        if model is None:
            raise ValueError(f"模型不存在或已停用: {connection_id}/{model_id}")
        if not self._route_model_allowed(route_id, connection_id):
            raise ValueError(f"接口 {connection_id} 暂未接入功能路由 {route_id}")
        required = definition["capability"]
        if required not in model.capabilities:
            raise ValueError(f"模型 {model_id} 尚未声明 {required} 能力")
        self._routes[route_id] = {"connection_id": connection_id, "model_id": model_id}
        self._save()
        from src.video_workflow.runtime_settings import runtime_settings
        provider = connection_id
        values: dict[str, Any] = {definition["setting_key"]: provider}
        if route_id == "image_generation":
            values["IMAGE_PROVIDER"] = connection_id
            values["GRSAI_IMAGE_MODEL" if connection_id == "grsai" else "ARK_IMAGE_MODEL"] = model_id
        elif route_id == "seedance_video":
            values["SEEDANCE_DEFAULT_MODEL"] = model_id
        elif route_id == "h3_video":
            values["H3_PROVIDER"] = connection_id
        elif route_id == "dialogue_audio":
            values["TTS_PROVIDER"] = "edge" if connection_id == "edge_tts" else "disabled"
        elif connection_id == "grsai":
            values["GRSAI_LLM_VISION_MODEL" if route_id == "reference_vision" else "GRSAI_LLM_MODEL"] = model_id
        elif connection_id == "openlux":
            values["OPENLUX_VISION_MODEL" if route_id == "reference_vision" else "OPENLUX_MODEL"] = model_id
        elif connection_id == "claude":
            values["CLAUDE_VISION_MODEL" if route_id == "reference_vision" else "CLAUDE_MODEL"] = model_id
        elif connection_id == "ark":
            values[definition["setting_key"]] = "ark_deepseek" if model_id.startswith("deepseek-v4") else "ark"
            if model_id.startswith("deepseek-v4") and route_id != "reference_vision":
                values["ARK_DEEPSEEK_MODEL"] = model_id
        elif connection_id.startswith("custom_"):
            values[definition["setting_key"]] = f"custom:{connection_id}:{model_id}"
        runtime_settings.update(values)

    def add_connection(self, values: dict[str, Any]) -> ProviderConnection:
        self._load()
        name = str(values.get("name") or "自定义接口").strip()
        base_url = str(values.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ValueError("Base URL 不能为空")
        protocol = str(values.get("protocol") or "openai_chat")
        if protocol not in {"openai_chat", "ark", "anthropic_messages", "gemini"}:
            raise ValueError("不支持的接口协议")
        connection = ProviderConnection(
            id=f"custom_{uuid4().hex[:12]}", name=name, protocol=protocol, base_url=base_url,
            api_key=str(values.get("api_key") or "") or None,
            models_url=str(values.get("models_url") or "") or None,
            api_key_header=str(values.get("api_key_header") or "Authorization"), builtin=False,
        )
        self._custom_connections[connection.id] = connection
        self._save()
        return connection

    def update_connection(self, connection_id: str, values: dict[str, Any]) -> ProviderConnection:
        self._load()
        current = self.connection(connection_id)
        if current.builtin and connection_id not in self._custom_connections:
            self._custom_connections[connection_id] = current
        stored = self._custom_connections.get(connection_id, current)
        update = {key: value for key, value in values.items() if key in {
            "name", "protocol", "base_url", "models_url", "api_key_header", "enabled",
            "last_checked_at", "last_status", "last_error",
            "last_discovery_at", "last_discovery_status", "last_discovery_error", "last_discovery_count",
            "last_test_at", "last_test_status", "last_test_error",
        }}
        if values.get("api_key"):
            update["api_key"] = values["api_key"]
        update["updated_at"] = _now()
        self._custom_connections[connection_id] = stored.model_copy(update=update)
        self._save()
        return self.connection(connection_id)

    def delete_connection(self, connection_id: str) -> None:
        self._load()
        current = self.connection(connection_id)
        if current.builtin:
            raise ValueError("内置接口请停用，不删除")
        self._custom_connections.pop(connection_id, None)
        self._custom_models = [item for item in self._custom_models if item.connection_id != connection_id]
        self._routes = {key: value for key, value in self._routes.items() if value.get("connection_id") != connection_id}
        self._save()

    def add_model(self, connection_id: str, values: dict[str, Any]) -> ProviderModel:
        self._load()
        self.connection(connection_id)
        model_id = str(values.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("模型 ID 不能为空")
        capabilities = [str(item) for item in values.get("capabilities", ["text", "json"]) if str(item)]
        source = str(values.get("source") or "manual")
        if source not in {"manual", "discovered"}:
            source = "manual"
        item = ProviderModel(
            connection_id=connection_id, model_id=model_id, label=str(values.get("label") or model_id),
            capabilities=list(dict.fromkeys(capabilities)),
            input_price_per_million=values.get("input_price_per_million"),
            output_price_per_million=values.get("output_price_per_million"),
            cache_price_per_million=values.get("cache_price_per_million"), source=source,
            last_seen_at=_now(),
        )
        self._custom_models = [m for m in self._custom_models if not (m.connection_id == connection_id and m.model_id == model_id)]
        self._custom_models.append(item)
        self._save()
        return item

    def _headers(self, connection: ProviderConnection) -> dict[str, str]:
        if not connection.api_key:
            return {}
        value = connection.api_key if connection.api_key_header.lower() != "authorization" else f"Bearer {connection.api_key}"
        return {connection.api_key_header: value}

    async def discover(self, connection_id: str) -> dict[str, Any]:
        connection = self.connection(connection_id)
        if connection.protocol != "openai_chat":
            message = "此协议未提供标准模型列表，请在模型目录手动添加"
            self.update_connection(connection_id, {
                "last_discovery_at": _now(), "last_discovery_status": "manual",
                "last_discovery_error": message, "last_discovery_count": None,
            })
            return {"supported": False, "requires_manual": True, "message": message}
        models_url = connection.models_url or f"{connection.base_url.rstrip('/')}/models"
        if not connection.models_url and not models_url.endswith("/models"):
            models_url += "/models"
        try:
            async with httpx.AsyncClient(timeout=20, trust_env=True) as client:
                response = await client.get(models_url, headers=self._headers(connection))
                if response.status_code in {404, 405}:
                    message = f"该接口未提供标准模型列表（HTTP {response.status_code}），请使用内置或手动模型目录"
                    self.update_connection(connection_id, {
                        "last_discovery_at": _now(), "last_discovery_status": "manual",
                        "last_discovery_error": message, "last_discovery_count": None,
                    })
                    return {"supported": False, "requires_manual": True, "message": message, "status_code": response.status_code}
                if response.status_code >= 400:
                    message = f"模型列表接口返回 HTTP {response.status_code}"
                    self.update_connection(connection_id, {
                        "last_discovery_at": _now(), "last_discovery_status": "discovery_failed",
                        "last_discovery_error": message, "last_discovery_count": None,
                    })
                    return {"supported": False, "requires_manual": True, "message": message, "status_code": response.status_code}
                payload = response.json()
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                added = []
                for row in rows:
                    model_id = str(row.get("id") if isinstance(row, dict) else row).strip()
                    if not model_id:
                        continue
                    existing = next((item for item in self.models() if item.connection_id == connection_id and item.model_id == model_id), None)
                    capabilities = existing.capabilities if existing else ["text", "json"]
                    added.append(self.add_model(connection_id, {"model_id": model_id, "label": model_id, "capabilities": capabilities, "source": "discovered"}))
                self.update_connection(connection_id, {
                    "last_discovery_at": _now(), "last_discovery_status": "ok",
                    "last_discovery_error": "", "last_discovery_count": len(added),
                })
                return {"supported": True, "models": [item.model_dump(mode="json") for item in added], "count": len(added)}
        except (httpx.HTTPError, ValueError) as exc:
            message = f"获取模型列表失败: {type(exc).__name__}"
            self.update_connection(connection_id, {
                "last_discovery_at": _now(), "last_discovery_status": "discovery_failed",
                "last_discovery_error": message, "last_discovery_count": None,
            })
            return {"supported": False, "requires_manual": True, "message": message}

    async def test_connection(self, connection_id: str, model_id: str | None = None) -> dict[str, Any]:
        connection = self.connection(connection_id)
        if not connection.api_key:
            message = "请先配置 API Key"
            self.update_connection(connection_id, {
                "last_test_at": _now(), "last_test_status": "test_failed", "last_test_error": message,
            })
            return {"ok": False, "message": message}
        if connection.protocol != "openai_chat":
            message = "此协议的连通测试请在对应服务适配器中完成"
            self.update_connection(connection_id, {
                "last_test_at": _now(), "last_test_status": "test_failed", "last_test_error": message,
            })
            return {"ok": False, "message": message}
        chosen = model_id or (self.models_for(connection_id, "text") or [None])[0]
        if isinstance(chosen, ProviderModel):
            chosen = chosen.model_id
        if not chosen:
            return {"ok": False, "message": "请先添加一个模型 ID"}
        endpoint = f"{connection.base_url.rstrip('/')}/chat/completions"
        body = {"model": chosen, "messages": [{"role": "user", "content": "回复：好"}], "max_tokens": 4, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=30, trust_env=True) as client:
                response = await client.post(endpoint, headers={**self._headers(connection), "Content-Type": "application/json"}, json=body)
            if response.status_code >= 400:
                message = f"测试请求返回 HTTP {response.status_code}"
                self.update_connection(connection_id, {
                    "last_test_at": _now(), "last_test_status": "test_failed", "last_test_error": message,
                })
                return {"ok": False, "message": message, "status_code": response.status_code}
            self.update_connection(connection_id, {
                "last_test_at": _now(), "last_test_status": "ok", "last_test_error": "",
            })
            payload = response.json() if response.content else {}
            return {"ok": True, "model_id": chosen, "usage": payload.get("usage") if isinstance(payload, dict) else None}
        except httpx.HTTPError as exc:
            message = f"测试请求失败: {type(exc).__name__}"
            self.update_connection(connection_id, {
                "last_test_at": _now(), "last_test_status": "test_failed", "last_test_error": message,
            })
            return {"ok": False, "message": message}

    def public_payload(self) -> dict[str, Any]:
        connections = []
        for item in self.connections():
            if item.hidden:
                continue
            connections.append({
                "id": item.id, "name": item.name, "protocol": item.protocol, "base_url": item.base_url,
                "models_url": item.models_url or "", "api_key_header": item.api_key_header,
                "enabled": item.enabled, "builtin": item.builtin, "configured": bool(item.api_key),
                "masked": f"••••{item.api_key[-4:]}" if item.api_key else "未配置",
                "last_checked_at": item.last_checked_at, "last_status": item.last_status, "last_error": item.last_error,
                "last_discovery_at": item.last_discovery_at, "last_discovery_status": item.last_discovery_status,
                "last_discovery_error": item.last_discovery_error, "last_discovery_count": item.last_discovery_count,
                "last_test_at": item.last_test_at, "last_test_status": item.last_test_status,
                "last_test_error": item.last_test_error,
            })
        visible_connection_ids = {item["id"] for item in connections}
        models = [
            item.model_dump(mode="json")
            for item in self.models()
            if item.connection_id in visible_connection_ids
        ]
        routes = []
        for definition in ROUTE_DEFINITIONS:
            connection_id, model_id = self.route_selection(definition["id"])
            options = []
            for model in self.models():
                if definition["capability"] not in model.capabilities:
                    continue
                if not self._route_model_allowed(definition["id"], model.connection_id):
                    continue
                provider = self.connection(model.connection_id)
                options.append({
                    "value": f"{model.connection_id}::{model.model_id}",
                    "connection_id": model.connection_id,
                    "model_id": model.model_id,
                    "label": f"{provider.name} / {model.label or model.model_id}",
                })
            routes.append({
                "id": definition["id"], "label": definition["label"], "capability": definition["capability"],
                "setting_key": definition["setting_key"], "selected": f"{connection_id}::{model_id}",
                "selected_connection_id": connection_id, "selected_model_id": model_id, "options": options,
                "description": definition.get("description", ""),
            })
        return {"connections": connections, "models": models, "routes": routes, "path": str(self.path)}


model_registry = ModelRegistry()
