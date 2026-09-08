from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from src.video_workflow.config import settings
from src.video_workflow.core.orchestrator import WorkflowOrchestrator, create_llm_generator, resolve_reference_llm_provider
from src.video_workflow.domain import (
    ApprovalStatus,
    Asset,
    AssetRole,
    AssetType,
    CharacterAppearanceProfile,
    CharacterProfile,
    DialogueTurn,
    FirstFrameCompleteness,
    GenerationMode,
    JobStatus,
    Project,
    ProjectAnalysisDraft,
    ProjectBrief,
    ProjectStatus,
    ProductionSeries,
    SceneConsistencyMode,
    SceneProfile,
    SeedanceReferenceMode,
    SeriesAsset,
    ScriptDurationAssessment,
    ScriptRewriteDraft,
    Shot,
    ShotContinuityMode,
    ShotSplitPreview,
    ShotSplitSegment,
    StyleAnalysisDraft,
    StyleProfile,
    VisualBeat,
    VoiceEvent,
    new_id,
)
from src.video_workflow.generators.image import GrsaiImageGenerator, ImageGenerationError, create_image_generator
from src.video_workflow.h3_prompt_skills import (
    get_h3_prompt_skill,
    h3_prompt_system_instruction,
)
from src.video_workflow.integrations.comfyui import h3_frames_for_seconds
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.speech_budget import (
    fit_voice_event_payloads,
    speech_budget_for_duration,
    spoken_character_count,
)
from src.video_workflow.storage import ProjectStore
from src.video_workflow.types import Scene

logger = logging.getLogger(__name__)

SEEDANCE_PROMPT_VERSION = "seedance-2.0-director-v6"
H3_DIRECTOR_VERSION = "h3-director-v7"

CHARACTER_REFERENCE_SHEET_LAYOUT = """生成一张横向 16:9 的单一角色四视图设定板，角色可以是人物、动物或拟人角色；画面严格只呈现同一个角色的四个视图，不是四个不同角色：
1. 左侧约占画面三分之一：人物使用大尺寸正脸近照，动物使用大尺寸正面头部近照；完整呈现人物的发型、脸型、五官与颈肩细节，或动物的品种特征、耳形、口鼻、眼睛、毛色与独特斑纹；
2. 右侧约占画面三分之二：依次并排放置同一角色的全身正面、标准侧面、全身背面，三者均从头顶到鞋底或足爪完整入画；人物使用中性站姿，动物使用自然四足站姿，三个全身视图比例、体型和基线一致；
3. 四个视图必须保持完全一致的身份、年龄或物种、脸型或头部结构、发型或毛发、体型、服装、项圈及其他配饰、材质与配色；动物必须固定品种、毛长、毛色分区、斑纹位置、耳尾形态和体型，侧面与背面准确补全对应结构；
4. 使用纯净统一的中性影棚背景和均匀柔光，不添加场景、动作叙事、边框、分隔线、文字标签、尺寸标注、Logo 或水印；
5. 保证版面整洁、各视图互不遮挡，不裁切头顶、身体、鞋子或足爪，不增加其他角度、表情小图、道具特写或第二个角色。"""

STYLIZED_2D_CHARACTER_RULE = (
    "人物和动物必须保持明显的二维手绘动画造型：清晰手绘勾线、概括化五官与皮肤、动画化头身比例、"
    "平涂结合柔和局部厚涂；禁止真人照片、摄影棚人像、毛孔级皮肤、照片级写实人物、3D/PBR/CGI 人物。"
    "生活道具可以保留真实结构、磨损和材质细节，但不得把人物渲染成真人。"
)


def _analysis_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "；".join(_analysis_text(item) for item in value if _analysis_text(item))
    if isinstance(value, dict):
        return "；".join(
            f"{key}：{_analysis_text(item)}"
            for key, item in value.items()
            if _analysis_text(item)
        )
    return str(value).strip()


def _normalize_project_analysis_payload(payload: dict, fallback_count: int) -> dict:
    """Accept common GPT/Claude JSON variations without discarding paid output."""
    if not isinstance(payload, dict):
        raise ValueError("剧本分析结果必须是 JSON 对象")
    nested = payload.get("analysis") or payload.get("result") or payload.get("data")
    if isinstance(nested, dict) and not any(
        key in payload for key in ("visual_style", "pacing", "characters")
    ):
        payload = nested

    normalized: dict[str, object] = {}
    for key in (
        "visual_style",
        "pacing",
        "audience",
        "style_bible",
        "negative_prompt",
        "delivery_notes",
        "shot_count_reason",
    ):
        normalized[key] = _analysis_text(payload.get(key))

    raw_count = payload.get("recommended_shot_count", fallback_count)
    if isinstance(raw_count, str):
        match = re.search(r"\d+", raw_count)
        raw_count = match.group(0) if match else fallback_count
    try:
        normalized["recommended_shot_count"] = int(float(raw_count))
    except (TypeError, ValueError):
        normalized["recommended_shot_count"] = fallback_count

    raw_characters = (
        payload.get("characters")
        or payload.get("character_profiles")
        or payload.get("roles")
        or []
    )
    if isinstance(raw_characters, dict):
        if "name" in raw_characters:
            raw_characters = [raw_characters]
        else:
            raw_characters = [
                ({"name": name, **value} if isinstance(value, dict) else {"name": name, "description": value})
                for name, value in raw_characters.items()
            ]
    if isinstance(raw_characters, str):
        try:
            parsed_characters = json.loads(raw_characters)
            raw_characters = parsed_characters if isinstance(parsed_characters, (list, dict)) else []
            if isinstance(raw_characters, dict):
                raw_characters = [raw_characters]
        except json.JSONDecodeError:
            raw_characters = []
    characters: list[dict[str, object]] = []
    if isinstance(raw_characters, list):
        for item in raw_characters:
            if not isinstance(item, dict):
                continue
            name = _analysis_text(item.get("name") or item.get("character_name"))
            if not name:
                continue
            character_id = item.get("character_id") or item.get("id")
            characters.append(
                {
                    "character_id": _analysis_text(character_id) or None,
                    "name": name,
                    "description": _analysis_text(item.get("description") or item.get("appearance")),
                    "wardrobe": _analysis_text(item.get("wardrobe") or item.get("costume")),
                    "voice_description": _analysis_text(item.get("voice_description") or item.get("voice")),
                    "reference_observations": _analysis_text(
                        item.get("reference_observations") or item.get("reference_notes")
                    ),
                }
            )
    notes = payload.get("analysis_notes") or payload.get("notes") or []
    if isinstance(notes, str):
        notes = [
            item.strip(" -•\t")
            for item in re.split(r"[\n；]+", notes)
            if item.strip(" -•\t")
        ]
    elif isinstance(notes, dict):
        notes = [f"{key}：{_analysis_text(value)}" for key, value in notes.items()]
    elif not isinstance(notes, list):
        notes = []
    normalized_notes = [_analysis_text(item) for item in notes if _analysis_text(item)]
    known_names = {str(character["name"]).strip().casefold() for character in characters}
    for index, note in enumerate(normalized_notes):
        match = re.search(
            r"(?P<name>[^，。；：]{1,36}?)(?:是否需要|是否要|需不需要)(?:单独|独立)(?:设定|设计)(?:形象|角色)?",
            note,
        )
        if not match:
            continue
        name = match.group("name").strip()
        for prefix in ("需确认", "确认", "剧中出现的", "剧中", "画面中的", "出现的", "中的"):
            if prefix in name:
                name = name.rsplit(prefix, 1)[-1].strip()
        name = re.sub(r"(?:角色|形象)$", "", name).strip()
        if not name or len(name) > 20:
            continue
        if name.casefold() in known_names:
            normalized_notes[index] = f"已默认将{name}作为独立固定形象建档；如不需要，可在上方角色草稿中删除。"
            continue
        characters.append(
            {
                "character_id": None,
                "name": name,
                "description": f"根据剧本为{name}建立独立、稳定、可跨镜复用的固定形象；请在应用前直接补充或修改识别特征。",
                "wardrobe": "无服装；如有项圈、挂件或其他固定配饰，请在应用前直接补充。",
                "voice_description": "无台词；如有叫声或拟人台词，请在应用前直接补充。",
                "reference_observations": f"由分析确认项自动建档：{note}",
            }
        )
        known_names.add(name.casefold())
        normalized_notes[index] = f"已默认将{name}作为独立固定形象建档；如不需要，可在上方角色草稿中删除。"
    normalized["characters"] = characters
    normalized["analysis_notes"] = normalized_notes
    return normalized


def _raw_response_hint(generator: object) -> str:
    path = getattr(generator, "last_response_path", None)
    return f"；已付费的原始结果保存在 {path}" if path else ""


class KeyframeBusyError(ValueError):
    """The same shot already has an in-flight keyframe request."""


class ShotVersionConflictError(ValueError):
    """An older whole-shot form must not overwrite newly saved work."""


class ShotSplitBusyError(ValueError):
    """A shot with in-flight generation work must not be replaced."""


class SceneReferenceConflictError(ValueError):
    """A stale scene editor or duplicate generation must not overwrite work."""


class ProjectCoverBusyError(ValueError):
    """Only one paid cover generation may run for a project at a time."""


class ProjectService:
    def __init__(self, store: ProjectStore):
        self.store = store
        self._active_keyframes: set[str] = set()
        self._active_scene_references: set[str] = set()
        self._active_project_covers: set[str] = set()

    def create_project(self, brief: ProjectBrief) -> Project:
        project = Project(brief=brief)
        self.project_dir(project.id).mkdir(parents=True, exist_ok=True)
        return self.store.save_project(project)

    def require_series(self, series_id: str) -> ProductionSeries:
        series = self.store.get_series(series_id)
        if series is None:
            raise KeyError(f"Series not found: {series_id}")
        return series

    def series_dir(self, series_id: str) -> Path:
        return settings.PROJECTS_DIR / "_series" / series_id

    @staticmethod
    def _remap_character(
        character: CharacterProfile,
        character_id: str,
        reference_ids: dict[str, str],
        *,
        renew_appearance_ids: bool = False,
    ) -> CharacterProfile:
        appearances = [
            appearance.model_copy(
                deep=True,
                update={
                    "id": new_id("appearance") if renew_appearance_ids else appearance.id,
                    "reference_asset_ids": [
                        reference_ids[asset_id]
                        for asset_id in appearance.reference_asset_ids
                        if asset_id in reference_ids
                    ],
                },
            )
            for appearance in character.appearance_profiles
        ]
        return character.model_copy(
            deep=True,
            update={
                "id": character_id,
                "reference_asset_ids": [
                    reference_ids[asset_id]
                    for asset_id in character.reference_asset_ids
                    if asset_id in reference_ids
                ],
                "appearance_profiles": appearances,
            },
        )

    def save_project_as_series(
        self,
        project_id: str,
        name: str = "",
        description: str = "",
    ) -> tuple[ProductionSeries, Project]:
        """Create or refresh a durable series identity from one episode."""
        project = self.require_project(project_id)
        current = self.store.get_series(project.series_id) if project.series_id else None
        series = current or ProductionSeries(name=name.strip() or project.brief.title)
        if name.strip():
            series.name = name.strip()
        if description.strip() or current is None:
            series.description = description.strip()

        project_assets = {asset.id: asset for asset in self.store.list_assets(project_id)}
        existing_characters = {
            character.name.strip().casefold(): character
            for character in (current.characters if current else [])
        }
        series_character_ids = {
            character.id: existing_characters.get(
                character.name.strip().casefold(),
                character,
            ).id
            for character in project.characters
        }
        referenced_ids: list[str] = [
            asset.id
            for asset in project_assets.values()
            if asset.role in {AssetRole.STYLE, AssetRole.PROP}
        ]
        if project.style_profile:
            referenced_ids.extend(project.style_profile.reference_asset_ids)
        character_owner: dict[str, str] = {}
        for character in project.characters:
            referenced_ids.extend(character.reference_asset_ids)
            character_owner.update({asset_id: series_character_ids[character.id] for asset_id in character.reference_asset_ids})
            for appearance in character.appearance_profiles:
                referenced_ids.extend(appearance.reference_asset_ids)
                character_owner.update({asset_id: series_character_ids[character.id] for asset_id in appearance.reference_asset_ids})
        referenced_ids = list(dict.fromkeys(referenced_ids))

        destination = self.series_dir(series.id) / "assets"
        reference_map: dict[str, str] = {}
        snapshots: list[SeriesAsset] = []
        for source_id in referenced_ids:
            source_asset = project_assets.get(source_id)
            if source_asset is None:
                continue
            source_path = resolve_media_path(source_asset.path)
            if not source_path.is_file():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_id = new_id("series_asset")
            target = destination / f"{snapshot_id}{source_path.suffix.lower()}"
            shutil.copy2(source_path, target)
            reference_map[source_id] = snapshot_id
            snapshots.append(
                SeriesAsset(
                    id=snapshot_id,
                    type=source_asset.type,
                    role=source_asset.role,
                    name=source_asset.name,
                    path=str(target.resolve()),
                    mime_type=source_asset.mime_type,
                    character_id=character_owner.get(source_id, source_asset.character_id),
                    description=source_asset.description,
                    tags=list(source_asset.tags),
                    approved=source_asset.approved,
                    sha256=source_asset.sha256,
                    size_bytes=source_asset.size_bytes,
                )
            )

        current_names = {character.name.strip().casefold() for character in project.characters}
        preserved_characters = [
            character.model_copy(deep=True)
            for character in (current.characters if current else [])
            if character.name.strip().casefold() not in current_names
        ]
        preserved_asset_ids = {
            asset_id
            for character in preserved_characters
            for asset_id in (
                *character.reference_asset_ids,
                *(
                    asset_id
                    for appearance in character.appearance_profiles
                    for asset_id in appearance.reference_asset_ids
                ),
            )
        }
        if current:
            current_prop_names = {
                asset.name.strip().casefold()
                for asset in snapshots
                if asset.role == AssetRole.PROP
            }
            snapshots.extend(
                asset.model_copy(deep=True)
                for asset in current.assets
                if asset.id in preserved_asset_ids
                or (
                    asset.role == AssetRole.PROP
                    and asset.name.strip().casefold() not in current_prop_names
                )
            )
        series.characters = [
            self._remap_character(character, series_character_ids[character.id], reference_map)
            for character in project.characters
        ] + preserved_characters
        series.assets = snapshots
        series.visual_style = self.reusable_series_style_text(project)
        series.style_bible = self.reusable_series_style_bible(project)
        series.negative_prompt = project.brief.negative_prompt
        series.style_profile = (
            project.style_profile.model_copy(
                deep=True,
                update={
                    "palette": "",
                    "lighting": "",
                    # These fields extracted from one episode often name its
                    # hero prop or location, so they are not series constants.
                    "composition": "",
                    "motion_language": "",
                    "analysis_summary": self.reusable_series_style_text(project),
                    "reference_asset_ids": [
                        reference_map[asset_id]
                        for asset_id in project.style_profile.reference_asset_ids
                        if asset_id in reference_map
                    ]
                },
            )
            if project.style_profile
            else None
        )
        saved_series = self.store.save_series(series)
        project.series_id = saved_series.id
        if project.episode_number is None:
            used_numbers = [
                item.episode_number or 0
                for item in self.store.list_projects()
                if item.series_id == saved_series.id and item.id != project.id
            ]
            project.episode_number = max(used_numbers, default=0) + 1
        return saved_series, self.store.save_project(project)

    def create_series_episode(
        self,
        series_id: str,
        brief: ProjectBrief,
        character_ids: list[str] | None = None,
    ) -> Project:
        """Create a clean episode while inheriting series style and selected cast."""
        series = self.require_series(series_id)
        known_ids = {character.id for character in series.characters}
        selected_ids = known_ids if character_ids is None else set(character_ids)
        unknown = selected_ids - known_ids
        if unknown:
            raise ValueError(f"系列中不存在这些角色: {', '.join(sorted(unknown))}")
        inherited_brief = brief.model_copy(
            update={
                "visual_style": series.visual_style,
                "negative_prompt": series.negative_prompt,
            }
        )
        project = self.create_project(inherited_brief)
        existing_numbers = [
            item.episode_number or 0
            for item in self.store.list_projects()
            if item.series_id == series.id and item.id != project.id
        ]
        project.series_id = series.id
        project.episode_number = max(existing_numbers, default=0) + 1
        project.style_bible = series.style_bible

        character_map = {
            character.id: new_id("character")
            for character in series.characters
            if character.id in selected_ids
        }
        series_assets = {asset.id: asset for asset in series.assets}
        required_asset_ids: set[str] = set()
        if series.style_profile:
            required_asset_ids.update(series.style_profile.reference_asset_ids)
        required_asset_ids.update(
            asset.id
            for asset in series.assets
            if asset.role in {AssetRole.STYLE, AssetRole.PROP}
        )
        for character in series.characters:
            if character.id not in selected_ids:
                continue
            required_asset_ids.update(character.reference_asset_ids)
            for appearance in character.appearance_profiles:
                required_asset_ids.update(appearance.reference_asset_ids)

        target_dir = self.project_dir(project.id) / "series_references"
        target_dir.mkdir(parents=True, exist_ok=True)
        asset_map: dict[str, str] = {}
        for source_id in required_asset_ids:
            source_asset = series_assets.get(source_id)
            if source_asset is None:
                continue
            source_path = resolve_media_path(source_asset.path)
            if not source_path.is_file():
                continue
            asset_id = new_id("asset")
            target = target_dir / f"{asset_id}{source_path.suffix.lower()}"
            shutil.copy2(source_path, target)
            asset_map[source_id] = asset_id
            self.store.save_asset(
                Asset(
                    id=asset_id,
                    project_id=project.id,
                    type=source_asset.type,
                    role=source_asset.role,
                    name=source_asset.name,
                    path=str(target.resolve()),
                    mime_type=source_asset.mime_type,
                    character_id=character_map.get(source_asset.character_id or ""),
                    description=source_asset.description,
                    tags=list(dict.fromkeys([*source_asset.tags, f"series:{series.id}"])),
                    approved=source_asset.approved,
                    sha256=source_asset.sha256,
                    size_bytes=source_asset.size_bytes,
                )
            )

        project.characters = [
            self._remap_character(
                character,
                character_map[character.id],
                asset_map,
                renew_appearance_ids=True,
            )
            for character in series.characters
            if character.id in selected_ids
        ]
        project.style_profile = (
            series.style_profile.model_copy(
                deep=True,
                update={
                    "reference_asset_ids": [
                        asset_map[asset_id]
                        for asset_id in series.style_profile.reference_asset_ids
                        if asset_id in asset_map
                    ]
                },
            )
            if series.style_profile
            else None
        )
        return self.store.save_project(project)

    def project_dir(self, project_id: str) -> Path:
        return settings.PROJECTS_DIR / project_id

    def require_project(self, project_id: str) -> Project:
        project = self.store.get_project(project_id)
        if project is None:
            raise KeyError(f"Project not found: {project_id}")
        return project

    def require_shot(self, shot_id: str) -> Shot:
        shot = self.store.get_shot(shot_id)
        if shot is None:
            raise KeyError(f"Shot not found: {shot_id}")
        return shot

    @staticmethod
    def _clarify_stylized_2d_style(value: str) -> str:
        """Disambiguate style wording that can accidentally request a photo."""
        text = value.strip()
        if not text or not any(marker in text for marker in ("二维", "二次元", "手绘", "动画")):
            return text
        text = text.replace("二次元写实融合渲染", "二维手绘动画插画渲染")
        text = text.replace("二次元写实融合", "二维手绘动画插画")
        text = text.replace("二次元萌感+中式写实特征", "二次元萌感＋中式动画角色特征")
        text = text.replace("二次元萌感 + 中式写实特征", "二次元萌感＋中式动画角色特征")
        if STYLIZED_2D_CHARACTER_RULE not in text:
            text = "；".join(part for part in (text.strip("；"), STYLIZED_2D_CHARACTER_RULE) if part)
        return text

    @classmethod
    def project_style_text(cls, project: Project) -> str:
        profile = project.style_profile
        if profile and profile.approved:
            fields = [
                profile.name,
                profile.medium,
                profile.palette,
                profile.lighting,
                profile.camera_language,
                profile.composition,
                profile.texture,
                profile.motion_language,
                profile.analysis_summary,
            ]
            return cls._clarify_stylized_2d_style(
                "；".join(item.strip() for item in fields if item and item.strip())
            )
        return cls._clarify_stylized_2d_style(project.brief.visual_style or project.style_bible)

    @classmethod
    def reusable_series_style_text(cls, project: Project) -> str:
        """Keep rendering language while leaving location lighting to each episode."""
        profile = project.style_profile
        if profile and profile.approved:
            values = (
                profile.name,
                profile.medium,
                profile.texture,
                profile.camera_language,
            )
            return cls._clarify_stylized_2d_style(
                "；".join(value.strip() for value in values if value and value.strip())
            )
        return cls._clarify_stylized_2d_style(project.brief.visual_style or project.style_bible)

    @classmethod
    def reusable_series_style_bible(cls, project: Project) -> str:
        style = cls.reusable_series_style_text(project)
        return (
            f"【系列固定画风】{style}\n"
            "【逐集场景规则】继承绘制媒介、线条、材质、人物比例、镜头语言和动作语言；"
            "每一集的地点、陈设、道具、主光源方向、冷暖色温、天气和时段只按本集场景档案执行，"
            "不得混入上一集其他地点的环境与光影。"
        )

    @classmethod
    def project_rendering_style_text(cls, project: Project) -> str:
        """Return only visual traits that are safe to reuse in every location.

        A style reference is often a finished shot. Its palette, lighting,
        framing and props describe that shot rather than the whole film. Scene
        profiles own those details; the project level contributes only the
        medium and rendering language when a scene profile is active.
        """
        profile = project.style_profile
        if profile and profile.approved:
            return cls._clarify_stylized_2d_style(
                "；".join(
                    value.strip()
                    for value in (profile.name, profile.medium)
                    if value and value.strip()
                )
            )
        return cls._clarify_stylized_2d_style(project.brief.visual_style or project.style_bible)

    @classmethod
    def character_rendering_style_text(cls, project: Project) -> str:
        """Return character medium/texture without scene-specific lighting."""
        profile = project.style_profile
        if profile and profile.approved:
            return cls._clarify_stylized_2d_style(
                "；".join(
                    value.strip()
                    for value in (profile.name, profile.medium, profile.texture)
                    if value and value.strip()
                )
            )
        return cls._clarify_stylized_2d_style(project.brief.visual_style or project.style_bible)

    @classmethod
    def shot_style_text(cls, project: Project, shot: Shot) -> str:
        """Resolve project rendering plus the authoritative local environment."""
        profiles = cls.scene_profiles_for_shot(project, shot)
        if not profiles:
            return cls.project_style_text(project)
        rendering = cls.project_rendering_style_text(project)
        if len(profiles) > 1:
            return rendering
        profile = profiles[0]
        scene = "；".join(
            value.strip()
            for value in (profile.description, profile.continuity_notes)
            if value and value.strip()
        )
        return "；".join(
            part for part in (
                rendering,
                f"本镜场景“{profile.name}”的环境、光线、色彩和陈设仅按场景档案执行：{scene}",
                "场景档案优先于全局风格分析或旧分镜中与之冲突的环境描述",
            ) if part
        )

    @staticmethod
    def scene_profile_for_shot(project: Project, shot: Shot) -> SceneProfile | None:
        profiles = ProjectService.scene_profiles_for_shot(project, shot)
        return profiles[0] if profiles else None

    @staticmethod
    def scene_profiles_for_shot(
        project: Project,
        shot: Shot,
        *,
        include_strict: bool = False,
    ) -> list[SceneProfile]:
        if not shot.use_scene_profile and not (
            include_strict and project.scene_consistency_mode == SceneConsistencyMode.STRICT
        ):
            return []
        profile_ids = list(dict.fromkeys([
            *(shot.scene_profile_ids or []),
            *([shot.scene_profile_id] if shot.scene_profile_id else []),
        ]))[:2]
        if not profile_ids:
            return []
        profile_map = {profile.id: profile for profile in project.scene_profiles}
        return [profile_map[profile_id] for profile_id in profile_ids if profile_id in profile_map]

    @staticmethod
    def scene_transition_contract(profiles: list[SceneProfile]) -> str:
        if not profiles:
            return ""
        if len(profiles) == 1:
            profile = profiles[0]
            return (
                f"【场景档案｜本镜最高优先级】本镜只使用场景“{profile.name}”："
                f"{profile.description}；{profile.continuity_notes or '保持空间布局、固定陈设、材质与光线'}；"
                "环境、光线、色彩和陈设只按此档案执行"
            )
        start, destination = profiles[:2]
        start_text = ProjectService._compact_prompt_text(
            "；".join(filter(None, [start.description, start.continuity_notes])),
            300,
        )
        destination_text = ProjectService._compact_prompt_text(
            "；".join(filter(None, [destination.description, destination.continuity_notes])),
            300,
        )
        return (
            "【跨场景档案｜严格按顺序执行】"
            f"0.00 秒与转场前只使用起始场景“{start.name}”：{start_text}；"
            f"角色穿过门、墙、传送通道或完成地点切换后，只使用目标场景“{destination.name}”：{destination_text}；"
            "两个场景是前后接续关系，禁止在同一阶段混合两套空间、陈设、色彩或光源，禁止把两个房间拼贴在同一画面"
        )

    def update_project(self, project: Project) -> Project:
        existing = self.store.get_project(project.id)
        if existing:
            latest_scenes = {item.id: item for item in existing.scene_profiles}
            project.scene_profiles = [
                latest_scenes[item.id] if item.id in latest_scenes and latest_scenes[item.id].version > item.version else item
                for item in project.scene_profiles
            ] + [item for item in existing.scene_profiles if item.id not in {scene.id for scene in project.scene_profiles}]
        style_changed = bool(
            existing
            and (
                existing.style_profile != project.style_profile
                or existing.style_bible != project.style_bible
                or existing.brief.visual_style != project.brief.visual_style
                or existing.brief.negative_prompt != project.brief.negative_prompt
            )
        )
        characters_changed = bool(existing and existing.characters != project.characters)
        if existing and existing.brief.story != project.brief.story and project.ai_recommended_shot_count == existing.ai_recommended_shot_count:
            project.ai_recommended_shot_count = None
        if existing and self.store.list_shots(project.id):
            content_changed = (
                existing.brief != project.brief
                or existing.style_bible != project.style_bible
                or existing.characters != project.characters
            )
            if content_changed and existing.status not in {ProjectStatus.BRIEF_DRAFT, ProjectStatus.STORYBOARD_DRAFT}:
                project.storyboard_version = max(project.storyboard_version, existing.storyboard_version + 1)
                project.status = ProjectStatus.STORYBOARD_DRAFT
                self._reset_shot_approvals(project.id)
        saved_project = self.store.save_project(project)
        if style_changed or characters_changed:
            self.refresh_shot_prompt_caches(project.id)
        return saved_project

    def refresh_shot_prompt_caches(
        self,
        project_id: str,
        shot_ids: list[str] | None = None,
    ) -> list[Shot]:
        """Recompile all prompt variants after a local asset/project change.

        Prompt text embeds reference manifests, so deleting or replacing an
        asset must advance the content revision even when the shot prose is
        unchanged. Paid/authored H3 text is durable user work, not a cache:
        retain it and its original source revision so the UI can flag review.
        """
        project = self.require_project(project_id)
        assets = self.store.list_assets(project_id)
        selected = set(shot_ids) if shot_ids is not None else None
        saved: list[Shot] = []
        for shot in self.store.list_shots(project_id):
            if selected is not None and shot.id not in selected:
                continue
            shot.voice_events = self._resolve_voice_events(
                project,
                self._effective_voice_events(shot),
                shot.duration_seconds,
                shot.duration_seconds,
            )
            self.synchronize_shot_cast(project, shot)
            shot.content_revision += 1
            character_ids = self.keyframe_character_ids(project, shot, shot.scene_description)
            scene_profile = self.scene_profile_for_shot(project, shot)
            shot.visual_prompt = self.compile_visual_prompt(
                project,
                shot.scene_description or shot.narrative,
                character_ids,
                shot.character_appearance_ids,
                scene_profile=scene_profile,
            )
            shot.keyframe_prompt = self.compile_keyframe_prompt(
                project,
                shot.scene_description or shot.narrative,
                character_ids,
                shot.character_appearance_ids,
                shot_size=shot.shot_size,
                camera_angle=shot.camera_angle,
                lens=shot.lens,
                text_policy=shot.text_policy,
                scene_profile=scene_profile,
            )
            shot.keyframe_prompt_source_revision = shot.content_revision
            if not shot.h3_prompt_skill_output.strip():
                shot.video_prompt_source = self.compile_base_video_prompt(project, shot.subject_motion, shot.narrative)
                shot.video_prompt = self.compile_h3_prompt(project, shot, assets)
                shot.h3_director_version = H3_DIRECTOR_VERSION
                shot.h3_prompt_source_revision = shot.content_revision
            shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
            shot.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
            shot.seedance_prompt_source_revision = shot.content_revision
            shot.approval_status = ApprovalStatus.DRAFT
            shot.version += 1
            saved.append(self.store.save_shot(shot))
        return saved

    def apply_scene_profiles_to_shots(
        self,
        project_id: str,
        scene_profile_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Enable scene profiles for all mapped shots and refresh local prompts.

        This is intentionally local and non-generative. Existing keyframe
        images and paid/authored H3 outputs remain durable; their older source
        revision makes the existing UI review warning visible.
        """
        project = self.require_project(project_id)
        requested = set(scene_profile_ids or [])
        profiles = [
            profile for profile in project.scene_profiles
            if not requested or profile.id in requested
        ]
        if requested - {profile.id for profile in profiles}:
            raise ValueError("所选场景档案已不存在，请刷新页面后重试")
        if not profiles:
            raise ValueError("当前没有可应用的场景档案")

        profiles_by_shot: dict[str, list[SceneProfile]] = {}
        for profile in profiles:
            for shot_id in profile.source_shot_ids:
                profiles_by_shot.setdefault(shot_id, []).append(profile)

        shots = self.store.list_shots(project_id)
        known_ids = {shot.id for shot in shots}
        matched_ids = [shot_id for shot_id in profiles_by_shot if shot_id in known_ids]
        if not matched_ids:
            raise ValueError("这些场景档案没有匹配当前分镜，请重新识别场景")
        if self._active_keyframes.intersection(matched_ids):
            raise KeyframeBusyError("相关镜头正在生成首帧，请完成后再批量应用场景档案")

        h3_preserved = 0
        keyframes_preserved = 0
        ordinals: list[int] = []
        for shot in shots:
            shot_profiles = profiles_by_shot.get(shot.id, [])[:2]
            if not shot_profiles:
                continue
            shot.scene_profile_ids = [profile.id for profile in shot_profiles]
            shot.scene_profile_id = shot.scene_profile_ids[0]
            shot.use_scene_profile = True
            h3_preserved += int(bool(shot.h3_prompt_skill_output.strip()))
            keyframes_preserved += int(bool(shot.keyframe_asset_id or shot.image_path))
            ordinals.append(shot.ordinal)
            self.store.save_shot(shot)

        project.scene_consistency_mode = SceneConsistencyMode.OPTIONAL
        self.store.save_project(project)
        saved = self.refresh_shot_prompt_caches(project_id, matched_ids)
        return {
            "applied_count": len(saved),
            "shot_ids": [shot.id for shot in saved],
            "shot_ordinals": sorted(ordinals),
            "scene_count": len(profiles),
            "h3_preserved_count": h3_preserved,
            "keyframes_preserved_count": keyframes_preserved,
        }

    def invalidate_storyboard(self, project_id: str) -> Project:
        project = self.require_project(project_id)
        if project.status not in {ProjectStatus.BRIEF_DRAFT, ProjectStatus.STORYBOARD_DRAFT}:
            project.storyboard_version += 1
        project.status = ProjectStatus.STORYBOARD_DRAFT
        self._reset_shot_approvals(project_id)
        return self.store.save_project(project)

    def _reset_shot_approvals(self, project_id: str) -> None:
        for shot in self.store.list_shots(project_id):
            if shot.approval_status != ApprovalStatus.DRAFT:
                shot.approval_status = ApprovalStatus.DRAFT
                self.store.save_shot(shot)

    async def generate_storyboard(
        self,
        project_id: str,
        shot_count: int | None = None,
        count_mode: str = "manual",
        user_suggestions: str = "",
        prompt_targets: list[str] | None = None,
        h3_skill_id: str = "h3-prompt-writing",
    ) -> list[Shot]:
        project = self.require_project(project_id)
        user_suggestions = user_suggestions.strip()
        minimum_count = max(1, math.ceil(project.brief.target_duration_seconds / 15.0))
        if count_mode == "ai":
            count = await self.recommend_shot_count(project_id, user_suggestions) if user_suggestions else (
                project.ai_recommended_shot_count or await self.recommend_shot_count(project_id)
            )
        else:
            # Manual means exact. Do not silently replace the user's requested
            # count with an AI recommendation or a duration-derived value.
            count = max(1, min(500, int(shot_count or minimum_count)))
        if count_mode == "ai":
            count = max(minimum_count, min(500, int(count)))
        assets = self.store.list_assets(project_id)
        reference = next(
            (str(resolve_media_path(asset.path)) for asset in assets if asset.type == AssetType.IMAGE and asset.role in {AssetRole.CHARACTER, AssetRole.STYLE}),
            None,
        )
        orchestrator = WorkflowOrchestrator()
        average_duration = project.brief.target_duration_seconds / count
        count_constraint = f"""【镜头数量最高优先级硬约束】
本次必须把完整剧情从开端到结局重新规划为严格 {count} 个分镜；scenes 数组长度必须等于 {count}，id 必须连续为 1 到 {count}。
不允许先按 4–8 秒拆出更多镜头，不允许返回候选镜头，不允许在一个数组元素中嵌套子镜头，也不允许省略结尾。
镜头数量要求高于默认拆镜建议。本项目目标时长为 {project.brief.target_duration_seconds:g} 秒，平均每镜约 {average_duration:.2f} 秒；请在单镜不超过 15 秒的范围内调整每镜信息密度和 duration。
每镜必须按自身 duration 编写从 0 秒连续覆盖到结尾的 visual_beats；平均 {average_duration:.2f} 秒的镜头应每 2–4 秒出现一次新的可见变化，形成触发、执行、环境反馈、后果与结果的完整动作弧线，禁止用静止等待填满时长。
第 1 镜必须承担故事开端，第 {count} 镜必须完整呈现原剧情结局；中间 {max(0, count - 2)} 镜覆盖关键因果、转折与必要对白，确保压缩后仍是完整故事。
输出 JSON 前先自行计数校验，只有 scenes 恰好 {count} 项才能提交。"""
        base_suggestions = "\n\n".join(item for item in [user_suggestions, count_constraint] if item)
        # One click performs one paid generation. Provider/SKD retries and
        # post-validation re-generations are deliberately disabled: an invalid
        # result remains recoverable and visible instead of charging 2–3 times.
        storyboard = await orchestrator.llm.generate_storyboard(
            topic=project.brief.story,
            count=count,
            reference_image=reference,
            include_dialogue=True,
            character_description=self.character_bible(project),
            image_style=self.project_style_text(project),
            user_suggestions=base_suggestions,
        )
        actual_count = len(storyboard.scenes)
        unmatched_character_names = list(
            dict.fromkeys(
                name
                for scene in storyboard.scenes
                for name in scene.character_names
                if not self.resolve_character_ids(project, explicit_names=[name])
            )
        )
        if actual_count != count:
            raw_hint = _raw_response_hint(orchestrator.llm)
            raise ValueError(
                f"AI 本次返回 {actual_count} 镜，不符合严格 {count} 镜要求。"
                "为避免重复扣费，系统没有自动再次调用；原有分镜保持不变"
                f"{raw_hint}。可调整建议后手动重试。"
            )
        storyboard_warnings = [
            f"AI 使用了项目中尚未建立档案的角色：{', '.join(unmatched_character_names)}。"
            "这些角色已保留在画面描述中，但未自动绑定参考图；可在‘需求与角色’中补建档案。"
        ] if unmatched_character_names else []
        if storyboard_warnings:
            logger.warning("项目 %s 分镜角色提示：%s", project_id, storyboard_warnings[0])
        durations = self._allocate_durations(
            [float(scene.duration) for scene in storyboard.scenes],
            project.brief.target_duration_seconds,
        )
        shots: list[Shot] = []
        sizes = ["全景", "中景", "近景", "特写", "中景"]
        for ordinal, (scene, duration) in enumerate(zip(storyboard.scenes, durations, strict=False), start=1):
            scene_event = (scene.event or scene.story_beat or scene.narrative or scene.opening_state or "").strip()
            scene_visual_source = (scene.opening_state or scene.visual_prompt or scene_event).strip()
            scene_opening = (scene.opening_state or scene.keyframe_prompt or scene.visual_prompt or scene_event).strip()
            scene_visual_description = "；".join(
                item for item in [
                    scene_visual_source,
                    f"本镜事件：{scene_event}" if scene_event and scene_event != scene_visual_source else "",
                ] if item
            )
            scene_motion = (scene.motion_prompt or "；".join(
                beat.subject_action for beat in scene.visual_beats if beat.subject_action
            ) or scene_event).strip()
            visual_beats = self._scale_visual_beats(
                scene.visual_beats,
                float(scene.duration),
                duration,
                shot_size=scene.shot_size or sizes[(ordinal - 1) % len(sizes)],
                camera_angle=scene.camera_angle or "平视",
                camera_motion=scene.camera_motion or "固定镜头",
                fallback_action=scene_motion,
            )
            voice_events = self._resolve_voice_events(
                project,
                scene.voice_events,
                float(scene.duration),
                duration,
            )
            speaker_id = self._resolve_dialogue_speaker_id(
                project,
                scene.dialogue_speaker,
                scene_event,
                scene.narrative,
                scene.dialogue,
            )
            character_voice_speaker = next(
                (event.speaker_id for event in voice_events if event.kind == "character" and event.speaker_id),
                None,
            )
            if character_voice_speaker:
                speaker_id = character_voice_speaker
            scene_character_ids = self.resolve_character_ids(
                project,
                scene_event,
                scene.narrative,
                scene_visual_source,
                scene.visual_prompt,
                scene_motion,
                explicit_names=scene.character_names,
                speaker_ids=[speaker_id],
            )
            normalized_motion = self._normalize_motion_duration(scene_motion, duration)
            base_video_prompt = self.compile_base_video_prompt(
                project,
                normalized_motion,
                scene_event or scene.narrative,
            )
            shot = Shot(
                project_id=project_id,
                ordinal=ordinal,
                title=f"镜头 {ordinal} · {(scene_event or scene_opening)[:18]}",
                narrative=scene_event or scene_opening,
                dialogue=scene.dialogue,
                dialogue_speaker_id=speaker_id,
                duration_seconds=round(duration, 3),
                scene_description=scene_opening,
                character_ids=scene_character_ids,
                reference_asset_ids=list(
                    dict.fromkeys(
                        asset_id
                        for character in project.characters
                        if character.id in scene_character_ids
                        for asset_id in character.reference_asset_ids
                    )
                ),
                shot_size=scene.shot_size or sizes[(ordinal - 1) % len(sizes)],
                camera_angle=scene.camera_angle or "平视",
                lens=scene.lens or ("35mm 广角" if ordinal == 1 else "50mm 标准镜头"),
                camera_motion=scene.camera_motion or self._camera_motion_from_prompt(normalized_motion),
                subject_motion=normalized_motion,
                transition=scene.transition or "硬切",
                audio_design=scene.audio_design,
                visual_beats=visual_beats,
                voice_events=voice_events,
                text_policy=scene.text_policy,
                visual_prompt=self.compile_visual_prompt(project, scene_visual_description, scene_character_ids),
                keyframe_prompt=self.compile_keyframe_prompt(
                    project,
                    scene_opening,
                    scene_character_ids,
                    shot_size=scene.shot_size or sizes[(ordinal - 1) % len(sizes)],
                    camera_angle=scene.camera_angle or "平视",
                    lens=scene.lens or ("35mm 广角" if ordinal == 1 else "50mm 标准镜头"),
                    text_policy=scene.text_policy,
                ),
                video_prompt_source=base_video_prompt,
                video_prompt=base_video_prompt,
                negative_prompt=project.brief.negative_prompt,
                h3_turbo=False,
                h3_steps=25,
                render_frames=h3_frames_for_seconds(duration),
            )
            shots.append(shot)
        requested_targets = ["seedance"] if prompt_targets is None else prompt_targets
        normalized_targets = [
            target for target in dict.fromkeys(requested_targets) if target in {"h3", "seedance"}
        ]
        # The LLM call above can take a while. Merge the storyboard fields onto
        # the newest project instead of saving the stale pre-call snapshot and
        # accidentally erasing scene/character assets created in parallel.
        latest_project = self.require_project(project_id)
        if prompt_targets is not None:
            latest_project.preferred_prompt_targets = normalized_targets
        latest_project.storyboard_count_mode = "ai" if count_mode == "ai" else "manual"
        latest_project.manual_shot_count = count if count_mode == "manual" else None
        latest_project.storyboard_warnings = storyboard_warnings
        latest_project.status = ProjectStatus.STORYBOARD_DRAFT
        latest_project.storyboard_version += 1
        self.store.save_project(latest_project)
        saved = self.store.replace_shots(project_id, shots)
        if "seedance" in normalized_targets:
            saved = self.generate_seedance_prompts(project_id, [shot.id for shot in saved])
        if "h3" in normalized_targets:
            saved = await self.generate_h3_prompts(
                project_id,
                [shot.id for shot in saved],
                h3_skill_id,
                user_suggestions,
            )
        return saved

    async def recommend_shot_count(self, project_id: str, user_suggestions: str = "") -> int:
        project = self.require_project(project_id)
        minimum = max(1, math.ceil(project.brief.target_duration_seconds / 15.0))
        fallback = max(minimum, min(500, math.ceil(project.brief.target_duration_seconds / 10.0)))
        provider = settings.SHOT_COUNT_PROVIDER
        if provider == "auto":
            provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        orchestrator = WorkflowOrchestrator(provider)
        prompt = f"""分析下面的剧情脚本，为总时长 {project.brief.target_duration_seconds:g} 秒的视频决定分镜数量。
每个生成片段不得超过 15 秒，通常按 6–12 秒规划；较长镜头必须每 2–4 秒有新的可见变化，以多个连续 visual beats 形成完整动作弧线。多人轮流发言或切换地点时拆镜；同一地点内连续的触发、执行、反馈和结果可保留在一个镜头。
合理范围是 {minimum} 到 500 镜。只返回 JSON：
{{"shot_count": 整数, "reason": "一句话理由"}}

剧情脚本：
{project.brief.story}
{f'''\n用户本次分镜建议（需要一并考虑）：\n{user_suggestions.strip()}''' if user_suggestions.strip() else ''}
"""
        try:
            payload = await orchestrator.llm.generate_json(
                "你是专业影视分镜导演，负责在可生成性、叙事完整度和成本之间做取舍。",
                prompt,
            )
            count = int(payload.get("shot_count", fallback))
        except Exception:
            count = fallback
        count = max(minimum, min(500, count))
        latest_project = self.require_project(project_id)
        latest_project.ai_recommended_shot_count = count
        self.store.save_project(latest_project)
        return count

    async def import_storyboard(
        self,
        project_id: str,
        rows: list[dict[str, str]],
        user_suggestions: str = "",
        prompt_targets: list[str] | None = None,
        h3_skill_id: str = "h3-prompt-writing",
    ) -> list[Shot]:
        """Import a customer storyboard and let the configured LLM fill blank production fields.

        Cells supplied by the customer always win over the AI response.  The
        row count is also immutable so an imported ten-row table stays ten
        shots throughout the workflow.
        """
        project = self.require_project(project_id)
        source_rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in rows[:500]
            if any(str(value or "").strip() for value in row.values())
        ]
        if not source_rows:
            raise ValueError("分镜表中没有可导入的数据行")
        # Reuse the same provider resolution as the normal storyboard flow.
        # In particular, runtime settings may expose ``auto`` while the
        # orchestrator resolves it to the configured production provider.
        llm = WorkflowOrchestrator().llm
        fields = (
            "title,narrative,dialogue,dialogue_speaker,scene_description,character_names,"
            "shot_size,camera_angle,lens,camera_motion,subject_motion,transition,audio_design,"
            "duration_seconds,generation_mode,visual_prompt,keyframe_prompt,h3_prompt,seedance_prompt"
        )
        ai_fields = (
            "title,narrative,dialogue,dialogue_speaker,scene_description,character_names,"
            "shot_size,camera_angle,lens,camera_motion,subject_motion,transition,audio_design,"
            "duration_seconds,generation_mode"
        )
        prompt = f"""根据项目剧本、角色设定和客户分镜表，补全每行缺失的制作字段。
必须严格返回 {len(source_rows)} 行，顺序不变，不合并、不拆分、不新增分镜；客户已填写的值原样保留。
每镜时长不超过 15 秒，全部时长适配目标成片 {project.brief.target_duration_seconds:g} 秒。
AI 只需返回这些字段：{ai_fields}。visual_prompt、keyframe_prompt、h3_prompt、seedance_prompt 由系统本地编译；客户表中若已有这些列，系统会原样保留。
character_names 使用逗号分隔项目中已有角色名；generation_mode 只能是 auto、i2v、r2v。
严格返回 JSON：{{"rows": [{{上述 AI 字段}}]}}。

项目剧本：
{project.brief.story}

角色设定：
{self.character_bible(project)}

统一风格：
{self.project_style_text(project)}

用户导入建议：
{user_suggestions.strip() or '无'}

客户分镜表：
{json.dumps(source_rows, ensure_ascii=False)}
"""
        try:
            payload = await llm.generate_json(
                "你是影视分镜统筹。你只补空白字段，绝不改变客户表格行数、顺序和已填写内容。",
                prompt,
            )
            ai_rows = payload.get("rows") if isinstance(payload, dict) else None
            if not isinstance(ai_rows, list):
                ai_rows = []
        except Exception:
            logger.exception("AI 补全导入分镜表失败，保留原始表格并使用规则补全")
            ai_rows = []

        merged_rows: list[dict[str, str]] = []
        for index, source in enumerate(source_rows):
            ai_row = ai_rows[index] if index < len(ai_rows) and isinstance(ai_rows[index], dict) else {}
            merged = {key: str(ai_row.get(key) or "").strip() for key in fields.split(",")}
            for key, value in source.items():
                if value:
                    merged[key] = value
            merged_rows.append(merged)

        weights: list[float] = []
        for row in merged_rows:
            try:
                weights.append(max(0.25, min(15.0, float(row.get("duration_seconds") or 5))))
            except (TypeError, ValueError):
                weights.append(5.0)
        durations = self._allocate_durations(weights, project.brief.target_duration_seconds)
        assets = self.store.list_assets(project_id)
        shots: list[Shot] = []
        for ordinal, (row, duration) in enumerate(zip(merged_rows, durations, strict=False), start=1):
            character_names = [
                item.strip()
                for item in re.split(r"[,，、;/；]", row.get("character_names", ""))
                if item.strip()
            ]
            speaker_id = self._resolve_dialogue_speaker_id(
                project,
                row.get("dialogue_speaker", ""),
                row.get("narrative", ""),
                row.get("dialogue", ""),
            )
            character_ids = self.resolve_character_ids(
                project,
                *(row.get(key, "") for key in (
                    "title",
                    "narrative",
                    "dialogue",
                    "scene_description",
                    "visual_prompt",
                    "keyframe_prompt",
                    "subject_motion",
                )),
                explicit_names=character_names,
                speaker_ids=[speaker_id],
            )
            scene = row.get("scene_description") or row.get("narrative") or f"镜头 {ordinal}"
            subject_motion = row.get("subject_motion") or "角色保持连贯、自然的小幅动作"
            try:
                generation_mode = GenerationMode(row.get("generation_mode") or "auto")
            except ValueError:
                generation_mode = GenerationMode.AUTO
            visual_beats = self._scale_visual_beats(
                [],
                duration,
                duration,
                shot_size=row.get("shot_size") or "中景",
                camera_angle=row.get("camera_angle") or "平视",
                camera_motion=row.get("camera_motion") or self._camera_motion_from_prompt(subject_motion),
                fallback_action=subject_motion,
            )
            voice_events: list[VoiceEvent] = []
            if row.get("dialogue", "").strip():
                voice_kind = self._voice_kind_from_label(row.get("dialogue_speaker", ""))
                voice_events = [
                    VoiceEvent(
                        kind=voice_kind,
                        speaker_id=speaker_id if voice_kind == "character" else None,
                        speaker_name=row.get("dialogue_speaker", ""),
                        text=row.get("dialogue", ""),
                        start_seconds=0.35,
                        end_seconds=min(duration, max(1.5, len(row.get("dialogue", "")) / 4.2)),
                        lip_sync=voice_kind == "character" and bool(speaker_id),
                    )
                ]
            shot = Shot(
                project_id=project_id,
                ordinal=ordinal,
                title=row.get("title") or f"镜头 {ordinal}",
                narrative=row.get("narrative") or scene,
                dialogue=row.get("dialogue", ""),
                dialogue_speaker_id=speaker_id,
                duration_seconds=round(duration, 3),
                scene_description=scene,
                character_ids=character_ids,
                reference_asset_ids=list(dict.fromkeys(
                    asset_id
                    for character in project.characters
                    if character.id in character_ids
                    for asset_id in character.reference_asset_ids
                )),
                shot_size=row.get("shot_size") or "中景",
                camera_angle=row.get("camera_angle") or "平视",
                lens=row.get("lens") or "50mm 标准镜头",
                camera_motion=row.get("camera_motion") or self._camera_motion_from_prompt(subject_motion),
                subject_motion=subject_motion,
                transition=row.get("transition") or "硬切",
                audio_design=row.get("audio_design", ""),
                visual_beats=visual_beats,
                voice_events=voice_events,
                generation_mode=generation_mode,
                visual_prompt=row.get("visual_prompt") or self.compile_visual_prompt(project, scene, character_ids),
                keyframe_prompt=row.get("keyframe_prompt") or self.compile_keyframe_prompt(
                    project,
                    scene,
                    character_ids,
                    shot_size=row.get("shot_size") or "中景",
                    camera_angle=row.get("camera_angle") or "平视",
                    lens=row.get("lens") or "50mm 标准镜头",
                ),
                video_prompt_source=self.compile_base_video_prompt(project, subject_motion, row.get("narrative") or scene),
                video_prompt=row.get("h3_prompt") or self.compile_base_video_prompt(project, subject_motion, row.get("narrative") or scene),
                seedance_prompt=row.get("seedance_prompt", ""),
                negative_prompt=project.brief.negative_prompt,
                h3_turbo=False,
                h3_steps=25,
                render_frames=h3_frames_for_seconds(duration),
            )
            if not shot.seedance_prompt:
                shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
            shots.append(shot)

        requested_targets = ["seedance"] if prompt_targets is None else prompt_targets
        normalized_targets = [
            target for target in dict.fromkeys(requested_targets) if target in {"h3", "seedance"}
        ]
        if prompt_targets is not None:
            project.preferred_prompt_targets = normalized_targets
        project.storyboard_count_mode = "manual"
        project.manual_shot_count = len(shots)
        project.status = ProjectStatus.STORYBOARD_DRAFT
        project.storyboard_version += 1
        self.store.save_project(project)
        saved = self.store.replace_shots(project_id, shots)
        if "seedance" in normalized_targets:
            saved = self.generate_seedance_prompts(project_id, [shot.id for shot in saved])
        if "h3" in normalized_targets:
            saved = await self.generate_h3_prompts(
                project_id,
                [shot.id for shot in saved],
                h3_skill_id,
                user_suggestions,
            )
        return saved

    async def analyze_brief(self, project_id: str) -> ProjectAnalysisDraft:
        project = self.require_project(project_id)
        assets = {asset.id: asset for asset in self.store.list_assets(project_id)}
        reference_paths: list[str] = []
        reference_labels: list[str] = []
        character_payload = []
        for character in project.characters:
            bound = []
            for asset_id in character.reference_asset_ids:
                asset = assets.get(asset_id)
                if not asset or asset.type != AssetType.IMAGE:
                    continue
                path = resolve_media_path(asset.path)
                if path.exists() and len(reference_paths) < 8:
                    reference_paths.append(str(path))
                    label = f"图片{len(reference_paths)}={character.name}/{asset.name}"
                    reference_labels.append(label)
                    bound.append(label)
            character_payload.append(
                {
                    "character_id": character.id,
                    "name": character.name,
                    "description": character.description,
                    "wardrobe": character.wardrobe,
                    "voice_description": character.voice_description,
                    "bound_references": bound,
                }
            )
        minimum = max(1, math.ceil(project.brief.target_duration_seconds / 15.0))
        prompt = f"""请分析客户剧情脚本，并生成可供用户审阅后回填的项目需求草稿。

总时长：{project.brief.target_duration_seconds:g} 秒
画幅：{project.brief.aspect_ratio}
当前视觉风格：{project.brief.visual_style or '未填写'}
当前角色（保留 character_id；参考图片按附图顺序对应）：
{json.dumps(character_payload, ensure_ascii=False, indent=2)}
参考图顺序：{'; '.join(reference_labels) or '无'}

剧情脚本：
{project.brief.story}

严格返回以下 JSON 对象，不要 Markdown：
{{
  "visual_style": "适合该故事、可执行的画风描述",
  "pacing": "起承转合、情绪曲线和剪辑节奏",
  "audience": "目标观众",
  "style_bible": "统一色彩、材质、光影、构图、镜头语言及禁止漂移项",
  "negative_prompt": "通用负面提示词",
  "delivery_notes": "成片、字幕、声音、版权素材等交付注意事项",
  "recommended_shot_count": {minimum},
  "shot_count_reason": "推荐数量理由",
  "characters": [{{
    "character_id": "已有角色必须沿用其 id；新角色填 null",
    "name": "角色名",
    "description": "结合剧本与参考图的固定外貌圣经",
    "wardrobe": "固定服装、配饰、颜色和可接受变化",
    "voice_description": "声音、口音、语气与说话节奏",
    "reference_observations": "从绑定参考图观察到的关键特征；无图时注明按剧本推断"
  }}],
  "analysis_notes": ["需要用户确认的推断或冲突"]
}}

recommended_shot_count 必须不少于 {minimum}，确保任一镜头不超过 15 秒；6–15 秒镜头要规划每 2–4 秒一次可见变化的动作弧线，不能靠静止等待填时长。角色描述必须具备跨镜头复用的一致性，不要改变参考图中的身份特征。
characters 必须覆盖剧情里每一个会被镜头清晰拍到、说话、执行动作或与主要角色互动的人物、动物、鬼魂、拟人角色及其他需要跨镜保持外观的可辨认主体。即使原文只写“短发男生”“一名女生”“搭档”“老黄狗”等未具名角色，也要为其建立独立、稳定、可复用的角色名和形象档案。动物的 description 必须固定品种、年龄感、体型、毛长、毛色分区、斑纹位置、耳尾形态、眼睛和项圈等识别特征，wardrobe 填写项圈、牵引物或“无服装”。纯远景且不可辨认的群众不建角色档案。
凡是涉及“是否需要单独设计形象”的主体，一律先按“需要”处理并写入 characters，不得只放进 analysis_notes。每条 analysis_notes 都必须先把推荐默认方案落实到 characters、style_bible、delivery_notes 或其他对应字段中，再用“已默认……；如需可调整……”说明，禁止只提出问题而不给默认结果。
"""
        provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        analysis_llm = create_llm_generator(provider)
        if reference_paths:
            analysis_llm = create_llm_generator(resolve_reference_llm_provider(provider))
        payload = await analysis_llm.generate_json(
            "你是资深影视策划、分镜导演和角色一致性设计师。把模糊客户需求整理成可直接用于 AI 视频生产的结构化参数。",
            prompt,
            reference_images=reference_paths,
        )
        payload = _normalize_project_analysis_payload(
            payload,
            math.ceil(project.brief.target_duration_seconds / 6.5),
        )
        payload["recommended_shot_count"] = max(
            minimum,
            min(500, int(payload.get("recommended_shot_count") or math.ceil(project.brief.target_duration_seconds / 6.5))),
        )
        try:
            draft = ProjectAnalysisDraft.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                f"AI 剧本分析结果校验失败: {exc}{_raw_response_hint(analysis_llm)}"
            ) from exc
        existing_ids = {character.id for character in project.characters}
        existing_by_name = {character.name: character.id for character in project.characters}
        for character in draft.characters:
            if character.character_id not in existing_ids:
                character.character_id = existing_by_name.get(character.name)
        return draft

    def _extract_style_frames(self, project_id: str, asset: Asset) -> list[str]:
        source = resolve_media_path(asset.path)
        if not source.is_file():
            return []
        if not shutil.which(settings.FFMPEG_BIN) or not shutil.which(settings.FFPROBE_BIN):
            raise RuntimeError("分析参考视频需要 ffmpeg 与 ffprobe")
        probe = subprocess.run(
            [
                settings.FFPROBE_BIN,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            duration = max(0.1, float(probe.stdout.strip()))
        except ValueError:
            duration = 3.0
        output_dir = self.project_dir(project_id) / "style_frames" / asset.id
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[str] = []
        for index, ratio in enumerate((0.05, 0.25, 0.5, 0.75, 0.95), start=1):
            target = output_dir / f"frame-{index}.jpg"
            result = subprocess.run(
                [
                    settings.FFMPEG_BIN,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{max(0.0, duration * ratio):.3f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale='min(1280,iw)':-2",
                    str(target),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and target.is_file() and target.stat().st_size:
                frames.append(str(target))
        return frames

    async def analyze_style(
        self,
        project_id: str,
        asset_ids: list[str] | None = None,
        *,
        apply: bool = False,
    ) -> StyleAnalysisDraft:
        project = self.require_project(project_id)
        assets = self.store.list_assets(project_id)
        selected = set(asset_ids or [])
        candidates = [
            asset
            for asset in assets
            if asset.type in {AssetType.IMAGE, AssetType.VIDEO}
            and (asset.id in selected if selected else asset.role in {AssetRole.STYLE, AssetRole.MOTION})
        ]
        if not candidates:
            raise ValueError("请先上传或选择至少一张风格截图或一段参考视频")
        references: list[str] = []
        for asset in candidates:
            if asset.type == AssetType.IMAGE:
                path = resolve_media_path(asset.path)
                if path.is_file():
                    references.append(str(path))
            elif len(references) < 8:
                references.extend(await asyncio.to_thread(self._extract_style_frames, project_id, asset))
            if len(references) >= 8:
                break
        if not references:
            raise ValueError("所选素材中没有可分析的画面")
        llm = create_llm_generator(resolve_reference_llm_provider())
        prompt = f"""分析这些客户参考画面，识别可复用的 AI 视频视觉风格。项目剧情仅用于判断适配性，不要凭剧情改写画面观察。

先明确参考画面属于真人摄影、3D/CGI，还是二维手绘/动画/插画。若画面属于二维手绘、动画或插画：
- medium、texture 和 analysis_summary 必须明确写“二维手绘动画/插画”，准确描述勾线、五官简化程度、头身比例、平涂与厚涂关系；
- 不得使用含混的“写实融合”“写实人物”“写实皮肤”来描述角色；道具结构、磨损或生活细节较真实时，只能明确限定为“道具写实细节”；
- negative_constraints 必须明确加入：真人照片、摄影棚肖像、毛孔级皮肤、照片级写实人物、3D/PBR/CGI 人物。
逐张区分可复用的绘制媒介与当前画面的场景光线；不要把单张画面的地点、主光方向、冷暖色温或陈设写成所有场景都必须遵守的全局规则。

项目：{project.brief.title}
剧情摘要：{self._compact_prompt_text(project.brief.story, 1200)}

严格返回 JSON：
{{
  "name": "简短风格名",
  "medium": "媒介与渲染方式",
  "palette": "主色、辅色、饱和度和对比度",
  "lighting": "光源方向、软硬、明暗与时间氛围",
  "camera_language": "景别、焦段、机位、运镜规律",
  "composition": "构图、空间层次和主体比例",
  "texture": "人物、材质、线条、颗粒和表面质感",
  "motion_language": "角色动作节奏、镜头运动和剪辑倾向",
  "analysis_summary": "可直接注入人物图、场景图、首帧和视频 Prompt 的统一风格圣经",
  "negative_constraints": "必须避免的风格漂移、材质、文字、畸形和构图问题",
  "confidence": 0.0,
  "observations": ["从参考画面直接观察到的证据"]
}}"""
        payload = await llm.generate_json(
            "你是影视美术指导和 AI 视频风格分析师，只输出结构化视觉结论。",
            prompt,
            reference_images=references,
        )
        payload["reference_asset_ids"] = [asset.id for asset in candidates]
        payload["approved"] = False
        draft = StyleAnalysisDraft.model_validate(payload)
        if apply:
            self.apply_style_analysis(project_id, draft)
            draft.approved = True
        return draft

    @staticmethod
    def _style_analysis_visual_style(draft: StyleAnalysisDraft) -> str:
        fields = [
            draft.name,
            draft.medium,
            draft.palette,
            draft.lighting,
            draft.texture,
        ]
        return "；".join(value.strip() for value in fields if value and value.strip())

    @staticmethod
    def _style_analysis_bible(draft: StyleAnalysisDraft) -> str:
        sections = [
            ("风格定位", draft.name),
            ("媒介与渲染", draft.medium),
            ("色彩体系", draft.palette),
            ("光影规则", draft.lighting),
            ("镜头语言", draft.camera_language),
            ("构图规则", draft.composition),
            ("材质与表面", draft.texture),
            ("动作与剪辑", draft.motion_language),
            ("统一执行准则", draft.analysis_summary),
            ("禁止与避免", draft.negative_constraints),
        ]
        return "\n".join(
            f"【{label}】{value.strip()}"
            for label, value in sections
            if value and value.strip()
        )

    @staticmethod
    def _merge_prompt_constraints(current: str, generated: str) -> str:
        values: list[str] = []
        for value in (current, generated):
            normalized = value.strip().strip("；")
            if normalized and normalized not in values:
                values.append(normalized)
        return "；".join(values)

    def apply_style_analysis(self, project_id: str, draft: StyleAnalysisDraft) -> Project:
        """Apply an analysis to the latest project snapshot.

        Re-reading here keeps an in-flight analysis from overwriting unrelated
        character or scene updates that may have completed in parallel.
        """
        project = self.require_project(project_id)
        project.style_profile = StyleProfile.model_validate(
            {
                **draft.model_dump(exclude={"confidence", "observations"}),
                "approved": True,
            }
        )
        project.brief.visual_style = self._style_analysis_visual_style(draft)
        project.style_bible = self._style_analysis_bible(draft)
        project.brief.negative_prompt = self._merge_prompt_constraints(
            project.brief.negative_prompt,
            draft.negative_constraints,
        )
        return self.update_project(project)

    async def generate_scene_profiles(
        self,
        project_id: str,
        user_suggestions: str = "",
    ) -> list[SceneProfile]:
        project = self.require_project(project_id)
        shots = self.store.list_shots(project_id)
        if not shots:
            raise ValueError("请先生成分镜，再按客户一致性需求建立场景档案")
        payload_shots = [
            {
                "shot_id": shot.id,
                "ordinal": shot.ordinal,
                "title": shot.title,
                "scene": shot.scene_description,
                "narrative": shot.narrative,
            }
            for shot in shots
        ]
        llm = create_llm_generator(settings.LLM_PROVIDER)
        payload = await llm.generate_json(
            "你是影视场景连续性设计师，只合并确实发生在同一物理空间和同一时间氛围的镜头。",
            f"""根据分镜识别可选的场景一致性组。场景变化频繁或无需硬一致性的镜头可以不归入任何组。
用户建议：{user_suggestions.strip() or '无'}
项目风格：{self.project_style_text(project) or '未设定'}
分镜：{json.dumps(payload_shots, ensure_ascii=False)}

严格返回 JSON：
{{"scenes":[{{"name":"场景名","description":"稳定空间布局、固定陈设、材质与灯光","continuity_notes":"允许变化的机位与必须保持的环境要素","shot_ordinals":[1,2]}}]}}""",
        )
        # A scene profile is derived from exact shot IDs. If the storyboard was
        # regenerated while the LLM was analysing it, saving the old mapping
        # would bind profiles to obsolete shots. Keep the new storyboard and ask
        # the user to analyse its scenes again instead.
        current_shots = self.store.list_shots(project_id)
        if [shot.id for shot in current_shots] != [shot.id for shot in shots]:
            raise ValueError("场景识别期间分镜已更新，本次文字档案未保存；请基于新分镜重新识别")

        latest_project = self.require_project(project_id)
        profiles: list[SceneProfile] = []
        by_ordinal = {shot.ordinal: shot for shot in current_shots}
        existing_profiles = list(latest_project.scene_profiles)
        for item in payload.get("scenes", []):
            ordinals = [int(value) for value in item.get("shot_ordinals", []) if str(value).isdigit()]
            source_ids = [by_ordinal[value].id for value in ordinals if value in by_ordinal]
            if not source_ids:
                continue
            name = str(item.get("name") or f"场景 {len(profiles) + 1}")
            existing = next(
                (
                    profile
                    for profile in existing_profiles
                    if set(profile.source_shot_ids) == set(source_ids)
                    or profile.name.strip().casefold() == name.strip().casefold()
                ),
                None,
            )
            profiles.append(
                SceneProfile(
                    **(existing.model_dump(exclude={"id", "name", "description", "continuity_notes", "reference_asset_ids", "source_shot_ids", "approved", "version"}) if existing else {}),
                    id=existing.id if existing else new_id("scene"),
                    name=name,
                    description=str(item.get("description") or ""),
                    continuity_notes=str(item.get("continuity_notes") or ""),
                    reference_asset_ids=list(existing.reference_asset_ids) if existing else [],
                    source_shot_ids=source_ids,
                    approved=bool(existing and existing.approved),
                    version=existing.version + 1 if existing else 1,
                )
            )
        latest_project.scene_profiles = profiles
        latest_project.scene_consistency_mode = SceneConsistencyMode.OPTIONAL if profiles else SceneConsistencyMode.OFF
        self.store.save_project(latest_project)
        # Bind the suggested profile without enabling it. Users opt in per shot
        # or switch the project to strict mode explicitly.
        profiles_by_shot: dict[str, list[str]] = {}
        for profile in profiles:
            for shot_id in profile.source_shot_ids:
                profiles_by_shot.setdefault(shot_id, []).append(profile.id)
        for shot in current_shots:
            latest_shot = self.require_shot(shot.id)
            latest_shot.scene_profile_ids = profiles_by_shot.get(shot.id, [])[:2]
            latest_shot.scene_profile_id = latest_shot.scene_profile_ids[0] if latest_shot.scene_profile_ids else None
            latest_shot.use_scene_profile = False
            self.store.save_shot(latest_shot)
        return profiles

    def require_scene_profile(self, project_id: str, scene_profile_id: str) -> tuple[Project, SceneProfile]:
        project = self.require_project(project_id)
        profile = next((item for item in project.scene_profiles if item.id == scene_profile_id), None)
        if profile is None:
            raise KeyError(f"Scene profile not found: {scene_profile_id}")
        return project, profile

    def ensure_scene_reference_idle(self, scene_profile_id: str) -> None:
        if scene_profile_id in self._active_scene_references:
            raise SceneReferenceConflictError("这个场景的母版图正在处理，请等待完成；其他场景可并行操作")

    @staticmethod
    def compile_scene_reference_prompt(project: Project, profile: SceneProfile) -> str:
        # Reuse the rendering medium only. Legacy palette/lighting/composition
        # fields describe a specific reference scene, not a global environment.
        style = project.style_profile
        rendering = "；".join(value for value in (style.name, style.medium) if value) if style and style.approved else ""
        return "\n".join(part for part in [
            "【任务】影视场景设定母版，单张无人物环境图。展示清晰的主视角与空间布局，不做多格拼贴，不生成文字、标注、Logo 或水印。",
            f"【场景】{profile.name}：{profile.description}",
            f"【连续性规则】{profile.continuity_notes}" if profile.continuity_notes else "",
            f"【绘制方式】{rendering}" if rendering else "【绘制方式】沿用项目的绘制媒介与笔触；具体光影、色彩和陈设只按本场景描述。",
            f"【画幅】{project.brief.aspect_ratio}",
        ] if part)

    def scene_reference_prompt(self, project_id: str, scene_profile_id: str) -> dict:
        project, profile = self.require_scene_profile(project_id, scene_profile_id)
        return {"prompt": profile.reference_prompt or self.compile_scene_reference_prompt(project, profile),
                "default_prompt": self.compile_scene_reference_prompt(project, profile), "profile": profile}

    def update_scene_profile(self, project_id: str, scene_profile_id: str, changes: dict, expected_version: int) -> SceneProfile:
        self.ensure_scene_reference_idle(scene_profile_id)
        project, profile = self.require_scene_profile(project_id, scene_profile_id)
        if profile.version != expected_version:
            raise SceneReferenceConflictError("场景档案已更新，请重新打开后编辑；当前草稿仍保留在编辑框中")
        allowed = {"name", "description", "continuity_notes", "reference_prompt"}
        for key, value in changes.items():
            if key in allowed:
                value = str(value).strip()
                if key in {"name", "description"} and not value:
                    raise ValueError("场景名称和空间描述请填写完整")
                setattr(profile, key, value)
        profile.version += 1
        self.store.save_project(project)
        # No prompt-cache refresh: editing a scene must not erase paid H3 text.
        return profile

    def _scene_output_dir(self, project_id: str, profile: SceneProfile) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", profile.reference_run_id):
            raise ValueError("没有可恢复的场景生图任务")
        return self.project_dir(project_id) / "scene_assets" / profile.id / profile.reference_run_id

    def _set_scene_reference_state(self, project_id: str, scene_profile_id: str, **changes) -> SceneProfile:
        project, profile = self.require_scene_profile(project_id, scene_profile_id)
        for key, value in changes.items():
            setattr(profile, key, value)
        profile.version += 1
        self.store.save_project(project)
        return profile

    def reconcile_scene_reference_status(self, project: Project) -> Project:
        """Expose download stage and recover interrupted requests after restart."""
        changed = False
        for profile in project.scene_profiles:
            if profile.reference_status not in {"generating", "downloading"}:
                continue
            checkpoint = self._scene_output_dir(project.id, profile) / "grsai_result.json"
            if profile.id not in self._active_scene_references:
                profile.reference_status = "download_failed" if checkpoint.is_file() else "failed"
                profile.reference_error = "处理已中断；可恢复已保存的服务商结果，不会重复提交生图" if checkpoint.is_file() else "处理已中断，尚未记录服务商结果；请先核对服务商后台，避免重复付费"
                changed = True
            elif checkpoint.is_file() and profile.reference_status == "generating":
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                if data.get("result_url"):
                    profile.reference_status = "downloading"
                    changed = True
        if changed:
            self.store.save_project(project)
        return project

    def _finish_scene_reference(self, project_id: str, scene_profile_id: str, output: str) -> Asset:
        project, profile = self.require_scene_profile(project_id, scene_profile_id)
        path = Path(output).resolve()
        asset = next((item for item in self.store.list_assets(project_id) if item.role == AssetRole.SCENE and Path(item.path).resolve() == path), None)
        if asset is None:
            asset = self.register_existing_asset(project_id, path, role=AssetRole.SCENE,
                                                 name=f"{profile.name} · 场景母版", description=profile.reference_generation_prompt)
        profile.reference_asset_ids = list(dict.fromkeys([*profile.reference_asset_ids, asset.id]))
        profile.approved = True
        profile.reference_status = "completed"
        profile.reference_error = ""
        profile.version += 1
        self.store.save_project(project)
        return asset

    async def generate_scene_reference(
        self,
        project_id: str,
        scene_profile_id: str,
        user_suggestions: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
        *,
        prompt: str | None = None,
        expected_version: int | None = None,
    ) -> Asset:
        self.ensure_scene_reference_idle(scene_profile_id)
        project, profile = self.require_scene_profile(project_id, scene_profile_id)
        if expected_version is not None and expected_version != profile.version:
            raise SceneReferenceConflictError("场景档案或 Prompt 已更新，请重新打开核对后生成")
        effective_prompt = (prompt if prompt is not None else profile.reference_prompt or self.compile_scene_reference_prompt(project, profile)).strip()
        if not effective_prompt:
            raise ValueError("母版图 Prompt 请填写完整")
        if prompt is None and user_suggestions.strip():
            effective_prompt += f"\n【补充建议】{user_suggestions.strip()}"
        image_gen = create_image_generator(image_provider=image_provider, model=image_model)
        profile.reference_run_id = uuid4().hex
        output_dir = self._scene_output_dir(project_id, profile)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._active_scene_references.add(scene_profile_id)
        try:
            self._set_scene_reference_state(project_id, scene_profile_id, reference_run_id=profile.reference_run_id,
                reference_prompt=effective_prompt, reference_generation_prompt=effective_prompt,
                reference_status="generating", reference_error="")
            scene = Scene(id=1, duration=5, narrative=profile.description, visual_prompt=effective_prompt, motion_prompt="")
            output = await image_gen.generate_image(scene, str(output_dir), None,
                seed=self._stable_seed(project_id, profile.id), character_description="", image_style="",
                aspect_ratio=project.brief.aspect_ratio)
            return self._finish_scene_reference(project_id, scene_profile_id, output)
        except Exception as exc:
            self._set_scene_reference_state(project_id, scene_profile_id,
                reference_status="download_failed" if not isinstance(exc, ImageGenerationError) and (output_dir / "grsai_result.json").is_file() else "failed",
                reference_error=str(exc) or type(exc).__name__)
            raise
        finally:
            self._active_scene_references.discard(scene_profile_id)

    async def retry_scene_reference_download(self, project_id: str, scene_profile_id: str) -> Asset:
        self.ensure_scene_reference_idle(scene_profile_id)
        _, profile = self.require_scene_profile(project_id, scene_profile_id)
        if profile.reference_status != "download_failed":
            raise ValueError("此场景没有待恢复的下载结果")
        output_dir = self._scene_output_dir(project_id, profile)
        self._active_scene_references.add(scene_profile_id)
        try:
            self._set_scene_reference_state(project_id, scene_profile_id, reference_status="downloading", reference_error="")
            output = await GrsaiImageGenerator.resume_image(str(output_dir))
            return self._finish_scene_reference(project_id, scene_profile_id, output)
        except Exception as exc:
            self._set_scene_reference_state(project_id, scene_profile_id, reference_status="failed" if isinstance(exc, ImageGenerationError) else "download_failed", reference_error=str(exc) or type(exc).__name__)
            raise
        finally:
            self._active_scene_references.discard(scene_profile_id)

    async def generate_character_references(
        self,
        project_id: str,
        character_ids: list[str] | None = None,
        user_suggestions: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
        reference_asset_ids: list[str] | None = None,
        appearance_profile_id: str | None = None,
        context_reference_asset_ids: list[str] | None = None,
        reference_strategy: str = "identity",
    ) -> list[Asset]:
        if reference_strategy not in {"identity", "project_style"}:
            raise ValueError("未知的角色参考策略")
        project = self.require_project(project_id)
        selected = set(character_ids or [])
        characters = [
            character for character in project.characters if not selected or character.id in selected
        ]
        if not characters:
            raise ValueError("请先通过剧本分析建立角色，或至少选择一个角色")
        image_gen = create_image_generator(image_provider=image_provider, model=image_model)
        assets_by_id = {asset.id: asset for asset in self.store.list_assets(project_id)}
        created: list[Asset] = []
        for character in characters:
            appearance = next(
                (item for item in character.appearance_profiles if item.id == appearance_profile_id),
                None,
            )
            if appearance_profile_id and appearance is None:
                continue
            target_reference_ids = list(
                dict.fromkeys(appearance.reference_asset_ids if appearance else character.reference_asset_ids)
            )
            requested_bound_ids = (
                list(dict.fromkeys(reference_asset_ids or target_reference_ids))
                if reference_strategy == "identity"
                else []
            )
            bound_ids = [
                asset_id
                for asset_id in requested_bound_ids
                if asset_id in assets_by_id
                and assets_by_id[asset_id].type == AssetType.IMAGE
                and resolve_media_path(assets_by_id[asset_id].path).is_file()
            ]
            requested_context_ids = (
                [*(reference_asset_ids or []), *(context_reference_asset_ids or [])]
                if reference_strategy == "project_style"
                else list(context_reference_asset_ids or [])
            )
            target_owned_ids = set(target_reference_ids)
            other_character_ids = [
                asset_id
                for other in project.characters
                if other.id != character.id
                for asset_id in self._valid_character_reference_ids(other, assets_by_id)
            ]
            other_character_ids.sort(
                key=lambda asset_id: (
                    not any(tag.startswith("series:") for tag in assets_by_id[asset_id].tags),
                    not assets_by_id[asset_id].approved,
                )
            )
            style_ids = [
                *(project.style_profile.reference_asset_ids if project.style_profile and project.style_profile.approved else []),
                *[
                    asset.id
                    for asset in assets_by_id.values()
                    if asset.role == AssetRole.STYLE and asset.type == AssetType.IMAGE
                ],
            ]
            requested_character_context_ids = [
                asset_id
                for asset_id in requested_context_ids
                if asset_id in assets_by_id and assets_by_id[asset_id].role == AssetRole.CHARACTER
            ]
            requested_style_context_ids = [
                asset_id for asset_id in requested_context_ids
                if asset_id not in requested_character_context_ids
            ]
            context_candidates = (
                [
                    *requested_character_context_ids,
                    *other_character_ids,
                    *requested_style_context_ids,
                    *style_ids,
                ]
                if reference_strategy == "project_style"
                else requested_context_ids
            )
            context_ids = [
                asset_id
                for asset_id in dict.fromkeys(context_candidates)
                if asset_id not in bound_ids
                and asset_id not in target_owned_ids
                and asset_id in assets_by_id
                and not (
                    reference_strategy == "project_style"
                    and assets_by_id[asset_id].character_id == character.id
                )
                and assets_by_id[asset_id].type == AssetType.IMAGE
                and resolve_media_path(assets_by_id[asset_id].path).is_file()
            ][:10]
            reference_paths = [
                str(resolve_media_path(assets_by_id[asset_id].path))
                for asset_id in [*bound_ids, *context_ids]
            ][:10]
            appearance_label = appearance.label if appearance else "默认形象"
            appearance_time = appearance.time_context if appearance else ""
            appearance_description = appearance.description if appearance else character.description
            appearance_wardrobe = appearance.wardrobe if appearance else character.wardrobe
            if bound_ids:
                reference_instruction = "严格保留目标角色身份参考图中的脸部或头部结构、发型或毛色斑纹、体型及配饰，并按档案补齐缺失角度与细节。"
            elif any(assets_by_id[asset_id].role == AssetRole.CHARACTER for asset_id in context_ids):
                reference_instruction = (
                    "输入参考图中的已确认角色图是本项目/同系列最高优先级画风锚点。严格继承其二维或三维媒介、"
                    "线条方式、五官简化程度、头身比例、上色方法和材质颗粒，使目标角色看起来属于同一部作品；"
                    "只继承画风，禁止复制参考图人物身份或动物角色身份，也不复制其脸、发型、服装、年龄或性别。目标角色必须根据剧本档案"
                    "设计独立且可区分的面部或物种特征、体型、毛发与服装配饰。"
                )
            elif context_ids:
                reference_instruction = (
                    "输入参考图是项目画风参考，严格继承其绘制媒介、线条、角色造型比例、上色方式、材质颗粒和年代语言；"
                    "目标角色必须根据剧本档案设计独立且可区分的身份，禁止复制参考图人物身份或动物角色身份。"
                )
            else:
                reference_instruction = "当前没有目标角色身份参考图，根据剧本角色档案设计唯一、稳定、可跨镜复用的角色身份；若为动物，必须固定品种、毛色斑纹、耳尾形态、体型与项圈等识别特征。"
            suggestion = user_suggestions.strip()
            prompt = (
                f"【角色四视图设定板版式】\n{CHARACTER_REFERENCE_SHEET_LAYOUT}\n"
                f"【参考图使用规则】\n{reference_instruction}"
                f"角色：{character.name}。形象档案：{appearance_label}。剧情时期/状态：{appearance_time}。"
                f"外貌：{appearance_description}。服装：{appearance_wardrobe}。"
                f"项目剧情依据：{self._compact_prompt_text(project.brief.story, 1800)}。"
                f"项目角色渲染风格：{self.character_rendering_style_text(project)}。"
                f"【用户生成建议】\n{suggestion or '无额外建议，严格按角色档案生成。'}\n"
                "用户建议用于细化角色外貌、服装、气质、材质和表现细节；最终成图仍须严格保持上述四视图数量、顺序和横向排版。"
            )
            output_dir = self.project_dir(project_id) / "character_assets" / character.id / uuid4().hex
            output_dir.mkdir(parents=True, exist_ok=True)
            scene = Scene(id=1, duration=5, narrative=character.name, visual_prompt=prompt, motion_prompt="")
            output = await image_gen.generate_image(
                scene,
                str(output_dir),
                ",".join(reference_paths) or None,
                seed=self._stable_seed(project_id, character.id, appearance_profile_id or "base"),
                character_description="，".join(filter(None, [appearance_description, appearance_wardrobe])),
                image_style=self.character_rendering_style_text(project),
                aspect_ratio="16:9",
            )
            asset = self.register_existing_asset(
                project_id,
                Path(output),
                role=AssetRole.CHARACTER,
                name=f"{character.name} · {appearance_label} · AI 四视图角色设定板",
                description=prompt,
            )
            asset.character_id = character.id
            asset.approved = True
            if reference_strategy == "project_style":
                asset.tags = list(dict.fromkeys([*asset.tags, "generation:project-style-match"]))
            if appearance:
                asset.tags = list(dict.fromkeys([*asset.tags, f"appearance:{appearance.id}"]))
            self.store.save_asset(asset)

            # Image generation is asynchronous and may overlap with storyboard,
            # scene, or another AI task.  Attach the result to the newest project
            # snapshot so a long-running character render never overwrites those
            # parallel updates with the stale project loaded above.
            latest_project = self.require_project(project_id)
            latest_character = next(
                (item for item in latest_project.characters if item.id == character.id),
                None,
            )
            if latest_character is None:
                raise KeyError(f"Character not found after generation: {character.id}")
            if appearance:
                latest_appearance = next(
                    (item for item in latest_character.appearance_profiles if item.id == appearance.id),
                    None,
                )
                if latest_appearance is None:
                    raise KeyError(f"Appearance profile not found after generation: {appearance.id}")
                latest_appearance.reference_asset_ids = (
                    [asset.id]
                    if reference_strategy == "project_style"
                    else list(dict.fromkeys([*latest_appearance.reference_asset_ids, *bound_ids, asset.id]))
                )
                latest_appearance.approved = True
            else:
                latest_character.reference_asset_ids = (
                    [asset.id]
                    if reference_strategy == "project_style"
                    else list(dict.fromkeys([*latest_character.reference_asset_ids, *bound_ids, asset.id]))
                )
            self.store.save_project(latest_project)
            created.append(asset)
        return created

    async def _merge_script_character_roster(self, project_id: str) -> Project:
        """Refresh the cast from the script while preserving all confirmed fields and references."""
        analysis = await self.analyze_brief(project_id)
        project = self.require_project(project_id)
        by_id = {character.id: character for character in project.characters}
        by_name = {character.name.strip(): character for character in project.characters if character.name.strip()}
        for draft in analysis.characters:
            name = draft.name.strip()
            if not name:
                continue
            current = by_id.get(draft.character_id or "") or by_name.get(name)
            if current:
                if not current.description.strip():
                    current.description = draft.description
                if not current.wardrobe.strip():
                    current.wardrobe = draft.wardrobe
                if not current.voice_description.strip():
                    current.voice_description = draft.voice_description
                continue
            character = CharacterProfile(
                name=name,
                description=draft.description,
                wardrobe=draft.wardrobe,
                voice_description=draft.voice_description,
            )
            project.characters.append(character)
            by_id[character.id] = character
            by_name[name] = character
        if not project.characters:
            raise ValueError("AI 未从剧本中识别出可生成的人物，请先在需求与角色中补充主要角色")
        return self.store.save_project(project)

    @staticmethod
    def _valid_character_reference_ids(
        character: CharacterProfile,
        assets_by_id: dict[str, Asset],
    ) -> list[str]:
        candidates = [
            *character.reference_asset_ids,
            *[
                asset.id
                for asset in assets_by_id.values()
                if asset.character_id == character.id and asset.role == AssetRole.CHARACTER
            ],
        ]
        return [
            asset_id
            for asset_id in dict.fromkeys(candidates)
            if asset_id in assets_by_id
            and assets_by_id[asset_id].type == AssetType.IMAGE
            and resolve_media_path(assets_by_id[asset_id].path).is_file()
        ]

    async def generate_missing_character_references(
        self,
        project_id: str,
        *,
        mode: str = "script_style",
        user_suggestions: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
        reference_asset_ids: list[str] | None = None,
    ) -> dict[str, object]:
        if mode not in {"script_style", "complete_missing"}:
            raise ValueError("未知的人物补齐模式")
        current = self.require_project(project_id)
        current_assets = self.store.list_assets(project_id)
        has_style_reference = bool(
            self.project_style_text(current).strip()
            or any(
                asset.role == AssetRole.STYLE and asset.type == AssetType.IMAGE
                for asset in current_assets
            )
        )
        if not has_style_reference:
            raise ValueError("请先填写视觉风格，或上传并分析至少一张画风参考图")
        project = await self._merge_script_character_roster(project_id)
        assets_by_id = {asset.id: asset for asset in self.store.list_assets(project_id)}
        references_by_character = {
            character.id: self._valid_character_reference_ids(character, assets_by_id)
            for character in project.characters
        }
        missing = [
            character
            for character in project.characters
            if not references_by_character[character.id]
        ]
        if not missing:
            return {
                "assets": [],
                "generated_character_ids": [],
                "character_count": len(project.characters),
                "missing_count": 0,
                "message": "剧本角色均已有可用人物参考图",
            }

        requested_context_ids = [
            asset_id
            for asset_id in dict.fromkeys(reference_asset_ids or [])
            if asset_id in assets_by_id and assets_by_id[asset_id].type == AssetType.IMAGE
        ]
        style_context_ids = list(
            dict.fromkeys(
                [
                    *(project.style_profile.reference_asset_ids if project.style_profile and project.style_profile.approved else []),
                    *[
                        asset.id
                        for asset in assets_by_id.values()
                        if asset.role == AssetRole.STYLE and asset.type == AssetType.IMAGE
                    ],
                ]
            )
        )
        character_context_ids = []
        if mode == "complete_missing":
            character_context_ids = [
                asset_id
                for character in project.characters
                if character.id not in {item.id for item in missing}
                for asset_id in references_by_character[character.id]
            ]
        # Confirmed character sheets are the strongest signal for how a new
        # cast member should be drawn. Keep them ahead of generic style stills
        # so provider reference limits never discard the character-style anchor.
        context_ids = list(
            dict.fromkeys([*requested_context_ids, *character_context_ids, *style_context_ids])
        )[:10]
        created = await self.generate_character_references(
            project_id=project_id,
            character_ids=[character.id for character in missing],
            user_suggestions=user_suggestions,
            image_provider=image_provider,
            image_model=image_model,
            context_reference_asset_ids=context_ids,
            reference_strategy="project_style" if mode == "complete_missing" else "identity",
        )
        return {
            "assets": created,
            "generated_character_ids": [asset.character_id for asset in created if asset.character_id],
            "character_count": len(project.characters),
            "missing_count": len(missing),
            "message": f"已为 {len(created)} 个缺失角色生成参考图",
        }

    async def analyze_character_references(
        self,
        project_id: str,
        character_id: str,
        reference_asset_ids: list[str],
        appearance_profile_id: str | None = None,
        appearance_label: str = "",
        user_suggestions: str = "",
    ) -> CharacterProfile:
        """Use selected uploaded images to fill a base or period-specific look."""
        project = self.require_project(project_id)
        character = next((item for item in project.characters if item.id == character_id), None)
        if character is None:
            raise KeyError(f"Character not found: {character_id}")
        assets = {asset.id: asset for asset in self.store.list_assets(project_id)}
        selected_assets = [
            assets[asset_id]
            for asset_id in dict.fromkeys(reference_asset_ids)
            if asset_id in assets and assets[asset_id].type == AssetType.IMAGE
        ]
        reference_paths = [
            str(resolve_media_path(asset.path))
            for asset in selected_assets
            if resolve_media_path(asset.path).is_file()
        ][:10]
        if not reference_paths:
            raise ValueError("请先在右侧素材库勾选至少一张人物参考图")

        appearance = next(
            (item for item in character.appearance_profiles if item.id == appearance_profile_id),
            None,
        )
        if appearance_profile_id and appearance is None:
            raise KeyError(f"Appearance profile not found: {appearance_profile_id}")
        target_label = appearance.label if appearance else (appearance_label.strip() or "基础角色形象")
        payload = await create_llm_generator(resolve_reference_llm_provider()).generate_json(
            "你是影视角色造型师和人物一致性设计师。只根据参考图、剧本与用户说明整理同一人物的稳定身份特征。",
            f"""分析附图中的人物，为角色“{character.name}”回填形象档案“{target_label}”。

剧情：{self._compact_prompt_text(project.brief.story, 1800)}
当前基础设定：{character.description}
当前服装：{character.wardrobe}
用户说明：{user_suggestions.strip() or '无'}

严格返回 JSON：
{{
  "name": "角色名；无充分依据时保持 {character.name}",
  "description": "年龄感、脸型、五官、肤色、发型、体型、可识别身份特征；不要把镜头角度当身体特征",
  "wardrobe": "服装版型、材质、颜色、配饰和稳定限制",
  "voice_description": "结合剧本推断的声音、语气与节奏；不确定时写建议",
  "time_context": "该形象对应的年龄、年代、剧情阶段或胖瘦/健康状态",
  "reference_observations": "参考图中直接观察到的依据"
}}""",
            reference_images=reference_paths,
        )
        ids = [asset.id for asset in selected_assets]

        # The LLM call above can take tens of seconds. Re-read the project before
        # saving so parallel character backfills merge into the latest state
        # instead of letting the last completed request overwrite the others.
        latest_project = self.require_project(project_id)
        latest_character = next((item for item in latest_project.characters if item.id == character_id), None)
        if latest_character is None:
            raise KeyError(f"Character not found: {character_id}")
        latest_appearance = next(
            (item for item in latest_character.appearance_profiles if item.id == appearance_profile_id),
            None,
        )
        if appearance_profile_id and latest_appearance is None:
            raise KeyError(f"Appearance profile not found: {appearance_profile_id}")
        if latest_appearance is None and appearance_label.strip():
            latest_appearance = next(
                (item for item in latest_character.appearance_profiles if item.label == appearance_label.strip()),
                None,
            )
            if latest_appearance is None:
                latest_appearance = CharacterAppearanceProfile(label=appearance_label.strip())
                latest_character.appearance_profiles.append(latest_appearance)
        if latest_appearance:
            latest_appearance.description = str(payload.get("description") or latest_appearance.description)
            latest_appearance.wardrobe = str(payload.get("wardrobe") or latest_appearance.wardrobe)
            latest_appearance.time_context = str(payload.get("time_context") or latest_appearance.time_context)
            latest_appearance.reference_asset_ids = list(dict.fromkeys([*latest_appearance.reference_asset_ids, *ids]))
            latest_appearance.approved = True
        else:
            latest_character.name = str(payload.get("name") or latest_character.name)
            latest_character.description = str(payload.get("description") or latest_character.description)
            latest_character.wardrobe = str(payload.get("wardrobe") or latest_character.wardrobe)
            latest_character.voice_description = str(payload.get("voice_description") or latest_character.voice_description)
            latest_character.reference_asset_ids = list(dict.fromkeys([*latest_character.reference_asset_ids, *ids]))
        for asset in selected_assets:
            asset.character_id = latest_character.id
            if latest_appearance:
                asset.tags = list(dict.fromkeys([*asset.tags, f"appearance:{latest_appearance.id}"]))
            self.store.save_asset(asset)
        self.store.save_project(latest_project)
        return latest_character

    async def assess_story_duration(self, project_id: str) -> ScriptDurationAssessment:
        """Estimate a script's natural edited runtime without padding to its target."""
        project = self.require_project(project_id)
        story = project.brief.story.strip()
        if not story:
            raise ValueError("请先填写或上传剧情故事脚本")

        provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        duration_llm = create_llm_generator(provider)
        target = project.brief.target_duration_seconds
        prompt = f"""请独立评估下面剧本在正常、紧凑但不仓促的成片剪辑中，能够自然支撑多少秒。

【目标时长仅用于比较，不得反向拉伸评估】
- 客户目标成片：{target:g} 秒
- 画幅：{project.brief.aspect_ratio}
- 叙事节奏要求：{project.brief.pacing or '未填写，按常规剧情短片节奏'}

【自然时长计时口径｜必须严格执行】
1. 先区分真正说出口的对白/旁白与只在画面中发生的动作，不得把所有叙述文字都当作配音朗读。
2. 中文对白和旁白按正常可懂语速约每秒 3.4 个有效字符估算，正常标点呼吸已包含在估值中。
3. 只计算推动剧情、表达情绪转折或传递信息所必需的可见动作、建立镜头和转场；对白期间可同步完成的动作不得重复累计。
4. 普通反应停顿通常只取 0.3–1 秒。只有剧本明确需要悬念、喜剧卡点或重大情绪转折时才增加必要停顿。
5. 严禁为了接近 {target:g} 秒而加入静止等待、重复表情、无新信息的环境展示、机械留白或把一个简单动作拖成数秒。
6. natural_duration_seconds 必须是依据现有内容得到的独立结论，可以明显短于或长于目标。min/max 表示紧凑剪辑到舒展剪辑的合理范围，而不是凑目标的范围。
7. dialogue_and_narration_seconds、visual_only_seconds、transition_seconds 是互不重叠的时间分类，三项之和应接近 natural_duration_seconds。
8. density_issues 只写会造成内容偏空、重复或过密的具体位置；没有则返回空数组。

【待评估剧本】
{story}

严格返回以下 JSON 对象，不要 Markdown：
{{
  "natural_duration_seconds": 120,
  "natural_duration_min_seconds": 110,
  "natural_duration_max_seconds": 132,
  "dialogue_and_narration_seconds": 75,
  "visual_only_seconds": 40,
  "transition_seconds": 5,
  "content_density": "sparse、balanced 或 dense",
  "summary": "自然时长结论及与客户目标的差距",
  "assessment_basis": ["关键计时依据"],
  "density_issues": ["可能导致空镜或信息过密的具体段落"]
}}"""
        payload = await duration_llm.generate_json(
            "你是资深影视编剧、剪辑师和制片统筹。你只按现有有效内容评估自然成片时长，不为满足客户目标虚构停顿或留白。",
            prompt,
        )

        def seconds(name: str, default: float) -> float:
            try:
                return max(0.0, float(payload.get(name, default)))
            except (TypeError, ValueError):
                return default

        natural = max(1.0, seconds("natural_duration_seconds", target))
        lower = max(1.0, min(natural, seconds("natural_duration_min_seconds", natural * 0.92)))
        upper = max(natural, seconds("natural_duration_max_seconds", natural * 1.08))
        difference = natural - target
        tolerance = max(8.0, target * 0.05)
        recommendation = "expand" if difference < -tolerance else "shorten" if difference > tolerance else "fit"
        density = str(payload.get("content_density") or "balanced").lower()
        if density not in {"sparse", "balanced", "dense"}:
            density = "balanced"

        def string_list(name: str) -> list[str]:
            value = payload.get(name)
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        return ScriptDurationAssessment(
            natural_duration_seconds=round(natural, 1),
            natural_duration_min_seconds=round(lower, 1),
            natural_duration_max_seconds=round(upper, 1),
            target_duration_seconds=round(target, 1),
            difference_seconds=round(difference, 1),
            recommendation=recommendation,
            content_density=density,
            dialogue_and_narration_seconds=round(seconds("dialogue_and_narration_seconds", 0), 1),
            visual_only_seconds=round(seconds("visual_only_seconds", natural), 1),
            transition_seconds=round(seconds("transition_seconds", 0), 1),
            summary=str(payload.get("summary") or "").strip(),
            assessment_basis=string_list("assessment_basis"),
            density_issues=string_list("density_issues"),
        )

    async def rewrite_story(
        self,
        project_id: str,
        mode: str = "auto",
        user_suggestions: str = "",
    ) -> ScriptRewriteDraft:
        project = self.require_project(project_id)
        story = project.brief.story.strip()
        if not story:
            raise ValueError("请先填写或上传剧情故事脚本")
        user_suggestions = user_suggestions.strip()
        mode_instruction = {
            "expand": "只做扩写：补足必要的情节铺垫、动作、场景转换和可表演对白，使内容充分支撑目标时长。",
            "shorten": "只做缩写：压缩重复信息、次要枝节和过长对白，保留核心人物、因果、冲突、转折与结局。",
            "auto": "自动判断扩写或缩写：根据当前剧本的信息密度与目标时长，选择最适合落地制作的改写方向。",
        }.get(mode)
        if mode_instruction is None:
            raise ValueError(f"不支持的剧本改写模式: {mode}")

        provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        rewrite_llm = create_llm_generator(provider)
        target = project.brief.target_duration_seconds
        minimum_shots = max(1, math.ceil(target / 15.0))
        prompt = f"""请根据用户目标时长和改写建议，重新整理一份可直接进入 AI 视频分镜阶段的完整剧情脚本。

【项目约束】
- 目标成片时长：{target:g} 秒
- 画幅：{project.brief.aspect_ratio}
- 视觉风格：{project.brief.visual_style or '未填写'}
- 叙事节奏：{project.brief.pacing or '未填写'}
- 最少可拆分镜头数：{minimum_shots}（任一生成片段不得超过 15 秒；长镜头每 2–4 秒安排一次新的可见变化，形成触发、执行、反馈和结果的动作弧线）
- 改写模式：{mode_instruction}

【用户本次改写建议｜高优先级】
{user_suggestions or '未填写，由你依据目标时长和原剧本自动处理'}

【原始剧情脚本】
{story}

【改写要求】
1. 改写后的剧本必须能在约 {target:g} 秒内通过 AI 视频实现，给动作、对白、停顿和转场留出合理时间。
2. 对白按普通中文语速约每秒 3–4 个汉字估算；不要让台词占满所有时间，要保留视觉叙事空间。
3. 缩写时优先合并重复问答、说明和弱情节；扩写时补充可视化动作、冲突递进、场景反应和必要过渡，不要凭空改变关键事实。
4. 保留角色名称、关键设定、核心因果和客户明确要求。输出的是完整改写剧本，不是摘要，也不是分镜表。
5. estimated_duration_seconds 应接近目标时长；若仍有制作风险，在 feasibility_notes 中明确说明。

严格返回以下 JSON 对象，不要 Markdown：
{{
  "rewritten_story": "完整改写后的剧情故事脚本",
  "rewrite_mode": "expand、shorten 或 balanced",
  "estimated_duration_seconds": {target:g},
  "change_summary": "本次扩写/缩写的主要调整",
  "feasibility_notes": ["仍需用户确认或存在的时长风险"]
}}
"""
        payload = await rewrite_llm.generate_json(
            "你是资深影视编剧和 AI 视频制片人。你的任务是在不破坏核心故事的前提下，让剧本内容量严格适配成片目标时长。",
            prompt,
        )
        rewrite_mode = str(payload.get("rewrite_mode") or "balanced").lower()
        if rewrite_mode not in {"expand", "shorten", "balanced"}:
            rewrite_mode = "balanced"
        payload["rewrite_mode"] = rewrite_mode
        payload["estimated_duration_seconds"] = max(
            1.0,
            float(payload.get("estimated_duration_seconds") or target),
        )
        return ScriptRewriteDraft.model_validate(payload)

    @staticmethod
    def _clean_legacy_motion(value: str) -> str:
        text = re.sub(
            r"^总时长\s*(?:约|大约)?\s*\d+(?:\.\d+)?\s*秒[。.]?",
            "",
            (value or "").strip(),
        ).strip()
        for pattern in (
            r"本镜头只完成这一项核心动作[；，,。]*",
            r"人物位移和肢体幅度克制[；，,。]*",
            r"镜头固定或仅做一次缓慢单向移动[；，,。]*",
            r"禁止临时增加转身、走位、开关物品或第二次运镜[；，,。]*",
            r"镜头保持固定或仅做极慢的单向移动[；，,。]*",
        ):
            text = re.sub(pattern, "", text).strip("；，,。 ")
        return text

    @staticmethod
    def _scale_visual_beats(
        beats: list[VisualBeat],
        source_duration: float,
        target_duration: float,
        *,
        shot_size: str,
        camera_angle: str,
        camera_motion: str,
        fallback_action: str,
    ) -> list[VisualBeat]:
        """Scale authored beat timing to the allocated shot without padding it."""
        target = max(0.25, min(15.0, float(target_duration)))
        source = max(0.25, float(source_duration or target))
        ordered = sorted(beats, key=lambda item: (item.start_seconds, item.end_seconds))
        if not ordered:
            count = max(2, min(5, math.ceil(target / 3.0)))
            cleaned_action = ProjectService._clean_legacy_motion(fallback_action)
            clauses = [
                item.strip(" ，。；")
                for item in re.split(r"[，,；;。]|随后|然后|接着|继而|最终|最后", cleaned_action)
                if item.strip(" ，。；")
            ]
            if len(clauses) > 1 and clauses[0].startswith(("画面仅", "画面中仅", "镜头中仅")):
                clauses[1] = f"{clauses[0]}，{clauses[1]}"
                clauses = clauses[1:]
            clauses = [
                re.sub(r"随(?:弹窗)?文字播报节奏", "随画外系统播报节奏", clause)
                for clause in clauses
            ]
            stages = [
                "从首帧状态立即响应触发事件",
                "执行推动剧情的明确主体动作",
                "道具或环境产生同步可见反馈",
                "动作后果进一步改变局面",
                "落在清晰结果态并建立下一镜衔接点",
            ]
            actions = clauses[:count]
            while len(actions) < count:
                actions.append(stages[len(actions)])
            base_camera = re.sub(r"极慢|缓慢", "平稳", camera_motion or "固定镜头")
            return [
                VisualBeat(
                    start_seconds=round(index * target / count, 2),
                    end_seconds=round((index + 1) * target / count, 2),
                    purpose="开场钩子" if index == 0 else "结果落点" if index == count - 1 else "推进事件",
                    subject_action=ProjectService._compact_prompt_text(actions[index], 180),
                    shot_size=shot_size,
                    camera_angle=camera_angle,
                    camera_motion=base_camera if index == 0 else "固定镜头",
                )
                for index in range(count)
            ]

        scaled: list[VisualBeat] = []
        previous_end = 0.0
        for index, beat in enumerate(ordered[:6]):
            end = target if index == len(ordered[:6]) - 1 else min(target, beat.end_seconds * target / source)
            end = max(previous_end + 0.05, end)
            end = min(target, end)
            if end <= previous_end:
                continue
            scaled.append(
                beat.model_copy(
                    update={
                        "start_seconds": round(previous_end, 2),
                        "end_seconds": round(end, 2),
                        "shot_size": beat.shot_size or shot_size,
                        "camera_angle": beat.camera_angle or camera_angle,
                        "camera_motion": beat.camera_motion or camera_motion,
                    }
                )
            )
            previous_end = end
        if scaled:
            scaled[-1] = scaled[-1].model_copy(update={"end_seconds": round(target, 2)})
        return scaled

    def _resolve_voice_events(
        self,
        project: Project,
        events: list[VoiceEvent],
        source_duration: float,
        target_duration: float,
    ) -> list[VoiceEvent]:
        source = max(0.25, float(source_duration or target_duration))
        target = max(0.25, min(15.0, float(target_duration)))
        resolved: list[VoiceEvent] = []
        for event in events:
            start = min(target, event.start_seconds * target / source)
            end = min(target, event.end_seconds * target / source)
            if end <= start:
                end = min(target, start + 0.5)
            speaker_id = event.speaker_id
            kind = event.kind
            if kind == "character":
                # The model sometimes returns a correct speaker_name paired
                # with the previous turn's speaker_id. The authored name is
                # human-readable and wins whenever it resolves unambiguously;
                # otherwise retain a valid supplied id as a fallback.
                named_speaker_id = (
                    self._resolve_dialogue_speaker_id(
                        project,
                        event.speaker_name,
                        event.text,
                    )
                    if event.speaker_name.strip()
                    else None
                )
                speaker_id = named_speaker_id or speaker_id or self._resolve_dialogue_speaker_id(
                    project,
                    "",
                    event.text,
                )
            else:
                speaker_id = None
            resolved.append(
                event.model_copy(
                    update={
                        "speaker_id": speaker_id,
                        "start_seconds": round(start, 2),
                        "end_seconds": round(end, 2),
                        "lip_sync": bool(event.lip_sync and kind == "character" and speaker_id),
                    }
                )
            )
        fitted = fit_voice_event_payloads(
            [event.model_dump(mode="json") for event in resolved],
            target,
        )
        return [VoiceEvent.model_validate(event) for event in fitted]

    @staticmethod
    def _allocate_durations(weights: list[float], target: float) -> list[float]:
        if not weights:
            return []
        safe = [max(0.25, value) for value in weights]
        total_weight = sum(safe)
        values = [min(15.0, max(0.25, target * value / total_weight)) for value in safe]
        residual = target - sum(values)
        for _ in range(10):
            if abs(residual) < 0.001:
                break
            candidates = [index for index, value in enumerate(values) if (residual > 0 and value < 15.0) or (residual < 0 and value > 0.25)]
            if not candidates:
                break
            delta = residual / len(candidates)
            for index in candidates:
                values[index] = min(15.0, max(0.25, values[index] + delta))
            residual = target - sum(values)
        return values

    @staticmethod
    def _camera_motion_from_prompt(prompt: str) -> str:
        for keyword, label in (
            ("环绕", "环绕运镜"),
            ("跟拍", "跟拍"),
            ("推进", "缓慢推镜"),
            ("推近", "缓慢推镜"),
            ("拉远", "拉镜"),
            ("摇镜", "摇镜"),
            ("横移", "横向移镜"),
        ):
            if keyword in prompt:
                return label
        return "固定镜头，轻微呼吸感"

    @staticmethod
    def _normalize_motion_duration(prompt: str, duration: float) -> str:
        """Keep LLM-authored total-duration text aligned with allocated shots."""
        if not prompt:
            return prompt
        duration_text = f"{duration:.2f}".rstrip("0").rstrip(".")
        replacement = f"总时长约{duration_text}秒"
        normalized = re.sub(
            r"总时长\s*(?:约|大约)?\s*\d+(?:\.\d+)?\s*秒",
            replacement,
            prompt,
            count=1,
        )
        return normalized

    @staticmethod
    def _character_aliases(name: str) -> list[str]:
        """Return stable aliases for exact roster-to-shot matching.

        Character cards often carry a qualifier in brackets (for example
        ``女试炼者A（胡浩监护人）``) while a storyboard uses the shorter visible
        label.  Treat those labels as aliases, but never use generic appearance
        words from the character description to guess an identity.
        """
        clean = re.sub(r"\s+", "", name or "").strip()
        if not clean:
            return []
        aliases: list[str] = [clean]
        base = re.split(r"[（(【[]", clean, maxsplit=1)[0].strip()
        if base:
            aliases.append(base)
        bracket_values = re.findall(r"[（(【[]([^）)】\]]+)[）)】\]]", clean)
        for bracket_value in bracket_values:
            for value in re.split(r"[，,、/；;]", bracket_value):
                value = re.sub(r"^(?:或|又名|别名)", "", value).strip()
                if value and not any(word in value for word in ("未具名", "需确认", "待确认")):
                    aliases.append(value)
        for value in list(aliases):
            without_title = re.sub(r"(?:同志|先生|女士|老师)$", "", value).strip()
            if without_title:
                aliases.append(without_title)
                if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", without_title):
                    aliases.append(without_title[0])
        return list(dict.fromkeys(alias for alias in aliases if alias))

    @staticmethod
    def _normalized_character_label(value: str) -> str:
        return re.sub(r"[\s的·•・，,、。；;：:（）()【】\[\]‘’“”\"']+", "", value or "").casefold()

    @classmethod
    def resolve_character_ids(
        cls,
        project: Project,
        *texts: str,
        explicit_names: list[str] | None = None,
        selected_ids: list[str] | None = None,
        speaker_ids: list[str | None] | None = None,
    ) -> list[str]:
        """Resolve every roster character explicitly visible in a shot.

        Explicit UI selections are retained.  Exact roster names, bracket-free
        labels and declared aliases are then added from the generated fields.
        This keeps the checkboxes, portraits and all downstream prompts on one
        cast list without fuzzy-matching generic traits such as “短发男生”.
        """
        project_ids = {character.id for character in project.characters}
        original_selected = list(dict.fromkeys(selected_ids or []))
        unresolved_selected = [
            character_id for character_id in original_selected if character_id not in project_ids
        ]
        resolved = {
            character_id
            for character_id in original_selected
            if character_id in project_ids
        }
        normalized_explicit = {
            cls._normalized_character_label(name)
            for name in (explicit_names or [])
            if cls._normalized_character_label(name)
        }
        normalized_text = cls._normalized_character_label("。".join(text for text in texts if text))
        for character in project.characters:
            aliases = cls._character_aliases(character.name)
            canonical_labels = {
                cls._normalized_character_label(character.name),
                cls._normalized_character_label(re.split(r"[（(【[]", character.name, maxsplit=1)[0]),
            }
            normalized_aliases = {
                cls._normalized_character_label(alias)
                for alias in aliases
                if len(cls._normalized_character_label(alias)) >= 2
                or cls._normalized_character_label(alias) in canonical_labels
            }
            if normalized_explicit.intersection(normalized_aliases) or any(
                alias in normalized_text for alias in normalized_aliases
            ):
                resolved.add(character.id)
        resolved.update(
            speaker_id
            for speaker_id in (speaker_ids or [])
            if speaker_id and speaker_id in project_ids
        )
        return [
            *(character.id for character in project.characters if character.id in resolved),
            *unresolved_selected,
        ]

    @classmethod
    def synchronize_shot_cast(cls, project: Project, shot: Shot) -> Shot:
        """Add prompt-mentioned characters and refresh inherited portraits."""
        shot.character_ids = cls.resolve_character_ids(
            project,
            shot.title,
            shot.narrative,
            shot.scene_description,
            shot.subject_motion,
            shot.visual_prompt,
            shot.keyframe_prompt,
            selected_ids=shot.character_ids,
            speaker_ids=[
                shot.dialogue_speaker_id,
                *(turn.speaker_id for turn in shot.dialogue_turns),
                *(event.speaker_id for event in shot.voice_events if event.kind == "character"),
            ],
        )
        allowed = set(shot.character_ids)
        shot.character_appearance_ids = {
            character_id: appearance_id
            for character_id, appearance_id in shot.character_appearance_ids.items()
            if character_id in allowed
        }
        known_character_refs = {
            asset_id
            for character in project.characters
            for asset_id in [
                *character.reference_asset_ids,
                *(
                    asset_id
                    for appearance in character.appearance_profiles
                    for asset_id in appearance.reference_asset_ids
                ),
            ]
        }
        allowed_character_refs: list[str] = []
        for character in project.characters:
            if character.id not in allowed:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            allowed_character_refs.extend(
                appearance.reference_asset_ids if appearance else character.reference_asset_ids
            )
        preserved_refs = [
            asset_id
            for asset_id in shot.reference_asset_ids
            if asset_id not in known_character_refs
        ]
        shot.reference_asset_ids = list(dict.fromkeys([*preserved_refs, *allowed_character_refs]))
        return shot

    def synchronize_storyboard_casts(self, project_id: str) -> list[Shot]:
        """Repair legacy storyboard cast omissions without removing user picks."""
        project = self.require_project(project_id)
        shots = self.store.list_shots(project_id)
        assets = self.store.list_assets(project_id)
        saved: list[Shot] = []
        for shot in shots:
            before = (tuple(shot.character_ids), tuple(shot.reference_asset_ids))
            self.synchronize_shot_cast(project, shot)
            after = (tuple(shot.character_ids), tuple(shot.reference_asset_ids))
            if before != after:
                shot.content_revision += 1
                scene_profile = self.scene_profile_for_shot(project, shot)
                shot.visual_prompt = self.compile_visual_prompt(
                    project,
                    shot.scene_description or shot.narrative,
                    shot.character_ids,
                    shot.character_appearance_ids,
                    scene_profile=scene_profile,
                )
                shot.keyframe_prompt = self.compile_keyframe_prompt(
                    project,
                    shot.scene_description or shot.narrative,
                    shot.character_ids,
                    shot.character_appearance_ids,
                    shot_size=shot.shot_size,
                    camera_angle=shot.camera_angle,
                    lens=shot.lens,
                    text_policy=shot.text_policy,
                    scene_profile=scene_profile,
                )
                shot.keyframe_prompt_source_revision = shot.content_revision
                if not shot.h3_prompt_skill_output.strip():
                    shot.video_prompt_source = self.compile_base_video_prompt(project, shot.subject_motion, shot.narrative)
                    shot.video_prompt = self.compile_h3_prompt(project, shot, assets)
                    shot.h3_prompt_source_revision = shot.content_revision
                shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
                shot.seedance_prompt_source_revision = shot.content_revision
                shot.approval_status = ApprovalStatus.DRAFT
                shot.version += 1
                saved.append(self.store.save_shot(shot))
            else:
                saved.append(shot)
        return saved

    def _resolve_dialogue_speaker_id(self, project: Project, explicit: str, *context: str) -> str | None:
        """Resolve one speaking character instead of leaving the video model to guess."""
        explicit_text = (explicit or "").strip()
        external_markers = (
            "旁白",
            "画外音",
            "系统",
            "播报",
            "广播",
            "通知音",
            "无对白",
            "无",
        )
        if explicit_text and any(marker in explicit_text for marker in external_markers):
            return None
        if explicit_text:
            explicit_matches: list[tuple[int, int, str]] = []
            for character in project.characters:
                for alias in self._character_aliases(character.name):
                    if not alias:
                        continue
                    if explicit_text == alias:
                        explicit_matches.append((2, len(alias), character.id))
                    elif alias in explicit_text:
                        explicit_matches.append((1, len(alias), character.id))
            if explicit_matches:
                # Prefer an exact/full-name match, then the longest alias. This
                # avoids resolving “女试炼者A（胡浩监护人）” as 胡浩 merely
                # because his shorter name appears inside the role note.
                return max(explicit_matches, key=lambda item: (item[0], item[1]))[2]

        text = "。".join(item for item in context if item)
        for character in project.characters:
            for alias in self._character_aliases(character.name):
                if re.search(rf"[【\[]?{re.escape(alias)}[】\]]?\s*[:：]", text):
                    return character.id

        speech_verbs = "说|说道|开口|介绍|自我介绍|要求|命令|询问|问道|提问|追问|回答|告知|说明|解释|提醒|喊|宣布|自述|讲述|检讨|表态|回应|表示|否认|安抚|谈话"
        matches: list[tuple[int, str]] = []
        for character in project.characters:
            for alias in self._character_aliases(character.name):
                match = re.search(rf"{re.escape(alias)}(?:同志|先生|女士|老师)?\s*(?:{speech_verbs})", text)
                if match:
                    matches.append((match.start(), character.id))
        if matches:
            return min(matches, key=lambda item: item[0])[1]
        return None

    @staticmethod
    def _voice_kind_from_label(label: str) -> str:
        normalized = re.sub(r"[\s（）()【】\[\]：:]", "", (label or "")).casefold()
        if "系统" in normalized or "播报" in normalized:
            return "system_vo"
        if "旁白" in normalized:
            return "narration"
        if "画外音" in normalized or "广播" in normalized:
            return "offscreen"
        return "character"

    @classmethod
    def _effective_voice_events(cls, shot: Shot) -> list[VoiceEvent]:
        if shot.voice_events:
            return shot.voice_events
        if not shot.dialogue.strip():
            return []
        voice_kind = cls._voice_kind_from_label(f"{shot.dialogue} {shot.audio_design}")
        if voice_kind == "character":
            return []
        return [
            VoiceEvent(
                kind=voice_kind,
                speaker_name={
                    "system_vo": "系统播报",
                    "narration": "旁白",
                    "offscreen": "画外音",
                }.get(voice_kind, "画外音"),
                text=cls.spoken_dialogue_text(shot.dialogue),
                start_seconds=min(0.35, max(0.0, shot.duration_seconds - 0.5)),
                end_seconds=min(
                    shot.duration_seconds,
                    max(0.85, len(cls.spoken_dialogue_text(shot.dialogue)) / 4.2),
                ),
                lip_sync=False,
            )
        ]

    def _speaker_cues(self, project: Project, text: str) -> list[str]:
        verbs = "说|说道|开口|介绍|自我介绍|要求|命令|询问|问道|提问|追问|回答|告知|说明|解释|提醒|喊|宣布|自述|讲述|检讨|表态|回应|表示|否认|安抚|谈话"
        matches: list[tuple[int, str]] = []
        for character in project.characters:
            for alias in self._character_aliases(character.name):
                for match in re.finditer(rf"{re.escape(alias)}(?:同志|先生|女士|老师)?\s*(?:{verbs})", text):
                    matches.append((match.start(), character.id))
        cues: list[str] = []
        for _, speaker_id in sorted(matches):
            if not cues or cues[-1] != speaker_id:
                cues.append(speaker_id)
        return cues

    def _dialogue_roles(self, project: Project, *sources: str) -> tuple[list[str], list[str], list[str]]:
        """Extract local speaker roles without letting a regex span whole paragraphs."""
        asker_verbs = "宣布|指出|安抚|询问|提问|追问|要求|命令|问道|指向|示意|说(?!完|明|出)"
        reply_verbs = "回答|回应|表示|否认|自述|讲述|检讨|表态|说明|点头|摇头"
        questioners: list[str] = []
        respondents: list[str] = []
        cues: list[str] = []
        for source in sources:
            for clause in re.split(r"[，,；;。]", source or ""):
                clause = clause.strip()
                if not clause:
                    continue
                events: list[tuple[int, str, str]] = []
                for character in project.characters:
                    for alias in self._character_aliases(character.name):
                        match = re.search(
                            rf"{re.escape(alias)}(?:同志|先生|女士|老师|绍辉|秉诚)?[^，,；;。]{{0,10}}?(?P<verb>{asker_verbs}|{reply_verbs})",
                            clause,
                        )
                        if match:
                            events.append((match.start(), character.id, match.group("verb")))
                            break
                # The first named character in a short action clause is the
                # grammatical subject (for example “C看向秦提问” means C asks).
                # Keeping later object names would incorrectly assign the verb.
                for _, speaker_id, verb in sorted(events)[:1]:
                    if not cues or cues[-1] != speaker_id:
                        cues.append(speaker_id)
                    target = respondents if re.fullmatch(reply_verbs, verb) else questioners
                    if not target or target[-1] != speaker_id:
                        target.append(speaker_id)
        return questioners, respondents, cues

    def resolve_dialogue_turns(self, project: Project, shot: Shot) -> list[dict[str, str | None]]:
        if shot.dialogue_turns:
            return [turn.model_dump() for turn in shot.dialogue_turns if turn.text.strip()]
        spoken = self.spoken_dialogue_text(shot.dialogue)
        if not spoken:
            return []
        sentences = [part.strip() for part in re.findall(r"[^。！？!?]+[。！？!?]?", spoken) if part.strip()]
        questioners, respondents, cues = self._dialogue_roles(project, shot.narrative, shot.subject_motion)
        distinct_cues = list(dict.fromkeys(cues))
        # A stored speaker id may have been inferred by an older version that
        # treated a complete Q&A paragraph as one line.  It remains authoritative
        # only when the authored action does not name several speakers.
        if shot.dialogue_speaker_id and len(distinct_cues) <= 1:
            return [{"speaker_id": shot.dialogue_speaker_id, "text": spoken}]
        questioners = questioners or cues[:1]
        respondents = list(dict.fromkeys(respondents))
        if not respondents and len(distinct_cues) > 1:
            respondents = [speaker_id for speaker_id in distinct_cues if speaker_id not in questioners]
        if not respondents:
            respondents = [character.id for character in project.characters if character.id in shot.character_ids and character.id not in questioners]

        turns: list[dict[str, str | None]] = []
        question_index = 0
        awaiting_response = False
        current_speaker: str | None = None
        for index, sentence in enumerate(sentences):
            is_question = sentence.endswith(("？", "?"))
            is_directive = bool(re.search(r"(?:请|说一下|说说|继续说|接着说|清楚了吗|是否|如何|怎么样)", sentence))
            if is_question or is_directive:
                speaker_id = questioners[question_index % len(questioners)] if questioners else (cues[0] if cues else None)
                question_index += 1
            elif awaiting_response or (
                current_speaker in questioners
                and respondents
                and len(sentence.rstrip("。！？!?")) <= 8
            ):
                speaker_id = respondents[0] if respondents else (cues[min(index, len(cues) - 1)] if cues else None)
            elif index == 0 and questioners and respondents:
                speaker_id = questioners[0]
            elif current_speaker in respondents:
                speaker_id = current_speaker
            elif current_speaker in questioners:
                speaker_id = current_speaker
            else:
                speaker_id = cues[min(index, len(cues) - 1)] if cues else (respondents[0] if respondents else None)
            turns.append({"speaker_id": speaker_id, "text": sentence})
            awaiting_response = is_question or is_directive
            if speaker_id in respondents:
                awaiting_response = False
            current_speaker = speaker_id
        merged: list[dict[str, str | None]] = []
        for turn in turns:
            if merged and merged[-1]["speaker_id"] == turn["speaker_id"]:
                merged[-1]["text"] = f"{merged[-1]['text']}{turn['text']}"
            else:
                merged.append(turn.copy())
        return merged

    @staticmethod
    def spoken_dialogue_text(value: str) -> str:
        """Remove speaker labels and trailing acting notes before TTS/H3 use."""
        text = re.sub(r"\s+", " ", (value or "").strip())
        text = re.sub(r"^[【\[][^】\]]{1,30}[】\]]\s*[:：]?\s*", "", text)
        # Only strip a genuine speaker label (short name plus optional title).
        # A broader prefix+colon rule would eat real dialogue fragments such as
        # “登入高危电力副本：” produced by long-line segmentation.
        text = re.sub(r"^\w{1,7}(?:同志|老师|先生|女士|队长|主任|医生|警官)?\s*[:：]\s*", "", text)
        text = text.strip("'\"“”‘’ ")
        text = re.sub(r"[（(][^（）()]{1,50}[）)]\s*$", "", text).strip()
        return text.strip("'\"“”‘’ ")

    @staticmethod
    def character_voice_anchor(character: CharacterProfile) -> str:
        """Return the approved voice identity shared by every video engine."""
        authored = ProjectService._compact_prompt_text(character.voice_description, 180)
        if authored:
            # Analysis drafts often phrase the result as “建议为…”. Once the
            # character bible is approved this is an instruction, not an
            # optional suggestion to the video model.
            return re.sub(r"^(?:建议|推荐)(?:采用|使用|选择|为)?[：:\s]*", "", authored)
        age_match = re.search(r"(\d{1,2})\s*岁", character.description)
        age = f"约{age_match.group(1)}岁" if age_match else "与人物外观年龄一致"
        if "男" in character.description:
            gender = "男性声线"
        elif "女" in character.description:
            gender = "女性声线"
        else:
            gender = "自然人声"
        return f"{age}{gender}，标准普通话，音色、年龄感、音高、口音与人物外观一致，吐字自然"

    @classmethod
    def _voice_anchor_for_event(
        cls,
        project: Project,
        event: VoiceEvent,
    ) -> str:
        speaker = next(
            (item for item in project.characters if item.id == event.speaker_id),
            None,
        )
        if speaker:
            return cls.character_voice_anchor(speaker)
        return {
            "system_vo": "冷静清晰的中性电子播报音，音高稳定、吐字精准、无真人口语感",
            "narration": "清晰自然的画外旁白，音色稳定、吐字从容、不过度表演",
            "offscreen": "与声音来源身份和场景距离一致的自然画外人声，音色全程稳定",
        }.get(event.kind, "清晰自然且前后一致的画外人声")

    @classmethod
    def h3_character_voice_direction_english(cls, character: CharacterProfile) -> str:
        """Compile safe H3 performance metadata without exposing Chinese prose as speech."""
        source = f"{character.description} {cls.character_voice_anchor(character)}"
        phrases: list[str] = []
        if "年轻" in source or re.search(r"(?:1[89]|2\d)\s*岁", source):
            age = "young adult"
        elif "中年" in source or re.search(r"(?:3\d|4\d|5[0-5])\s*岁", source):
            age = "mature adult"
        elif "老年" in source or re.search(r"(?:5[6-9]|[6-9]\d)\s*岁", source):
            age = "older adult"
        else:
            age = "age-matched"
        if "女" in source:
            gender = "female"
        elif "男" in source:
            gender = "male"
        else:
            gender = "gender-neutral"
        phrases.append(f"{age} {gender} voice")
        if any(token in source for token in ("清亮", "清脆", "明亮", "清透")):
            phrases.append("clear, bright timbre")
        if any(token in source for token in ("低沉", "低音", "浑厚")):
            phrases.append("low, full-bodied timbre")
        if any(token in source for token in ("活泼", "有活力", "高能量", "元气")):
            phrases.append("lively, energetic delivery")
        if any(token in source for token in ("情绪起伏大", "情绪丰富", "表现力")):
            phrases.append("wide, expressive emotional range")
        if any(token in source for token in ("蔫蔫", "拖腔", "慵懒")):
            phrases.append("idle moments may sound languid and slightly drawn out")
        if any(token in source for token in ("语速瞬间变快", "语速加快", "看到高绩效", "看到任务")):
            phrases.append("when excited by a task, snap into a quicker, brighter delivery while keeping every word clear")
        elif any(token in source for token in ("节奏偏快", "语速偏快", "说话快")):
            phrases.append("brisk pace while remaining fully intelligible")
        if any(token in source for token in ("发亮", "兴奋", "惊喜")):
            phrases.append("a sparkling, excited lift in pitch and energy")
        if any(token in source for token in ("卡顿", "语塞", "结巴")):
            phrases.append("brief flustered hesitations when challenged")
        if "诚恳" in source:
            phrases.append("earnest delivery")
        if "急切" in source:
            phrases.append("urgent emotional intent")
        if any(token in source for token in ("咋呼", "咋咋呼呼")):
            phrases.append("slightly excitable reactions")
        if any(token in source for token in ("邻家", "没有公职", "公职人员的架子", "无架子")):
            phrases.append("informal, approachable girl-next-door conversational tone, never official or bureaucratic")
        if any(token in source for token in ("毛躁", "不耐烦")):
            phrases.append("a restless, impatient edge")
        if any(token in source for token in ("沉稳", "稳重")):
            phrases.append("steady, composed delivery")
        if any(token in source for token in ("威严", "权威")):
            phrases.append("controlled authority")
        if any(token in source for token in ("普通话", "吐字", "清晰", "清楚")):
            phrases.append("natural, clearly articulated Mandarin")
        else:
            phrases.append("natural, intelligible Mandarin")
        return "; ".join(dict.fromkeys(phrases))

    @classmethod
    def h3_voice_direction_english(cls, project: Project, event: VoiceEvent) -> str:
        speaker = next(
            (item for item in project.characters if item.id == event.speaker_id),
            None,
        )
        if speaker:
            authored = cls.character_voice_anchor(speaker)
            spoken = cls.spoken_dialogue_text(event.text)
            if (
                any(token in authored for token in ("看到高绩效任务", "看到任务", "接到任务"))
                and any(token in spoken for token in ("高危", "大单", "任务"))
            ):
                return (
                    "young adult female voice; clear, bright timbre; lively, energetic delivery; for this excited "
                    "task reaction, snap immediately into a quicker, brighter delivery with a sparkling, excited lift in pitch "
                    "and energy; brisk and punchy but every word stays clear; informal, approachable girl-next-door "
                    "tone, never official or bureaucratic; natural Mandarin"
                )
            return cls.h3_character_voice_direction_english(speaker)
        return {
            "system_vo": (
                "calm, clear, gender-neutral electronic announcement voice; stable pitch; precise articulation; "
                "no human conversational mannerisms"
            ),
            "narration": "clear, natural off-screen narration; stable timbre; measured articulation; restrained performance",
            "offscreen": "natural off-screen voice matched to the source identity and scene distance; stable timbre",
        }.get(event.kind, "clear, natural off-screen voice with stable timbre")

    @classmethod
    def h3_silent_voice_contract(cls, project: Project, events: list[VoiceEvent]) -> str:
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for event in events:
            if not cls.spoken_dialogue_text(event.text):
                continue
            source_id = event.speaker_id or event.speaker_name or event.kind
            key = (event.kind, source_id)
            if key not in grouped:
                grouped[key] = {"event": event, "timings": []}
            timings = grouped[key]["timings"]
            assert isinstance(timings, list)
            timings.append(f"{event.start_seconds:.2f}-{event.end_seconds:.2f}s")
        if not grouped:
            return ""
        lines = [
            "speech_control (NON-SPOKEN PRODUCTION INSTRUCTION): Only the exact text inside existing <d>...</d> "
            "tags may become speech. Never vocalize prompt prose, speaker labels, names, timings, parentheses, "
            "metadata, acting directions, or voice descriptions. Do not add, paraphrase, or repeat spoken words."
        ]
        for index, item in enumerate(grouped.values(), start=1):
            event = item["event"]
            timings = item["timings"]
            assert isinstance(event, VoiceEvent)
            assert isinstance(timings, list)
            if event.kind == "character":
                source_role = "visible character dialogue"
                lip_rule = "lip-sync only during the tagged words" if event.lip_sync else "no visible lip-sync"
            elif event.kind == "system_vo":
                source_role = "off-screen system announcement"
                lip_rule = "no visible character lip movement"
            elif event.kind == "narration":
                source_role = "off-screen narration"
                lip_rule = "no visible character lip movement"
            else:
                source_role = "off-screen dialogue"
                lip_rule = "no visible character lip movement"
            lines.append(
                f"silent_voice_direction_{index} (never spoken; applies to {', '.join(timings)} {source_role}): "
                f"{cls.h3_voice_direction_english(project, event)}; {lip_rule}."
            )
        return "\n".join(lines)

    @staticmethod
    def strip_h3_spoken_voice_metadata(prompt: str) -> str:
        """Remove legacy voice-anchor prose that H3 may continue reading as dialogue."""
        cleaned = re.sub(
            r"\s*\([^()\n]{0,120}(?:voice\s+anchor|声音锚点)\s*:[^()\n]{1,1200}\)",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"(?im)^\s*approved_voice_identity\s*:.*(?:\n|$)", "", cleaned)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    @staticmethod
    def character_bible(
        project: Project,
        character_ids: list[str] | None = None,
        appearance_ids: dict[str, str] | None = None,
    ) -> str:
        selected = set(character_ids) if character_ids is not None else None
        parts = []
        for character in project.characters:
            if selected is not None and character.id not in selected:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == (appearance_ids or {}).get(character.id)
                ),
                None,
            )
            details = "，".join(
                item
                for item in [
                    appearance.time_context if appearance else "",
                    appearance.description if appearance else character.description,
                    appearance.wardrobe if appearance else character.wardrobe,
                    character.voice_description,
                ]
                if item
            )
            parts.append(f"{character.name}：{details}" if details else character.name)
        return "；".join(parts)

    @staticmethod
    def compile_visual_prompt(
        project: Project,
        description: str,
        character_ids: list[str] | None = None,
        appearance_ids: dict[str, str] | None = None,
        *,
        scene_profile: SceneProfile | None = None,
    ) -> str:
        style = (
            ProjectService.project_rendering_style_text(project)
            if scene_profile
            else ProjectService.project_style_text(project)
        )
        character_text = ProjectService.character_bible(project, character_ids, appearance_ids)
        scene_text = (
            f"场景“{scene_profile.name}”为本镜唯一环境准则：{scene_profile.description}；"
            f"{scene_profile.continuity_notes}；忽略其他冲突的全局环境、光线和色彩描述"
            if scene_profile else ""
        )
        return "，".join(item.strip("，。 ") for item in [style, scene_text, character_text, description] if item)

    @staticmethod
    def character_visual_bible(
        project: Project,
        character_ids: list[str] | None = None,
        appearance_ids: dict[str, str] | None = None,
        detail_limit: int | None = None,
    ) -> str:
        """Return only appearance/wardrobe details suitable for image models."""
        selected = set(character_ids) if character_ids is not None else None
        parts: list[str] = []
        for character in project.characters:
            if selected is not None and character.id not in selected:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == (appearance_ids or {}).get(character.id)
                ),
                None,
            )
            details = "，".join(
                item
                for item in [
                    appearance.time_context if appearance else "",
                    appearance.description if appearance else character.description,
                    appearance.wardrobe if appearance else character.wardrobe,
                ]
                if item
            )
            if detail_limit is not None:
                details = ProjectService._compact_prompt_text(details, detail_limit)
            parts.append(f"{character.name}：{details}" if details else character.name)
        return "；".join(parts)

    @staticmethod
    def compile_keyframe_prompt(
        project: Project,
        description: str,
        character_ids: list[str] | None = None,
        appearance_ids: dict[str, str] | None = None,
        *,
        shot_size: str = "",
        camera_angle: str = "",
        lens: str = "",
        text_policy: str = "post_overlay",
        scene_profile: SceneProfile | None = None,
    ) -> str:
        """Compile the one final prompt submitted to the still-image provider."""
        source = (description or "").strip()
        # Older saved prompts used a prose wrapper. Extract its actual opening
        # image so recompilation does not recursively duplicate style/cast data.
        if source.startswith("静态分镜首帧"):
            match = re.search(r"画面内容：(.*?)(?:。构图与机位：|。本镜出场角色：|。统一视觉风格：|$)", source)
            if match:
                source = match.group(1).strip()
        text_markers = (
            "文字",
            "字样",
            "书法字",
            "字幕",
            "标题",
            "台词字卡",
            "显示文案",
            "写着",
            "标注",
            "倒计时",
        )
        if text_policy != "reference_locked" and any(marker in source for marker in text_markers):
            clauses = re.split(r"([，,；;。])", source)
            sanitized: list[str] = []
            for clause in clauses:
                if any(marker in clause for marker in text_markers):
                    carrier = next(
                        (item for item in ("UI 界面", "屏幕", "面板", "标牌", "路牌", "卡片", "画面中央") if item in clause),
                        "相应构图区域",
                    )
                    clause = f"{carrier}预留干净无字信息区，供后期叠加准确文字"
                sanitized.append(clause)
            source = "".join(sanitized).strip("，,；;。 ")
        style_source = (
            ProjectService.project_rendering_style_text(project)
            if scene_profile
            else ProjectService.project_style_text(project)
        )
        style = ProjectService._compact_prompt_text(style_source, 260)
        # Compact each subject independently. A single detailed protagonist
        # must not consume the whole budget and erase later people/pets from
        # the provider prompt.
        character_text = ProjectService._compact_prompt_text(
            ProjectService.character_visual_bible(
                project,
                character_ids,
                appearance_ids,
                detail_limit=260,
            ),
            1600,
        )
        composition = "，".join(item.strip("，。 ") for item in [shot_size, camera_angle, lens] if item)
        text_rule = (
            "仅保留用户已经锁定并在参考图中清晰存在的文字，逐字准确，不新增其他文案"
            if text_policy == "reference_locked"
            else "画面内不生成任何可读文字、汉字、字母、数字、字幕、标题、UI 文案、Logo、水印或乱码，文字统一后期叠加"
        )
        parts = [
            f"【首帧画面】{ProjectService._compact_prompt_text(source, 520)}" if source else "",
            (
                f"【场景档案｜本镜最高优先级】{scene_profile.name}：{scene_profile.description}；"
                f"{scene_profile.continuity_notes or '保持空间布局、固定陈设、材质与本场景光线'}；"
                "本镜环境、光线、色彩和陈设只按此档案执行，忽略全局风格分析或旧分镜中的冲突描述"
                if scene_profile else ""
            ),
            f"【构图】{composition}，画幅 {project.brief.aspect_ratio}" if composition else f"【构图】画幅 {project.brief.aspect_ratio}",
            f"【角色锚点】{character_text}" if character_text else "",
            f"【风格锚点】{style}" if style else "",
            (
                "【静帧约束】这是视频 0.00 秒的单张静态画面，只呈现动作发生前的起始姿态与空间关系；"
                "保持人物身份、脸型、发型、服装、配饰、人数、道具、场景结构和光线一致；"
                "不表现未来动作、动作过程、运镜、转场、时长、对白、配音或音效；"
                f"{text_rule}"
            ),
        ]
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def keyframe_character_ids(project: Project, shot: Shot, prompt: str = "") -> list[str]:
        """Return only characters that should condition this still image.

        Older storyboards assigned every project character whenever the model
        did not mention a name.  That made title cards and empty shots submit
        every portrait and the entire character bible.  Explicit name/speaker
        matches take priority; a deliberately selected subset is preserved.
        """
        combined_text = " ".join(
            item
            for item in [prompt, shot.scene_description, shot.narrative, shot.dialogue]
            if item
        )
        resolved_from_text = ProjectService.resolve_character_ids(
            project,
            combined_text,
            speaker_ids=[
                shot.dialogue_speaker_id,
                *(turn.speaker_id for turn in shot.dialogue_turns),
            ],
        )
        project_id_set = {character.id for character in project.characters}
        selected = [character_id for character_id in shot.character_ids if character_id in project_id_set]
        if selected and set(selected) != project_id_set:
            # A per-shot cast subset is an explicit storyboard decision. Keep
            # every selected subject even when prose uses a role alias such as
            # “王大爷” while the canonical character card is named “王德发”.
            return ProjectService.resolve_character_ids(
                project,
                combined_text,
                selected_ids=selected,
                speaker_ids=[
                    shot.dialogue_speaker_id,
                    *(turn.speaker_id for turn in shot.dialogue_turns),
                ],
            )

        if resolved_from_text:
            return resolved_from_text

        group_cues = ("众人", "全体", "所有人", "多人", "大家", "他们", "她们")
        if selected and any(cue in combined_text for cue in group_cues):
            return selected
        return []

    @staticmethod
    def merge_keyframe_suggestions(prompt: str, suggestions: str) -> str:
        """Append a revision instruction once, ignoring copied prompt text."""
        base = prompt.strip()
        revision = suggestions.strip()
        if not revision:
            return base

        def comparable(value: str) -> str:
            return re.sub(r"[\s，。；：、,.!?！？:;'\"“”‘’（）()【】\[\]]+", "", value).casefold()

        normalized_base = comparable(base)
        normalized_revision = comparable(revision)
        if not normalized_revision or normalized_revision == normalized_base or normalized_revision in normalized_base:
            return base
        return f"{base}\n\n【本次修改建议｜高优先级】{revision}".strip()

    @classmethod
    def apply_fixed_prop_anchor(cls, prompt: str, references: list[Asset]) -> str:
        """Keep selected recurring props physically stable across still frames."""
        prop_assets = [asset for asset in references if asset.role == AssetRole.PROP]
        base = re.sub(
            r"\n*【固定物锚点】.*?(?=\n【|\Z)",
            "",
            (prompt or "").strip(),
            flags=re.DOTALL,
        ).strip()
        if not prop_assets:
            return base
        descriptions = "；".join(
            f"{cls._compact_prompt_text(asset.name, 80)}："
            f"{cls._compact_prompt_text(asset.description or '严格按参考图保持外形、结构、材质与尺度', 420)}"
            for asset in prop_assets
        )
        anchor = (
            f"【固定物锚点】{descriptions}；以对应物品参考图为唯一造型与尺度依据，"
            "跨镜保持数量、整体尺寸、长宽高比例、结构、颜色、材质、接口和连接线一致；"
            "必须按场景中的人物、家具和门框呈现可信真实尺度，禁止缩成掌心玩具、便携小盒或随意改变结构"
        )
        return f"{base}\n{anchor}".strip()

    @classmethod
    def _automatic_keyframe_reference_assets(
        cls,
        project: Project,
        shot: Shot,
        assets: list[Asset],
        character_ids: list[str],
    ) -> list[Asset]:
        """Filter inherited character portraits to this shot's visible cast."""
        asset_map = {asset.id: asset for asset in assets}
        allowed_character_ref_list: list[str] = []
        for character in project.characters:
            if character.id not in character_ids:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            allowed_character_ref_list.extend(
                appearance.reference_asset_ids if appearance else character.reference_asset_ids
            )
        allowed_character_refs = set(allowed_character_ref_list)
        known_character_refs = {
            asset_id
            for character in project.characters
            for asset_id in [
                *character.reference_asset_ids,
                *(ref for appearance in character.appearance_profiles for ref in appearance.reference_asset_ids),
            ]
        }
        selected: list[Asset] = []
        profile = next(iter(cls.scene_profiles_for_shot(project, shot, include_strict=True)), None)
        ordered_ids = [
            *(profile.reference_asset_ids if profile else []),
            *allowed_character_ref_list,
            *(
                project.style_profile.reference_asset_ids
                if profile is None and project.style_profile and project.style_profile.approved
                else []
            ),
            *shot.reference_asset_ids,
        ]
        for asset_id in dict.fromkeys(ordered_ids):
            asset = asset_map.get(asset_id)
            if asset is None or asset.type != AssetType.IMAGE:
                continue
            if asset.role == AssetRole.CHARACTER:
                if asset.character_id and asset.character_id not in character_ids:
                    continue
                if asset_id in known_character_refs and asset_id not in allowed_character_refs:
                    continue
            selected.append(asset)
        return selected

    @classmethod
    def keyframe_reference_assets(
        cls,
        project: Project,
        shot: Shot,
        assets: list[Asset],
        character_ids: list[str],
    ) -> list[Asset]:
        """Return the exact still-image references, honoring a saved allowlist."""
        if shot.keyframe_reference_asset_ids is None:
            return cls._automatic_keyframe_reference_assets(
                project, shot, assets, character_ids
            )
        asset_map = {asset.id: asset for asset in assets}
        return [
            asset_map[asset_id]
            for asset_id in dict.fromkeys(shot.keyframe_reference_asset_ids)
            if asset_id in asset_map and asset_map[asset_id].type == AssetType.IMAGE
        ]

    @classmethod
    def keyframe_material_diagnostics(
        cls,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> dict[str, object]:
        effective_prompt = (
            shot.keyframe_prompt.strip()
            or shot.scene_description.strip()
            or shot.visual_prompt.strip()
        )
        character_ids = cls.keyframe_character_ids(project, shot, effective_prompt)
        automatic = cls._automatic_keyframe_reference_assets(
            project, shot, assets, character_ids
        )
        selected = cls.keyframe_reference_assets(
            project, shot, assets, character_ids
        )
        selected_ids = {asset.id for asset in selected}
        automatic_ids = {asset.id for asset in automatic}
        eligible_roles = {AssetRole.CHARACTER, AssetRole.PROP, AssetRole.SCENE, AssetRole.STYLE}
        available = [
            asset
            for asset in assets
            if asset.type == AssetType.IMAGE
            and (
                asset.role in eligible_roles
                or asset.id in selected_ids
                or asset.id in automatic_ids
            )
        ]

        warnings: list[str] = []
        missing_characters: list[str] = []
        for character in project.characters:
            if character.id not in character_ids:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            reference_ids = (
                appearance.reference_asset_ids
                if appearance
                else character.reference_asset_ids
            )
            if reference_ids and not selected_ids.intersection(reference_ids):
                missing_characters.append(character.name)
        if missing_characters:
            warnings.append(
                f"本镜画面包含{'、'.join(missing_characters)}，但其人物形象图未被勾选"
            )
        start_profile = cls.scene_profile_for_shot(project, shot)
        if (
            start_profile
            and start_profile.reference_asset_ids
            and not selected_ids.intersection(start_profile.reference_asset_ids)
        ):
            warnings.append(f"起始场景“{start_profile.name}”的场景母版未被勾选")
        if shot.keyframe_reference_asset_ids == []:
            warnings.append("本镜已手动取消全部首帧参考图，将只使用文字 Prompt 生成")
        bound_prop_ids = {
            asset_id
            for asset_id in shot.reference_asset_ids
            if asset_id in {asset.id for asset in assets if asset.role == AssetRole.PROP}
        }
        missing_prop_ids = bound_prop_ids - selected_ids
        if missing_prop_ids:
            missing_names = [asset.name for asset in assets if asset.id in missing_prop_ids]
            warnings.append(
                f"本镜已绑定固定物品{'、'.join(missing_names)}，但其物品参考图未被勾选"
            )

        def item(asset: Asset) -> dict[str, object]:
            return {
                "id": asset.id,
                "name": asset.name,
                "type": asset.type.value,
                "role": asset.role.value,
                "character_id": asset.character_id,
            }

        return {
            "selection_mode": (
                "automatic"
                if shot.keyframe_reference_asset_ids is None
                else "manual"
            ),
            "effective_character_ids": character_ids,
            "materials": [item(asset) for asset in selected],
            "automatic_materials": [item(asset) for asset in automatic],
            "available_materials": [item(asset) for asset in available],
            "warnings": warnings,
        }

    @staticmethod
    def compile_base_video_prompt(project: Project, motion: str, narrative: str = "") -> str:
        parts = [
            ProjectService.project_style_text(project),
            motion,
            f"画面叙事：{narrative}" if narrative else "",
        ]
        return "。".join(part.strip("。 ") for part in parts if part)

    def resolve_generation_mode(self, shot: Shot, assets: list[Asset]) -> GenerationMode:
        if shot.generation_mode != GenerationMode.AUTO:
            return shot.generation_mode
        configured_reference_ids = (
            shot.video_reference_asset_ids
            if shot.video_reference_asset_ids is not None
            else shot.reference_asset_ids
        )
        # Clean dialogue needs Ref2VA: the stock FL2VA/I2V node accepts image
        # anchors but has no audio-conditioning input. Ref2VA can keep the
        # approved group keyframe as Picture 1 while the exact clean TTS track
        # drives the named speaker's mouth. This does not change shot size.
        if (
            shot.dialogue.strip()
            and settings.H3_POSTPROCESS_AUDIO
            and settings.H3_AUDIO_MODE == "clean_tts"
            and settings.TTS_PROVIDER != "disabled"
            and (shot.keyframe_asset_id or shot.image_path or configured_reference_ids)
        ):
            return GenerationMode.R2V
        # A composed first frame is a much stronger spatial/identity anchor than
        # several independent character portraits.  Prefer it whenever it is
        # available; R2V remains available when the user selects it explicitly
        # or when no approved first frame exists.
        if shot.keyframe_asset_id or shot.image_path:
            return GenerationMode.I2V
        references = [asset for asset in assets if asset.id in configured_reference_ids]
        if references:
            return GenerationMode.R2V
        return GenerationMode.I2V

    @staticmethod
    def _compact_prompt_text(value: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", (value or "").strip())
        if len(text) <= limit:
            return text
        return text[:limit].rsplit("。", 1)[0].rstrip("，；、 ") or text[:limit]

    def _compact_subject_motion(self, shot: Shot) -> str:
        """Keep concrete shot actions while removing generic contradictory beats."""
        motion = self._normalize_motion_duration(shot.subject_motion or shot.narrative, shot.duration_seconds)
        if "第1段" in motion:
            # Storyboard generation sometimes appends generic second/third beat
            # boilerplate (push, track, then zoom) even to a fixed-camera shot.
            # The first beat already contains the authored action arc.
            motion = re.split(r"。第(?:2|3|4|5)段", motion, maxsplit=1)[0]
            motion = re.sub(r"^总时长约?\d+(?:\.\d+)?秒。?", "", motion)
            motion = re.sub(r"^第1段\([^)]*\)：", "", motion)
        return self._compact_prompt_text(motion, 520)

    @staticmethod
    def _clean_audio_design(value: str) -> str:
        parts = re.split(r"[，,；;。]", value or "")
        # Let production music be mixed during finalization. Asking H3 to create
        # dialogue and music simultaneously is a common source of muddy speech.
        excluded = ("音乐", "说话", "对白", "人声", "旁白", "回音", "口型")
        return "，".join(part.strip() for part in parts if part.strip() and not any(word in part for word in excluded))

    def _automatic_reference_assets_in_shot_order(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> list[Asset]:
        asset_map = {asset.id: asset for asset in assets}
        profiles = self.scene_profiles_for_shot(project, shot, include_strict=True)
        character_refs: list[str] = []
        for character in project.characters:
            if character.id not in shot.character_ids:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            character_refs.extend(appearance.reference_asset_ids if appearance else character.reference_asset_ids)
        style_refs = (
            project.style_profile.reference_asset_ids
            if not profiles and project.style_profile and project.style_profile.approved
            else []
        )
        continuity_input_id = self.continuity_input_asset_id(project, shot)
        ordered_ids = [
            asset_id
            for asset_id in [
                continuity_input_id,
                # A real previous tail supersedes the authored current
                # keyframe as the opening composition. Keep one composition
                # anchor, then add scene and character sheets.
                shot.keyframe_asset_id if not continuity_input_id else None,
                *(
                    asset_id
                    for profile in profiles
                    for asset_id in profile.reference_asset_ids
                ),
                *character_refs,
                *style_refs,
                *shot.reference_asset_ids,
            ]
            if asset_id
        ]
        return [asset_map[asset_id] for asset_id in dict.fromkeys(ordered_ids) if asset_id in asset_map]

    def _reference_assets_in_shot_order(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> list[Asset]:
        """Return exact video references, or the automatic ordered defaults."""
        if shot.video_reference_asset_ids is None:
            return self._automatic_reference_assets_in_shot_order(
                project, shot, assets
            )
        asset_map = {asset.id: asset for asset in assets}
        supported = {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
        manual_ids = list(dict.fromkeys(shot.video_reference_asset_ids))
        # A continuous shot has one authoritative opening composition: the
        # selected tail of its source shot.  A stale manually checked current
        # keyframe must not be sent alongside that tail, because the two
        # images describe competing frame-zero compositions.  Keep the rest
        # of the user's manual allowlist (character, scene, prop, audio, ...)
        # unchanged.
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS:
            continuity_input_id = self.continuity_input_asset_id(project, shot)
            if continuity_input_id:
                manual_ids = [
                    continuity_input_id,
                    *(
                        asset_id
                        for asset_id in manual_ids
                        if asset_id not in {continuity_input_id, shot.keyframe_asset_id}
                    ),
                ]
        return [
            asset_map[asset_id]
            for asset_id in dict.fromkeys(manual_ids)
            if asset_id in asset_map and asset_map[asset_id].type in supported
        ]

    def seedance_reference_assets(
        self,
        shot: Shot,
        assets: list[Asset],
        project: Project | None = None,
    ) -> list[Asset]:
        """Return the exact numbered-material order used by Seedance.

        Frame mode and reference mode are mutually exclusive. Full multimodal
        mode sends the composition, scene and every visible character sheet in
        one request. AUTO may resolve to strict first-frame only after the user
        confirms that the approved still already contains the complete shot.
        """
        project = project or self.require_project(shot.project_id)
        supported = {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
        resolved_mode = self.resolve_seedance_reference_mode(project, shot, assets)
        # Strict first/last-frame mode cannot be mixed with reference media.
        # Keep this legacy/explicit path, but multimodal continuous shots send
        # the previous tail as Picture 1 followed by scene and character sheets.
        if (
            shot.continuity_mode == ShotContinuityMode.CONTINUOUS
            and resolved_mode == SeedanceReferenceMode.STRICT_FIRST_FRAME
        ):
            continuity_input_id = self.continuity_input_asset_id(project, shot)
            continuity_asset = next(
                (
                    asset
                    for asset in assets
                    if asset.id == continuity_input_id and asset.type == AssetType.IMAGE
                ),
                None,
            )
            if continuity_asset:
                return [continuity_asset]
        keyframe = next(
            (
                asset
                for asset in assets
                if asset.id == shot.keyframe_asset_id and asset.type == AssetType.IMAGE
            ),
            None,
        )
        if keyframe and resolved_mode == SeedanceReferenceMode.STRICT_FIRST_FRAME:
            return [keyframe]
        return [
            asset
            for asset in self._reference_assets_in_shot_order(project, shot, assets)
            if asset.type in supported
        ]

    @staticmethod
    def _seedance_opening_text(shot: Shot) -> str:
        """Return only frame-zero prose, excluding generic cast/style anchors."""
        prompt = shot.keyframe_prompt or shot.scene_description
        return re.split(
            r"【(?:角色锚点|风格锚点|静帧约束|本次修改建议|场景档案)",
            prompt,
            maxsplit=1,
        )[0]

    def seedance_identity_risk_characters(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> list[str]:
        """Find referenced cast whose full appearance is not anchored at 0s."""
        if not shot.keyframe_asset_id:
            return []
        asset_ids = {asset.id for asset in assets if asset.type == AssetType.IMAGE}
        opening = self._normalized_character_label(self._seedance_opening_text(shot))
        reveal_cues = (
            "半个脑袋",
            "只露出头",
            "头部轮廓",
            "从墙里钻出",
            "从墙内探出",
            "从门外进入",
            "走入画面",
            "进入画面",
            "随后出现",
            "逐渐现身",
        )
        risky: list[str] = []
        for character in project.characters:
            if character.id not in shot.character_ids:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            refs = appearance.reference_asset_ids if appearance else character.reference_asset_ids
            if not any(asset_id in asset_ids for asset_id in refs):
                continue
            aliases = [
                self._normalized_character_label(alias)
                for alias in self._character_aliases(character.name)
            ]
            visible_at_zero = any(alias and alias in opening for alias in aliases)
            partial_reveal = False
            for cue in reveal_cues:
                start = opening.find(cue)
                while start >= 0:
                    window = opening[max(0, start - 12) : start + len(cue) + 12]
                    if any(alias and alias in window for alias in aliases):
                        partial_reveal = True
                        break
                    start = opening.find(cue, start + len(cue))
                if partial_reveal:
                    break
            if not visible_at_zero or partial_reveal:
                risky.append(character.name)
        return risky

    def resolve_seedance_reference_mode(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> SeedanceReferenceMode:
        # An in-shot location transition needs both ordered scene sheets. The
        # strict first-frame API path accepts only one image, so use the full
        # multimodal path for this case even if an older shot requested strict.
        if len(self.scene_profiles_for_shot(project, shot)) > 1:
            return SeedanceReferenceMode.MULTIMODAL_REFERENCE
        if shot.seedance_reference_mode != SeedanceReferenceMode.AUTO:
            return shot.seedance_reference_mode
        return self._resolve_auto_seedance_reference_mode(project, shot, assets)[0]

    def _resolve_auto_seedance_reference_mode(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> tuple[SeedanceReferenceMode, str]:
        """Resolve AUTO without guessing that an arbitrary still is complete.

        The still-image generator cannot expose reliable pixel-level coverage
        to this service. AUTO therefore trusts the user's per-shot completeness
        choice, while still forcing multimodal references for transitions,
        late reveals, and explicit extra media selections.
        """
        if len(self.scene_profiles_for_shot(project, shot)) > 1:
            return (
                SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                "跨场景镜头必须携带起始和目标场景母版",
            )
        keyframe = next(
            (
                asset
                for asset in assets
                if asset.id == shot.keyframe_asset_id
                and asset.type == AssetType.IMAGE
            ),
            None,
        )
        if keyframe is None:
            return (
                SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                "尚未登记可提交的完整首帧",
            )
        if shot.first_frame_completeness == FirstFrameCompleteness.INCOMPLETE:
            return (
                SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                "首帧已标记为信息不完整",
            )
        if shot.first_frame_completeness != FirstFrameCompleteness.COMPLETE:
            return (
                SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                "首帧完整度尚未确认，默认保留全模态",
            )
        risky = self.seedance_identity_risk_characters(project, shot, assets)
        if risky:
            return (
                SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                f"{'、'.join(risky)}在首帧后才完整出场",
            )
        if shot.video_reference_asset_ids is not None:
            manual_ids = set(shot.video_reference_asset_ids)
            if not manual_ids:
                return (
                    SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                    "本镜已手动取消视频参考素材",
                )
            extra_ids = manual_ids - {keyframe.id}
            if extra_ids:
                return (
                    SeedanceReferenceMode.MULTIMODAL_REFERENCE,
                    "手动清单包含首帧之外的参考素材",
                )
        return (
            SeedanceReferenceMode.STRICT_FIRST_FRAME,
            "首帧已确认包含本镜所需人物、场景和固定物品",
        )

    def seedance_reference_mode_reason(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> str:
        """Explain the resolved Seedance mode for the material diagnostics UI."""
        if len(self.scene_profiles_for_shot(project, shot)) > 1:
            return "跨场景镜头必须携带起始和目标场景母版"
        if shot.seedance_reference_mode == SeedanceReferenceMode.AUTO:
            return self._resolve_auto_seedance_reference_mode(project, shot, assets)[1]
        if shot.seedance_reference_mode == SeedanceReferenceMode.STRICT_FIRST_FRAME:
            return "手动指定严格首帧"
        return "手动指定全模态参考"

    def seedance_material_diagnostics(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> dict[str, object]:
        refs = self.seedance_reference_assets(shot, assets, project)
        resolved = self.resolve_seedance_reference_mode(project, shot, assets)
        risky = self.seedance_identity_risk_characters(project, shot, assets)
        first_frame_id = self.seedance_first_frame_asset_id(project, shot, assets)
        if resolved == SeedanceReferenceMode.STRICT_FIRST_FRAME:
            automatic = [
                asset for asset in assets
                if asset.id == first_frame_id and asset.type == AssetType.IMAGE
            ]
        else:
            automatic = [
                asset
                for asset in self._automatic_reference_assets_in_shot_order(
                    project, shot, assets
                )
                if asset.type in {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
            ]
        selected_ids = {asset.id for asset in refs}
        automatic_ids = {asset.id for asset in automatic}
        continuity_id = self.continuity_input_asset_id(project, shot)
        available = [
            asset
            for asset in assets
            if asset.type in {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
            and asset.role != AssetRole.OUTPUT
            and (
                asset.role not in {AssetRole.KEYFRAME, AssetRole.LAST_FRAME}
                or asset.id in {
                    shot.keyframe_asset_id,
                    continuity_id,
                    *selected_ids,
                    *automatic_ids,
                }
            )
        ]
        warnings: list[str] = []
        missing_character_refs: list[str] = []
        for character in project.characters:
            if character.id not in shot.character_ids:
                continue
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            reference_ids = appearance.reference_asset_ids if appearance else character.reference_asset_ids
            if reference_ids and not selected_ids.intersection(reference_ids):
                missing_character_refs.append(character.name)
        if risky and resolved == SeedanceReferenceMode.STRICT_FIRST_FRAME:
            warnings.append(
                f"严格首帧只提交一张构图图，{'、'.join(risky)}在 0 秒未完整出场，服装和发型可能漂移"
            )
        elif missing_character_refs:
            warnings.append(
                f"本镜包含{'、'.join(missing_character_refs)}，但其人物形象图未被选入视频参考"
            )
        elif risky:
            warnings.append(
                f"检测到{'、'.join(risky)}在首帧后才完整出场，已同时提交人物参考图以锁定服装和发型"
            )
        if len(self.scene_profiles_for_shot(project, shot)) > 1:
            scene_ids = {
                asset_id
                for profile in self.scene_profiles_for_shot(project, shot)
                for asset_id in profile.reference_asset_ids
            }
            if scene_ids.issubset(selected_ids):
                warnings.append("跨场景镜头已切换为全模态参考，并按起始→目标顺序提交两张场景母版")
            else:
                warnings.append("跨场景镜头缺少起始或目标场景母版，请检查手动参考选择")
        if shot.video_reference_asset_ids == []:
            warnings.append("本镜已手动取消全部视频参考素材")
        prop_assets = {
            asset.id: asset
            for asset in assets
            if asset.role == AssetRole.PROP
        }
        missing_prop_ids = {
            asset_id
            for asset_id in shot.reference_asset_ids
            if asset_id in prop_assets and asset_id not in selected_ids
        }
        if missing_prop_ids:
            warnings.append(
                "本镜已绑定固定物品"
                + "、".join(prop_assets[asset_id].name for asset_id in missing_prop_ids)
                + "，但其物品参考图未被选入视频参考"
            )

        def item(asset: Asset) -> dict[str, object]:
            return {
                "id": asset.id,
                "name": asset.name,
                "type": asset.type.value,
                "role": asset.role.value,
                "character_id": asset.character_id,
            }
        return {
            "requested_mode": shot.seedance_reference_mode.value,
            "resolved_mode": resolved.value,
            "first_frame_completeness": shot.first_frame_completeness.value,
            "reference_mode_reason": self.seedance_reference_mode_reason(project, shot, assets),
            "first_frame_asset_id": first_frame_id,
            "identity_risk_characters": risky,
            "materials": [item(asset) for asset in refs],
            "automatic_materials": [item(asset) for asset in automatic],
            "available_materials": [item(asset) for asset in available],
            "selection_mode": "automatic" if shot.video_reference_asset_ids is None else "manual",
            "warnings": warnings,
        }

    def continuity_source_shot(self, project: Project, shot: Shot) -> Shot | None:
        if shot.continuity_mode != ShotContinuityMode.CONTINUOUS:
            return None
        shots = self.store.list_shots(project.id)
        if shot.continuity_source_shot_id:
            return next((item for item in shots if item.id == shot.continuity_source_shot_id), None)
        return next((item for item in shots if item.ordinal == shot.ordinal - 1), None)

    def continuity_input_asset_id(self, project: Project, shot: Shot) -> str | None:
        source = self.continuity_source_shot(project, shot)
        return source.last_frame_asset_id if source else None

    def seedance_first_frame_asset_id(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> str | None:
        asset_ids = {asset.id for asset in assets if asset.type == AssetType.IMAGE}
        continuity_input_id = self.continuity_input_asset_id(project, shot)
        if (
            self.resolve_seedance_reference_mode(project, shot, assets)
            == SeedanceReferenceMode.STRICT_FIRST_FRAME
            and shot.continuity_mode == ShotContinuityMode.CONTINUOUS
            and continuity_input_id in asset_ids
        ):
            return continuity_input_id
        if (
            self.resolve_seedance_reference_mode(project, shot, assets)
            == SeedanceReferenceMode.STRICT_FIRST_FRAME
            and shot.keyframe_asset_id in asset_ids
        ):
            return shot.keyframe_asset_id
        return None

    def compile_seedance_prompt(self, project: Project, shot: Shot, assets: list[Asset]) -> str:
        refs = self.seedance_reference_assets(shot, assets, project)
        image_numbers: dict[str, int] = {}
        manifests: list[str] = []
        counters = {AssetType.IMAGE: 0, AssetType.VIDEO: 0, AssetType.AUDIO: 0}
        labels = {AssetType.IMAGE: "图片", AssetType.VIDEO: "视频", AssetType.AUDIO: "音频"}
        for asset in refs:
            counters[asset.type] += 1
            number = counters[asset.type]
            label = f"{labels[asset.type]}{number}"
            # Asset descriptions may contain the complete keyframe generation
            # prompt. Material manifests stay short so they do not flood the
            # video prompt with duplicate scene/style/cast instructions.
            manifests.append(f"@{label}（{self._compact_prompt_text(asset.name, 48)}）")
            if asset.type == AssetType.IMAGE:
                image_numbers[asset.id] = number

        visible_characters = [
            character for character in project.characters if character.id in shot.character_ids
        ]
        first_frame_id = self.seedance_first_frame_asset_id(project, shot, assets)
        strict_first_frame = bool(
            first_frame_id
            and len(refs) == 1
            and refs[0].id == first_frame_id
            and refs[0].type == AssetType.IMAGE
        )
        definitions: list[str] = []
        for character in visible_characters:
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            character_ref_ids = appearance.reference_asset_ids if appearance else character.reference_asset_ids
            pictures = [
                image_numbers[asset_id]
                for asset_id in character_ref_ids
                if asset_id in image_numbers
            ]
            traits = self._compact_prompt_text(
                "，".join(
                    item
                    for item in [
                        appearance.time_context if appearance else "",
                        appearance.description if appearance else character.description,
                        appearance.wardrobe if appearance else character.wardrobe,
                    ]
                    if item
                ),
                180,
            )
            if strict_first_frame:
                definitions.append(
                    f"@图片1中的人物“{character.name}”保持身份、脸型、发型、服装和配饰不变"
                )
            elif pictures:
                definitions.append(
                    f"将@图片{pictures[0]}中{traits or character.name}的角色定义为“{character.name}”"
                )
            else:
                definitions.append(f"将{traits or character.name}的角色定义为“{character.name}”")

        scene_profiles = self.scene_profiles_for_shot(project, shot)
        for index, profile in enumerate(scene_profiles):
            pictures = [
                image_numbers[asset_id]
                for asset_id in profile.reference_asset_ids
                if asset_id in image_numbers
            ]
            if pictures:
                phase = "转场前的起始场景" if index == 0 and len(scene_profiles) > 1 else (
                    "转场完成后的目标场景" if index == 1 else "本镜场景"
                )
                definitions.append(
                    f"将@图片{pictures[0]}中的无人环境定义为{phase}“{profile.name}”，"
                    "只锁定空间布局、固定陈设、材质、色彩与光源，不从场景图新增人物"
                )

        for asset in refs:
            if asset.role != AssetRole.PROP or asset.id not in image_numbers:
                continue
            prop_details = self._compact_prompt_text(
                asset.description or "严格按参考图保持外形、结构、材质与尺度",
                360,
            )
            definitions.append(
                f"将@图片{image_numbers[asset.id]}中的固定物品定义为“{self._compact_prompt_text(asset.name, 64)}”："
                f"{prop_details}；全片保持数量、真实尺寸、比例、结构、颜色、接口与连接线一致，"
                "按人物和家具呈现可信尺度，禁止缩成掌心玩具或便携小盒"
            )

        style = self._compact_prompt_text(
            self.project_rendering_style_text(project)
            if scene_profiles
            else self.project_style_text(project),
            180,
        )
        event = self._compact_prompt_text(shot.narrative or shot.scene_description, 220)
        audio_design = self._compact_prompt_text(shot.audio_design, 160)

        visual_beats = shot.visual_beats or self._scale_visual_beats(
            [],
            shot.duration_seconds,
            shot.duration_seconds,
            shot_size=shot.shot_size,
            camera_angle=shot.camera_angle,
            camera_motion=shot.camera_motion or "固定镜头",
            fallback_action=shot.subject_motion or shot.video_prompt_source,
        )
        beat_lines: list[str] = []
        for beat in visual_beats:
            action = self._compact_prompt_text(beat.subject_action, 180)
            environment = self._compact_prompt_text(beat.environment_action, 100)
            camera = "，".join(
                item
                for item in [
                    beat.shot_size or shot.shot_size,
                    beat.camera_angle or shot.camera_angle,
                    beat.camera_motion or shot.camera_motion or "固定镜头",
                ]
                if item
            )
            details = "；".join(
                item
                for item in [
                    f"{beat.purpose}" if beat.purpose else "",
                    f"主体：{action}" if action else "",
                    f"环境：{environment}" if environment else "",
                    f"镜头：{camera}" if camera else "",
                    f"声音卡点：{self._compact_prompt_text(beat.sound_cue, 80)}" if beat.sound_cue else "",
                ]
                if item
            )
            beat_lines.append(f"{beat.start_seconds:.2f}–{beat.end_seconds:.2f}秒：{details}")

        voice_lines: list[str] = []
        effective_voice_events = self._resolve_voice_events(
            project,
            self._effective_voice_events(shot),
            shot.duration_seconds,
            shot.duration_seconds,
        )
        if effective_voice_events:
            for voice in effective_voice_events:
                text = self._compact_prompt_text(self.spoken_dialogue_text(voice.text), 180)
                if not text:
                    continue
                speaker = next(
                    (item for item in project.characters if item.id == voice.speaker_id),
                    None,
                )
                timing = f"{voice.start_seconds:.2f}–{voice.end_seconds:.2f}秒"
                if voice.kind == "character" and speaker:
                    lip_rule = "自然口型同步" if voice.lip_sync else "声音出现但不强调口型"
                    voice_anchor = self.character_voice_anchor(speaker)
                    voice_lines.append(
                        f"{timing}，唯一发言者“{speaker.name}”说：“{text}”，{lip_rule}；"
                        f"声音锚点：{voice_anchor}；本镜保持同一音色、年龄感、性别感、音高和口音；"
                        "节奏描述只控制表演情绪，不为塞满台词而加速，保持自然清晰和完整停顿；其余人物闭嘴"
                    )
                else:
                    source_names = {
                        "system_vo": "系统播报",
                        "narration": "旁白",
                        "offscreen": "画外音",
                    }
                    source = source_names.get(voice.kind, voice.speaker_name or "画外音")
                    voice_anchor = self._voice_anchor_for_event(project, voice)
                    voice_lines.append(
                        f"{timing}，{source}说：“{text}”；声音锚点：{voice_anchor}；"
                        "声音来自画外，画面内所有人物不张嘴、不做说话口型"
                    )
        elif shot.dialogue_turns:
            for turn in shot.dialogue_turns:
                speaker = next((item for item in project.characters if item.id == turn.speaker_id), None)
                if turn.text.strip():
                    voice_anchor = self.character_voice_anchor(speaker) if speaker else "清晰自然的画外人声"
                    voice_lines.append(
                        f"“{speaker.name if speaker else '画外音'}”说：“{self.spoken_dialogue_text(turn.text)}”；"
                        f"声音锚点：{voice_anchor}；其余人物闭嘴"
                    )
        elif shot.dialogue.strip():
            speaker = next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)
            if speaker:
                voice_anchor = self.character_voice_anchor(speaker)
                voice_lines.append(
                    f"唯一发言者“{speaker.name}”说：“{self.spoken_dialogue_text(shot.dialogue)}”，自然口型同步；"
                    f"声音锚点：{voice_anchor}；本镜保持同一音色、年龄感、性别感、音高和口音；"
                    "节奏描述只控制表演情绪，不为塞满台词而加速，保持自然清晰和完整停顿；其余人物闭嘴"
                )
            else:
                voice_lines.append(
                    f"画外音说：“{self.spoken_dialogue_text(shot.dialogue)}”；画面内所有人物不张嘴、不做说话口型"
                )

        reference_instructions: list[str] = []
        if image_numbers:
            if strict_first_frame:
                reference_instructions.append(
                    "@图片1是严格首帧：目标视频 0.00 秒必须与其构图一致，并继承人物身份、人数、服装、站位、道具、场景、景别和光影；随后从该状态立即开始第一个节拍"
                )
            else:
                reference_instructions.append("使用上述@图片锁定对应角色、服装、场景、固定物品外形尺度或视觉风格")
                keyframe_number = image_numbers.get(shot.keyframe_asset_id or "")
                if keyframe_number:
                    reference_instructions.append(
                        f"@图片{keyframe_number}是开场构图参考，0.00 秒尽量保持其机位、人物站位、道具、场景和光影；"
                        "若角色在首帧后才完整出现，其脸型、发型、服装和配饰以对应人物参考图为准"
                    )
        if counters[AssetType.VIDEO]:
            reference_instructions.append("参考上述@视频的主体动作、节奏或运镜，不复制无关内容")
        if counters[AssetType.AUDIO]:
            reference_instructions.append("参考上述@音频的音色、对白或声音氛围")

        continuity = ""
        if scene_profiles:
            continuity = self.scene_transition_contract(scene_profiles)
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS:
            continuity = (
                continuity + "。" if continuity else "【连续长镜头】"
            ) + "@图片1是上一镜真实尾帧；从该画面无跳切地继续动作、视线、站位、光线和声音，不复述上一镜事件"

        if shot.text_policy == "reference_locked":
            text_constraint = "仅保留参考素材中已经清晰存在且用户锁定的文字，不新增或改写任何文字"
        else:
            text_constraint = (
                "画面内不生成任何可读文字、汉字、字母、数字、字幕、标题、UI 文案、Logo、水印或乱码；"
                "所有信息文字统一后期叠加"
            )

        parts = [
            "【素材定义】" + "；".join(definitions) if definitions else "",
            "【可用素材】" + "；".join(manifests) if manifests else "",
            "【参考要求】" + "；".join(reference_instructions) if reference_instructions else "",
            continuity,
            f"【镜头任务】总时长 {shot.duration_seconds:.2f} 秒；{event}" if event else f"【镜头任务】总时长 {shot.duration_seconds:.2f} 秒",
            "【逐段时间轴】\n" + "\n".join(beat_lines),
            f"【对白与声音】{'；'.join(voice_lines)}。{audio_design or '生成与画面动作同步的清晰环境声'}" if voice_lines else f"【声音】无对白；{audio_design or '生成与画面动作同步的克制环境声'}",
            f"【视觉风格与画质】{style or '延续项目统一风格'}；主体清晰，动作可读，人物五官与肢体稳定",
            (
                "【约束】" + text_constraint + "；严格保持同一角色身份、脸型、发型、服装与人物数量；"
                "禁止人物复制、融合、重影、畸形肢体、身份互换、说话人错位和重复台词；"
                "每个时间段只执行该段指定的一个主体动作和一种主要运镜，段与段之间连续衔接"
            ),
        ]
        return "\n".join(part for part in parts if part).strip()

    def effective_seedance_prompt(self, project: Project, shot: Shot, assets: list[Asset]) -> str:
        """Reuse only prompts compiled by the current director contract."""
        if (
            shot.seedance_prompt.strip()
            and shot.seedance_prompt_version == SEEDANCE_PROMPT_VERSION
            and shot.seedance_prompt_source_revision == shot.content_revision
        ):
            return shot.seedance_prompt.strip()
        return self.compile_seedance_prompt(project, shot, assets)

    def generate_seedance_prompts(
        self,
        project_id: str,
        shot_ids: list[str] | None = None,
    ) -> list[Shot]:
        project = self.require_project(project_id)
        all_shots = self.store.list_shots(project_id)
        selected = set(shot_ids) if shot_ids is not None else None
        shots = [shot for shot in all_shots if selected is None or shot.id in selected]
        if not shots:
            raise ValueError("请至少选择一个镜头")
        assets = self.store.list_assets(project_id)
        saved: list[Shot] = []
        for shot in shots:
            self.synchronize_shot_cast(project, shot)
            effective_voice_events = self._effective_voice_events(shot)
            if effective_voice_events and not shot.voice_events:
                shot.voice_events = effective_voice_events
                shot.dialogue_speaker_id = None
                shot.dialogue_turns = []
            has_external_voice = any(event.kind != "character" for event in effective_voice_events)
            if shot.dialogue.strip() and not shot.dialogue_turns and not has_external_voice:
                turns = self.resolve_dialogue_turns(project, shot)
                shot.dialogue_turns = [DialogueTurn.model_validate(turn) for turn in turns]
                if len(turns) == 1:
                    shot.dialogue_speaker_id = turns[0]["speaker_id"]
            shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
            shot.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
            shot.seedance_prompt_source_revision = shot.content_revision
            shot.version += 1
            saved.append(self.store.save_shot(shot))
        return saved

    def migrate_all_seedance_shots_to_multimodal(self) -> dict[str, int]:
        """Upgrade legacy Seedance prompts without overwriting current choices.

        Older prompt versions predate the explicit AUTO/STRICT/MULTIMODAL
        policy, so they are rebuilt with multimodal references once. Current
        prompts keep their per-shot mode and first-frame completeness choice.
        """
        project_count = 0
        shot_count = 0
        for project in self.store.list_projects():
            assets = self.store.list_assets(project.id)
            changed_in_project = 0
            for shot in self.store.list_shots(project.id):
                if shot.seedance_prompt_version == SEEDANCE_PROMPT_VERSION:
                    continue
                shot.seedance_reference_mode = SeedanceReferenceMode.MULTIMODAL_REFERENCE
                self.synchronize_shot_cast(project, shot)
                shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
                shot.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
                shot.seedance_prompt_source_revision = shot.content_revision
                shot.version += 1
                self.store.save_shot(shot)
                changed_in_project += 1
            if changed_in_project:
                project_count += 1
                shot_count += changed_in_project
        return {"project_count": project_count, "shot_count": shot_count}

    @staticmethod
    def _dialogue_text_tag(value: str) -> str:
        text = ProjectService.spoken_dialogue_text(value).strip()
        if text and text[-1] not in "。！？.!?":
            text += "。"
        return f"<d>[Chinese] {text}</d>"

    @staticmethod
    def _subject_reference_context(
        project: Project,
        shot: Shot,
        refs: list[Asset],
    ) -> tuple[list[str], dict[str, int]]:
        """Build stable Subject/Picture definitions for the current R2V call."""
        picture_numbers = {
            asset.id: index
            for index, asset in enumerate((item for item in refs if item.type == AssetType.IMAGE), start=1)
        }
        visible_ids = list(dict.fromkeys(shot.character_ids))
        visible = [
            character
            for character_id in visible_ids
            for character in project.characters
            if character.id == character_id
        ]
        subject_numbers = {character.id: index for index, character in enumerate(visible, start=1)}
        definitions: list[str] = []
        for character in visible:
            subject_number = subject_numbers[character.id]
            appearance = next(
                (
                    item
                    for item in character.appearance_profiles
                    if item.id == shot.character_appearance_ids.get(character.id)
                ),
                None,
            )
            character_ref_ids = appearance.reference_asset_ids if appearance else character.reference_asset_ids
            ref_pictures = [
                picture_numbers[asset_id]
                for asset_id in character_ref_ids
                if asset_id in picture_numbers
            ]
            picture_note = (
                " whose identity is defined by " + ", ".join(f"<Picture {item}>" for item in ref_pictures)
                if ref_pictures
                else " whose appearance and position are preserved from <Picture 1>"
            )
            profile = "，".join(
                item
                for item in [
                    appearance.time_context if appearance else "",
                    appearance.description if appearance else character.description,
                    appearance.wardrobe if appearance else character.wardrobe,
                ]
                if item
            )
            definitions.append(
                f"<Subject {subject_number}> is {character.name}{picture_note}; fixed profile: {profile or character.name}."
            )
        return definitions, subject_numbers

    def compile_h3_prompt(self, project: Project, shot: Shot, assets: list[Asset]) -> str:
        # A director upgrade or reference edit may make this text need review,
        # but neither authorizes replacing paid/user-authored text with a template.
        if shot.h3_prompt_skill_output.strip():
            return shot.h3_prompt_skill_output.strip()
        mode = self.resolve_generation_mode(shot, assets)
        effective_voice_events = self._resolve_voice_events(
            project,
            self._effective_voice_events(shot),
            shot.duration_seconds,
            shot.duration_seconds,
        )
        character_voice_events = [
            event for event in effective_voice_events if event.kind == "character" and event.speaker_id
        ]
        dialogue_audio_bound = bool(
            mode == GenerationMode.R2V
            and shot.dialogue.strip()
            and (not effective_voice_events or character_voice_events)
            and settings.H3_POSTPROCESS_AUDIO
            and settings.H3_AUDIO_MODE == "clean_tts"
            and settings.TTS_PROVIDER != "disabled"
        )
        scene_profiles = self.scene_profiles_for_shot(project, shot)
        scene_contract = self.scene_transition_contract(scene_profiles)
        style = self._compact_prompt_text(self.shot_style_text(project, shot), 360)
        narrative = self._compact_prompt_text(shot.narrative or shot.scene_description, 260)
        motion = self._compact_subject_motion(shot)
        camera = "，".join(item for item in [shot.shot_size, shot.camera_angle, shot.lens, shot.camera_motion] if item)
        audio_design = self._clean_audio_design(shot.audio_design)
        if effective_voice_events:
            voice_parts: list[str] = []
            for event in effective_voice_events:
                spoken = self._compact_prompt_text(self.spoken_dialogue_text(event.text), 160)
                if not spoken:
                    continue
                timing = f"{event.start_seconds:.2f}-{event.end_seconds:.2f}秒"
                speaker = next(
                    (item for item in project.characters if item.id == event.speaker_id),
                    None,
                )
                if event.kind == "character" and speaker:
                    speech = "由<Audio 1>提供唯一对白" if dialogue_audio_bound else f"说{self._dialogue_text_tag(spoken)}"
                    lip_rule = "自然口型同步" if event.lip_sync else "不强调口型"
                    voice_direction = self.h3_voice_direction_english(project, event)
                    non_speakers = [
                        item.name
                        for item in project.characters
                        if item.id in shot.character_ids and item.id != speaker.id
                    ]
                    silence_rule = f"{','.join(non_speakers)}全程闭嘴" if non_speakers else "其余人物闭嘴"
                    voice_parts.append(
                        f"{timing}，唯一发言者明确为{speaker.name}，{speech}，{lip_rule}；"
                        f"silent voice direction (non-spoken): {voice_direction}；"
                        f"节奏描述只控制表演情绪，不为塞满台词而加速，保持自然清晰和完整停顿；{silence_rule}"
                    )
                else:
                    source = {
                        "system_vo": "系统播报",
                        "narration": "旁白",
                        "offscreen": "画外音",
                    }.get(event.kind, event.speaker_name or "画外音")
                    voice_direction = self.h3_voice_direction_english(project, event)
                    voice_parts.append(
                        f"{timing}，{source}说{self._dialogue_text_tag(spoken)}；"
                        f"silent voice direction (non-spoken): {voice_direction}；"
                        "声音来自画外，画面人物均不张嘴、不做口型"
                    )
            audio = "声音事件：" + "；".join(voice_parts) + f"；{audio_design or '自然环境声与动作音同步'}；无背景音乐"
        elif shot.dialogue.strip():
            if len(shot.dialogue_turns) > 1 and not shot.dialogue_speaker_id:
                turn_summary = "；".join(
                    (
                        f"{speaker.name if speaker else '旁白'}说“{self._compact_prompt_text(turn.text, 80)}”"
                        f"，silent voice direction (non-spoken): "
                        f"{self.h3_character_voice_direction_english(speaker) if speaker else 'clear, natural off-screen narration'}"
                    )
                    for turn in shot.dialogue_turns
                    for speaker in [next((item for item in project.characters if item.id == turn.speaker_id), None)]
                )
                audio = (
                    f"多角色对白将由系统拆成逐人短段续帧生成：{turn_summary}；"
                    "任一短段只有当前发言者张嘴，其他人物闭嘴，禁止抢话和角色互换"
                )
            else:
                if not shot.dialogue_speaker_id:
                    shot.dialogue_speaker_id = self._resolve_dialogue_speaker_id(
                        project,
                        "",
                        shot.narrative,
                        shot.subject_motion,
                        shot.dialogue,
                    )
                speaker = next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)
                spoken = self._compact_prompt_text(self.spoken_dialogue_text(shot.dialogue), 220)
                if speaker:
                    non_speakers = [
                        item.name
                        for item in project.characters
                        if item.id in shot.character_ids and item.id != speaker.id
                    ]
                    silence_rule = f"{','.join(non_speakers)}全程闭嘴，只做克制反应" if non_speakers else "其他人物全程闭嘴"
                    voice = (
                        f"，silent voice direction (non-spoken): {self.h3_character_voice_direction_english(speaker)}，"
                        "节奏描述只控制表演情绪，不为塞满台词而加速，保持自然清晰和完整停顿"
                    )
                    if dialogue_audio_bound:
                        # The R2V audio-binding block below owns the one and
                        # only <d> tag. Repeating the tag here makes H3 treat
                        # the same line as a second utterance and may hand it
                        # to another visible face.
                        audio = (
                            f"对白表演补充约束：唯一发言者明确为{speaker.name} (S1){voice}；"
                            "整段只执行前述由 <Audio 1> 和唯一对白标签定义的一次发言，禁止重复台词；"
                            f"{silence_rule}；单人声、口型同步、无抢话、无角色互换、无回声、无爆音、无背景音乐"
                        )
                    else:
                        dialogue_tag = self._dialogue_text_tag(spoken)
                        audio = (
                            f"对白表演：唯一发言者明确为{speaker.name} (S1){voice}，"
                            f"只由{speaker.name} (S1) 嘴唇自然张合并说 {dialogue_tag}；"
                            f"{silence_rule}；单人声、口型同步、无抢话、无角色互换、无回声、无爆音、无背景音乐"
                        )
                else:
                    audio = (
                        f"对白表演：旁白说“{spoken}”；画面内所有人物全程闭嘴，不做说话口型；"
                        "单人声、无重叠人声、无回声、无爆音、无背景音乐"
                    )
        else:
            visible_names = [
                item.name for item in project.characters if item.id in shot.character_ids
            ]
            silence_subjects = "、".join(visible_names) if visible_names else "画面内所有人物"
            audio = (
                f"声音：无对白，{silence_subjects}全程嘴唇闭合、不做说话口型；"
                f"{audio_design or '仅保留轻微自然环境声'}，无背景音乐"
            )

        visual_beats = shot.visual_beats or self._scale_visual_beats(
            [],
            shot.duration_seconds,
            shot.duration_seconds,
            shot_size=shot.shot_size,
            camera_angle=shot.camera_angle,
            camera_motion=shot.camera_motion or "固定镜头",
            fallback_action=shot.subject_motion or shot.video_prompt_source,
        )
        beat_timeline = "；".join(
            f"{beat.start_seconds:.2f}-{beat.end_seconds:.2f}秒："
            + "，".join(
                item
                for item in [
                    beat.purpose,
                    beat.subject_action,
                    f"环境反馈{beat.environment_action}" if beat.environment_action else "",
                    beat.shot_size or shot.shot_size,
                    beat.camera_angle or shot.camera_angle,
                    beat.camera_motion or shot.camera_motion,
                    f"声音卡点{beat.sound_cue}" if beat.sound_cue else "",
                ]
                if item
            )
            for beat in visual_beats
        )
        text_rule = (
            "仅保留参考画面中已锁定的可读文字，不新增或改写文字"
            if shot.text_policy == "reference_locked"
            else (
                "H3 可显示分镜明确写出的标题、标识、界面、文件或其他画面文字；逐字保留原文并在对应时间段出现，"
                "不翻译、不改写、不补充未授权文案；不得把对白或旁白自动转成字幕"
                if shot.text_policy != "none"
                else "画面内不出现任何字幕、台词文字、标题、标识、水印、数字、界面文字或乱码；不得把对白或声音转写成可见文字"
            )
        )
        details = [
            f"纯净画面约束：{text_rule}",
            f"画面风格：{style}" if style else "",
            scene_contract,
            f"场景与事件：{narrative}" if narrative else "",
            f"逐段动作弧线：{beat_timeline}" if beat_timeline else f"动作：{motion}" if motion else "",
            f"整镜基础镜头：{camera}；每个时间段只执行该段的一种主要运镜" if camera else "",
            audio,
            self.h3_silent_voice_contract(project, effective_voice_events),
            f"时长约{shot.duration_seconds:.2f}秒",
        ]
        if mode == GenerationMode.R2V:
            refs = self._reference_assets_in_shot_order(project, shot, assets)
            if not refs and shot.keyframe_asset_id:
                keyframe = next((asset for asset in assets if asset.id == shot.keyframe_asset_id), None)
                if keyframe:
                    refs = [keyframe]
            picture_index = video_index = 0
            reserves_dialogue_audio = dialogue_audio_bound
            audio_index = 1 if reserves_dialogue_audio else 0
            tag_notes: list[str] = []
            for asset in refs:
                if asset.type == AssetType.IMAGE:
                    picture_index += 1
                    prop_note = (
                        f"；{self._compact_prompt_text(asset.description, 360)}；"
                        "严格保持真实尺度、长宽高比例、结构、接口与连接线，禁止缩成掌心小物"
                        if asset.role == AssetRole.PROP else ""
                    )
                    tag_notes.append(
                        f"<Picture {picture_index}> 用作{asset.role.value}参考："
                        f"{self._compact_prompt_text(asset.name, 64)}{prop_note}"
                    )
                elif asset.type == AssetType.VIDEO:
                    video_index += 1
                    tag_notes.append(f"<Video {video_index}> 用作动作与镜头参考：{self._compact_prompt_text(asset.name, 64)}")
                elif asset.type == AssetType.AUDIO:
                    audio_index += 1
                    tag_notes.append(f"<Audio {audio_index}> 用作声音参考：{self._compact_prompt_text(asset.name, 64)}")
            subject_definitions, subject_numbers = self._subject_reference_context(project, shot, refs)
            composition_note = ""
            if picture_index:
                composition_note = (
                    "The target shot begins from <Picture 1> and preserves its exact character count, identities, "
                    "wardrobe, blocking, background, camera angle, and authored shot size."
                )
            speaker = next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)
            dialogue_binding = ""
            if reserves_dialogue_audio and speaker:
                subject_number = subject_numbers.get(speaker.id)
                speaker_label = f"<Subject {subject_number}> (S1)" if subject_number else f"{speaker.name} (S1)"
                dialogue_tag = self._dialogue_text_tag(shot.dialogue)
                voice_direction = self.h3_character_voice_direction_english(speaker)
                dialogue_binding = (
                    f"<Audio 1> is the directly reused clean Chinese dialogue track for {speaker_label}. "
                    f"Its silent, non-spoken voice direction is: {voice_direction}. Preserve that age, gender presentation, timbre, "
                    "pitch range, accent, pacing, and emotional delivery for the whole utterance. "
                    f"Only {speaker_label} physically speaks and says, {dialogue_tag} Mouth movement and timing "
                    "follow <Audio 1> exactly. Every other visible subject keeps closed lips and makes only subtle "
                    "listening reactions. Do not change to a single-person close-up; preserve the authored group, "
                    "two-shot, or over-the-shoulder composition."
                )
            details = [
                "subject_definitions:\n" + "\n".join(subject_definitions) if subject_definitions else "",
                *tag_notes,
                composition_note,
                dialogue_binding,
                (
                    "质量约束：每个参考标签只对应一个明确角色或素材；严格保持角色身份与服装，"
                    "只生成剧情需要的人物数量，禁止复制人物、融合人脸、重影和额外肢体"
                ),
                *details,
            ]
        else:
            details = [
                (
                    "从输入首帧自然开始；严格保持首帧的人物数量、身份、脸、发型、服装、站位、"
                    "空间关系和背景；禁止新增或复制人物，禁止换脸、融脸、重影、残影和额外肢体；"
                    "人物动作幅度克制、身体结构稳定、逐帧连续"
                ),
                *details,
            ]
        return "。".join(item.strip("。 ") for item in details if item)

    def shot_quality_issues(self, shot: Shot) -> list[str]:
        issues: list[str] = []
        motion = shot.subject_motion or shot.narrative
        action_terms = re.findall(r"说|介绍|要求|询问|回答|起身|转身|走|跑|拿|掏|放|开门|关门|拉开|推开|坐下", motion)
        if shot.duration_seconds > 8:
            issues.append(f"时长 {shot.duration_seconds:g}s：托管 15s 模型按完整时间轴生成，自托管 H3 会按上限续帧")
        if len(action_terms) >= 3 and not shot.visual_beats:
            issues.append(f"检测到 {len(action_terms)} 个动作节点但尚无逐段时间轴，建议重新生成导演提示词")
        has_external_voice = any(event.kind != "character" for event in shot.voice_events)
        if shot.dialogue.strip() and not shot.dialogue_speaker_id and not has_external_voice:
            issues.append("对白未手动绑定发言者，将按剧情动词推断；建议在分镜页确认")
        if not shot.visual_beats:
            issues.append("缺少 visual beats，生成时将使用规则节拍；建议重新生成分镜以获得语义化动作弧线")
        spoken_characters = sum(
            spoken_character_count(self.spoken_dialogue_text(event.text))
            for event in self._effective_voice_events(shot)
        ) or spoken_character_count(self.spoken_dialogue_text(shot.dialogue))
        speech_budget = speech_budget_for_duration(shot.duration_seconds)
        if spoken_characters > speech_budget:
            issues.append(
                f"声音文字 {spoken_characters} 字超过舒适预算 {speech_budget} 字；H3 提交前会自动精简并改用正常语速 TTS"
            )
        return issues

    @staticmethod
    def _motion_phases(motion: str) -> list[str]:
        normalized = re.sub(r"^总时长\s*(?:约|大约)?\s*\d+(?:\.\d+)?\s*秒[。.]?", "", motion.strip())
        marker = re.compile(r"(?:^|[；;。])\s*(?:起势|发展|收束|第\d+段(?:\([^)]*\))?)\s*[：:]")
        matches = list(marker.finditer(normalized))
        if not matches:
            return [normalized] if normalized else []
        phases: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            value = normalized[match.end():end].strip("；;。 ")
            value = re.sub(r"[，,]先给出明确起势与主体站位.*$", "", value).strip("；;。 ")
            if value.startswith(("动作强度逐步提升并出现转折", "动作完成并自然收束")):
                continue
            if value:
                phases.append(value)
        return phases

    def _single_clean_tts_payload(
        self,
        project: Project,
        shot: Shot,
    ) -> tuple[str, str | None, list[VoiceEvent]]:
        """Expose one-source voice events to the existing clean-TTS pipeline.

        H3 stores system narration in ``voice_events`` rather than the legacy
        ``dialogue`` field.  A sequence from the same source can be synthesized
        as one natural track without changing lip ownership.  Mixed-speaker
        shots stay on their authored multi-voice path instead of being assigned
        one incorrect voice.
        """
        if shot.dialogue.strip():
            return (
                self.spoken_dialogue_text(shot.dialogue),
                shot.dialogue_speaker_id,
                self._resolve_voice_events(
                    project,
                    self._effective_voice_events(shot),
                    shot.duration_seconds,
                    shot.duration_seconds,
                ),
            )
        events = self._resolve_voice_events(
            project,
            self._effective_voice_events(shot),
            shot.duration_seconds,
            shot.duration_seconds,
        )
        if not events:
            return "", None, []
        source_keys = {
            (
                event.kind,
                event.speaker_id if event.kind == "character" else (event.speaker_name or event.kind),
            )
            for event in events
        }
        if len(source_keys) != 1:
            return "", None, events
        text = "".join(
            value if value.endswith(("。", "！", "？", ".", "!", "?")) else f"{value}。"
            for value in (self.spoken_dialogue_text(event.text) for event in events)
            if value
        )
        speaker_id = events[0].speaker_id if events[0].kind == "character" else None
        return text, speaker_id, events

    def h3_segment_plan(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
        max_segment_seconds: float | None = None,
    ) -> list[dict[str, object]]:
        """Plan stable sequential I2V segments while keeping one user-facing shot.

        ``max_segment_seconds`` is the per-call render cap of the active
        provider.  The self-hosted ComfyUI stack needs the aggressive dialogue
        splitting below; hosted APIs that render up to 15s per call only split
        when the shot truly exceeds their cap, so a 15s shot is not stacked
        from near-duplicate 5s clips.
        """
        hosted_cap = max_segment_seconds is not None
        cap = float(max_segment_seconds if max_segment_seconds is not None else settings.H3_MAX_SEGMENT_SECONDS)
        clean_tts_dialogue, clean_tts_speaker_id, clean_tts_events = self._single_clean_tts_payload(
            project,
            shot,
        )
        dialogue_turns = self.resolve_dialogue_turns(project, shot)
        longest_spoken_seconds = max(
            (len(str(turn["text"])) / 4.0 + 0.35 for turn in dialogue_turns),
            default=0.0,
        )
        if hosted_cap:
            split_dialogue = shot.duration_seconds > cap + 0.001 or longest_spoken_seconds > cap
        else:
            split_dialogue = len(dialogue_turns) > 1 or longest_spoken_seconds > cap
        if split_dialogue:
            # One visible speaker per H3 call.  This prevents a model from
            # assigning the next line to whichever face happens to be moving.
            visual_turns = [turn.copy() for turn in dialogue_turns]

            def split_text(value: str) -> tuple[str, str]:
                midpoint = len(value) // 2
                boundaries = [match.end() for match in re.finditer(r"[。！？!?；;，,]", value)]
                boundary = min(boundaries, key=lambda item: abs(item - midpoint)) if boundaries else midpoint
                boundary = max(1, min(len(value) - 1, boundary))
                return value[:boundary].strip(), value[boundary:].strip()

            for _ in range(20):
                weights = [max(1.0, len(str(turn["text"])) / 4.0 + 0.35) for turn in visual_turns]
                weight_total = sum(weights)
                durations = [shot.duration_seconds * weight / weight_total for weight in weights]
                longest_index = max(range(len(durations)), key=durations.__getitem__)
                if durations[longest_index] <= cap + 0.001:
                    break
                left, right = split_text(str(visual_turns[longest_index]["text"]))
                if not left or not right:
                    break
                original = visual_turns[longest_index]
                visual_turns[longest_index:longest_index + 1] = [
                    {"speaker_id": original.get("speaker_id"), "text": left},
                    {"speaker_id": original.get("speaker_id"), "text": right},
                ]

            # Splitting at the punctuation nearest the midpoint can leave a
            # punctuation-only fragment (“。”) that TTS cannot voice; fold it
            # back into the previous spoken fragment.
            merged_turns: list[dict[str, str | None]] = []
            for turn in visual_turns:
                text = str(turn["text"])
                if merged_turns and not re.sub(r"[，。！？!?；;、\s]", "", text):
                    merged_turns[-1]["text"] = f"{merged_turns[-1]['text']}{text}"
                else:
                    merged_turns.append(turn.copy())
            if merged_turns:
                visual_turns = merged_turns

            weights = [max(1.0, len(str(turn["text"])) / 4.0 + 0.35) for turn in visual_turns]
            weight_total = sum(weights)
            durations = [shot.duration_seconds * weight / weight_total for weight in weights]
            durations[-1] += shot.duration_seconds - sum(durations)
            result: list[dict[str, object]] = []
            for turn, duration in zip(visual_turns, durations, strict=False):
                segment = shot.model_copy(deep=True)
                segment.h3_prompt_skill_output = ""
                delivery_duration = round(max(0.35, duration), 3)
                generation_duration = max(5.0, min(cap, delivery_duration))
                segment.duration_seconds = generation_duration
                segment.dialogue = str(turn["text"])
                segment.dialogue_speaker_id = turn.get("speaker_id")
                segment.dialogue_turns = [DialogueTurn.model_validate(turn)]
                segment.visual_beats = []
                segment.voice_events = [
                    VoiceEvent(
                        kind="character" if turn.get("speaker_id") else "narration",
                        speaker_id=turn.get("speaker_id"),
                        speaker_name=next(
                            (item.name for item in project.characters if item.id == turn.get("speaker_id")),
                            "旁白",
                        ),
                        text=str(turn["text"]),
                        start_seconds=0.2,
                        end_seconds=min(
                            generation_duration,
                            max(0.7, len(str(turn["text"])) / 4.0 + 0.25),
                        ),
                        lip_sync=bool(turn.get("speaker_id")),
                    )
                ]
                speaker = next((item for item in project.characters if item.id == turn.get("speaker_id")), None)
                speaker_name = speaker.name if speaker else "画外旁白"
                segment.subject_motion = (
                    f"当前短段只表现{speaker_name}说这一句话；{speaker_name}仅做轻微自然口型、眨眼和克制点头，"
                    "其他人物保持安静并闭嘴；人物站位、脸、服装和背景逐帧稳定；禁止走位、转身、挥手、"
                    "多人同时张嘴、角色互换、重影、复制人物和额外肢体"
                )
                segment.narrative = f"当前只聚焦{speaker_name}完成本句对白"
                segment.camera_motion = "固定镜头，禁止切换机位"
                result.append(
                    {
                        "duration": delivery_duration,
                        "generation_duration": generation_duration,
                        "prompt": self.compile_h3_prompt(project, segment, assets),
                        "motion": segment.subject_motion,
                        "has_dialogue": True,
                        "dialogue": str(turn["text"]),
                        "speaker_id": turn.get("speaker_id"),
                    }
                )
            return result

        # A short, single-turn dialogue shot is already inside H3's stable
        # duration window.  Splitting it by action verbs creates a second
        # generation boundary, which is more likely to drift identities,
        # blocking, lip ownership, and continuity than a single guided R2V
        # call.  Keep the authored group/two-shot/OTS composition intact and
        # let the full prompt describe the internal action beats.
        if shot.duration_seconds <= 8.0:
            return [{
                "duration": shot.duration_seconds,
                "generation_duration": max(5.0, shot.duration_seconds),
                "prompt": self.compile_h3_prompt(project, shot, assets),
                "motion": shot.subject_motion,
                "has_dialogue": bool(clean_tts_dialogue),
                "dialogue": clean_tts_dialogue,
                "speaker_id": clean_tts_speaker_id,
                "voice_events": [event.model_dump(mode="json") for event in clean_tts_events],
                "dialogue_start_seconds": clean_tts_events[0].start_seconds if clean_tts_events else shot.dialogue_start_seconds,
            }]

        phases = self._motion_phases(shot.subject_motion or shot.narrative)
        action_count = len(re.findall(r"说|介绍|要求|询问|回答|起身|转身|走|跑|拿|掏|放|开门|关门|拉开|推开|坐下", shot.subject_motion or shot.narrative))
        duration_count = max(1, math.ceil(shot.duration_seconds / max(4.0, cap)))
        action_segments = min(len(phases), max(1, action_count)) if phases else 1
        if hosted_cap and shot.duration_seconds <= cap:
            # A hosted call renders the whole shot continuously; beat-splitting
            # would only stitch near-duplicate clips back together.
            action_segments = 1
        segment_count = max(duration_count, action_segments)
        if not settings.H3_AUTO_SEGMENT_COMPLEX_SHOTS or segment_count == 1:
            return [{
                "duration": shot.duration_seconds,
                "generation_duration": max(5.0, shot.duration_seconds),
                "prompt": self.compile_h3_prompt(project, shot, assets),
                "motion": shot.subject_motion,
                "has_dialogue": bool(clean_tts_dialogue),
                "dialogue": clean_tts_dialogue,
                "speaker_id": clean_tts_speaker_id,
                "voice_events": [event.model_dump(mode="json") for event in clean_tts_events],
                "dialogue_start_seconds": clean_tts_events[0].start_seconds if clean_tts_events else shot.dialogue_start_seconds,
            }]

        if len(phases) < segment_count:
            phases.extend([phases[-1] if phases else shot.subject_motion or shot.narrative] * (segment_count - len(phases)))
        elif len(phases) > segment_count:
            grouped: list[list[str]] = [[] for _ in range(segment_count)]
            for index, phase in enumerate(phases):
                grouped[min(segment_count - 1, index * segment_count // len(phases))].append(phase)
            phases = ["；".join(group) for group in grouped if group]
            segment_count = len(phases)

        speaking_index = 0
        if shot.dialogue.strip():
            speaking_index = next(
                (index for index, phase in enumerate(phases) if re.search(r"说|介绍|要求|询问|回答|嘴唇|开口", phase)),
                0,
            )
        durations = [shot.duration_seconds / segment_count for _ in range(segment_count)]
        if shot.dialogue.strip() and segment_count > 1:
            estimated_speech = min(
                cap,
                max(durations[speaking_index], len(self.spoken_dialogue_text(shot.dialogue)) / 4.0 + 0.5),
            )
            remaining = max(0.75 * (segment_count - 1), shot.duration_seconds - estimated_speech)
            durations = [remaining / (segment_count - 1) for _ in range(segment_count)]
            durations[speaking_index] = shot.duration_seconds - remaining
        durations[-1] += shot.duration_seconds - sum(durations)

        result: list[dict[str, object]] = []
        for index, (phase, duration) in enumerate(zip(phases, durations, strict=False)):
            segment = shot.model_copy(deep=True)
            segment.h3_prompt_skill_output = ""
            delivery_duration = round(duration, 3)
            generation_duration = max(5.0, min(settings.H3_MAX_SEGMENT_SECONDS, delivery_duration))
            segment.duration_seconds = generation_duration
            segment.visual_beats = []
            segment.subject_motion = (
                f"当前短段清晰完成这一项动作：{phase}。动作速度、幅度和冲击力与事件匹配，"
                "人物数量、脸、服装和空间位置连续稳定；本段只使用一种主要运镜，结尾落在可读结果态"
            )
            # Do not carry the whole-shot narrative into every continuation.
            # A preceding dialogue verb (for example "D介绍秦绍辉") causes H3
            # to invent another speaker and even burn pseudo-subtitles into a
            # later silent segment.  Each call must describe only its own beat.
            segment.narrative = f"本短段只表现：{phase}"
            segment.camera_motion = "沿用当前动作节拍指定的一种主要运镜"
            if index != speaking_index:
                segment.dialogue = ""
                segment.dialogue_turns = []
                segment.dialogue_speaker_id = None
                segment.voice_events = []
                visible_names = [
                    item.name for item in project.characters if item.id in shot.character_ids
                ]
                silence_subjects = "、".join(visible_names) if visible_names else "画面内所有人物"
                segment.subject_motion = (
                    f"当前短段只完成这一项无对白动作：{phase}。{silence_subjects}全程嘴唇闭合，"
                    "不说话、不做说话口型、不抬手模拟发言，只做与当前动作直接相关的克制反应；"
                    "画面中不显示对白、字幕、标题、标识、水印或乱码；人物数量、脸、服装和空间位置"
                    "逐帧连续稳定；本段只使用一种主要运镜，禁止角色互换、重影和复制人物；"
                    "结尾保持清晰稳定姿态"
                )
            elif segment.voice_events:
                segment.voice_events = [
                    event.model_copy(
                        update={
                            "start_seconds": min(0.2, max(0.0, generation_duration - 0.5)),
                            "end_seconds": min(
                                generation_duration,
                                max(0.7, len(self.spoken_dialogue_text(segment.dialogue)) / 4.0 + 0.25),
                            ),
                        }
                    )
                    for event in segment.voice_events[:1]
                ]
            result.append(
                {
                    "duration": delivery_duration,
                    "generation_duration": generation_duration,
                    "prompt": self.compile_h3_prompt(project, segment, assets),
                    "motion": phase,
                    "has_dialogue": bool(segment.dialogue.strip()),
                    "dialogue": self.spoken_dialogue_text(segment.dialogue),
                    "speaker_id": segment.dialogue_speaker_id,
                }
            )
        return result

    def plan_shot(self, project_id: str, shot_id: str) -> Shot:
        project = self.require_project(project_id)
        shot = self.require_shot(shot_id)
        self.synchronize_shot_cast(project, shot)
        if not shot.visual_beats:
            shot.visual_beats = self._scale_visual_beats(
                [],
                shot.duration_seconds,
                shot.duration_seconds,
                shot_size=shot.shot_size,
                camera_angle=shot.camera_angle,
                camera_motion=shot.camera_motion or "固定镜头",
                fallback_action=shot.subject_motion or shot.video_prompt_source,
            )
        effective_voice_events = self._effective_voice_events(shot)
        if effective_voice_events and not shot.voice_events:
            shot.voice_events = effective_voice_events
            shot.dialogue_speaker_id = None
            shot.dialogue_turns = []
        assets = self.store.list_assets(project_id)
        character_map = {character.id: character for character in project.characters}
        inherited_refs = [
            asset_id
            for character_id in shot.character_ids
            if character_id in character_map
            for asset_id in character_map[character_id].reference_asset_ids
        ]
        shot.reference_asset_ids = list(dict.fromkeys([*shot.reference_asset_ids, *inherited_refs]))
        shot.resolved_generation_mode = self.resolve_generation_mode(shot, assets)
        if shot.generation_mode == GenerationMode.AUTO:
            shot.ref_image_size = "match"
        shot.render_frames = h3_frames_for_seconds(shot.duration_seconds)
        has_external_voice = any(event.kind != "character" for event in shot.voice_events)
        if shot.dialogue.strip() and not shot.dialogue_turns and not has_external_voice:
            turns = self.resolve_dialogue_turns(project, shot)
            shot.dialogue_turns = [DialogueTurn.model_validate(turn) for turn in turns]
            if len(turns) == 1:
                shot.dialogue_speaker_id = turns[0]["speaker_id"]
            elif len(turns) > 1:
                shot.dialogue_speaker_id = None
        if not shot.video_prompt_source:
            shot.video_prompt_source = shot.video_prompt or self.compile_base_video_prompt(
                project,
                shot.subject_motion,
                shot.narrative,
            )
        shot.video_prompt = self.compile_h3_prompt(project, shot, assets)
        return self.store.save_shot(shot)

    def plan_all_shots(self, project_id: str) -> list[Shot]:
        return [self.plan_shot(project_id, shot.id) for shot in self.store.list_shots(project_id)]

    @staticmethod
    def _shot_content_signature(shot: Shot) -> tuple[object, ...]:
        return (
            shot.title,
            shot.narrative,
            shot.dialogue,
            shot.dialogue_speaker_id,
            tuple((turn.speaker_id, turn.text) for turn in shot.dialogue_turns),
            shot.duration_seconds,
            shot.scene_description,
            tuple(shot.scene_profile_ids),
            shot.scene_profile_id,
            shot.use_scene_profile,
            shot.continuity_mode,
            shot.continuity_source_shot_id,
            shot.first_frame_completeness,
            tuple(shot.character_ids),
            shot.shot_size,
            shot.camera_angle,
            shot.lens,
            shot.camera_motion,
            shot.subject_motion,
            shot.transition,
            shot.audio_design,
            tuple(
                (
                    beat.start_seconds,
                    beat.end_seconds,
                    beat.purpose,
                    beat.subject_action,
                    beat.environment_action,
                    beat.shot_size,
                    beat.camera_angle,
                    beat.camera_motion,
                    beat.sound_cue,
                )
                for beat in shot.visual_beats
            ),
            tuple(
                (
                    event.kind,
                    event.speaker_id,
                    event.speaker_name,
                    event.text,
                    event.start_seconds,
                    event.end_seconds,
                    event.lip_sync,
                )
                for event in shot.voice_events
            ),
            shot.text_policy,
            shot.visual_prompt,
            tuple(shot.reference_asset_ids),
            None if shot.keyframe_reference_asset_ids is None else tuple(shot.keyframe_reference_asset_ids),
            None if shot.video_reference_asset_ids is None else tuple(shot.video_reference_asset_ids),
        )

    def update_shot(self, project_id: str, incoming: Shot) -> Shot:
        """Save an edited shot and immediately refresh all dependent prompts.

        Direct edits to a prompt field are preserved. When storyboard content
        changes, generated prompt variants are invalidated and rebuilt from the
        new source fields so the production page never shows stale content.
        """
        project = self.require_project(project_id)
        existing = self.require_shot(incoming.id)
        if incoming.project_id != project_id or existing.project_id != project_id:
            raise ValueError("Shot project mismatch")
        if incoming.version != existing.version:
            raise ShotVersionConflictError("本镜已有新的保存结果，已阻止旧页面覆盖。请刷新后再编辑保存。")
        known_scene_ids = {profile.id for profile in project.scene_profiles}
        incoming.scene_profile_ids = [
            profile_id
            for profile_id in dict.fromkeys([
                *incoming.scene_profile_ids,
                *([incoming.scene_profile_id] if incoming.scene_profile_id else []),
            ])
            if profile_id in known_scene_ids
        ][:2]
        incoming.scene_profile_id = incoming.scene_profile_ids[0] if incoming.scene_profile_ids else None
        if not incoming.scene_profile_ids:
            incoming.use_scene_profile = False
        asset_types = {
            asset.id: asset.type for asset in self.store.list_assets(project_id)
        }
        if incoming.keyframe_reference_asset_ids is not None:
            incoming.keyframe_reference_asset_ids = [
                asset_id
                for asset_id in dict.fromkeys(incoming.keyframe_reference_asset_ids)
                if asset_types.get(asset_id) == AssetType.IMAGE
            ][:20]
        if incoming.video_reference_asset_ids is not None:
            incoming.video_reference_asset_ids = [
                asset_id
                for asset_id in dict.fromkeys(incoming.video_reference_asset_ids)
                if asset_types.get(asset_id) in {
                    AssetType.IMAGE,
                    AssetType.VIDEO,
                    AssetType.AUDIO,
                }
            ][:20]
        # These are server-owned generation metadata. Preserve them even when
        # an older client omits them; explicit prompt edits are handled below.
        incoming.h3_prompt_skill_output = existing.h3_prompt_skill_output
        incoming.h3_prompt_skill_id = existing.h3_prompt_skill_id
        incoming.h3_prompt_skill_version = existing.h3_prompt_skill_version
        incoming.h3_director_version = existing.h3_director_version
        incoming.h3_prompt_source_revision = existing.h3_prompt_source_revision
        # Render selection is server-owned. A stale editor save must never
        # unlock an adopted card or point continuity at a different tail frame.
        incoming.selected_video_job_id = existing.selected_video_job_id
        incoming.video_path = existing.video_path
        incoming.last_frame_asset_id = existing.last_frame_asset_id
        incoming.video_status = existing.video_status
        self.synchronize_shot_cast(project, incoming)
        content_changed = self._shot_content_signature(existing) != self._shot_content_signature(incoming)
        seedance_reference_mode_changed = (
            existing.seedance_reference_mode != incoming.seedance_reference_mode
        )
        keyframe_authored = incoming.keyframe_prompt != existing.keyframe_prompt
        h3_authored = incoming.video_prompt != existing.video_prompt
        seedance_authored = incoming.seedance_prompt != existing.seedance_prompt
        if content_changed:
            incoming.content_revision = existing.content_revision + 1
            character_ids = self.keyframe_character_ids(project, incoming, incoming.scene_description)
            scene_profile = self.scene_profile_for_shot(project, incoming)
            assets = self.store.list_assets(project_id)
            if not keyframe_authored:
                incoming.keyframe_prompt = self.compile_keyframe_prompt(
                    project,
                    incoming.scene_description or incoming.narrative,
                    character_ids,
                    incoming.character_appearance_ids,
                    shot_size=incoming.shot_size,
                    camera_angle=incoming.camera_angle,
                    lens=incoming.lens,
                    text_policy=incoming.text_policy,
                    scene_profile=scene_profile,
                )
                incoming.keyframe_prompt = self.apply_fixed_prop_anchor(
                    incoming.keyframe_prompt,
                    self.keyframe_reference_assets(
                        project,
                        incoming,
                        assets,
                        character_ids,
                    ),
                )
            incoming.keyframe_prompt_source_revision = incoming.content_revision
            if not seedance_authored:
                incoming.seedance_prompt = self.compile_seedance_prompt(project, incoming, assets)
                incoming.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
            incoming.seedance_prompt_source_revision = incoming.content_revision
            if not h3_authored and not existing.h3_prompt_skill_output.strip():
                incoming.video_prompt_source = self.compile_base_video_prompt(
                    project,
                    incoming.subject_motion,
                    incoming.narrative,
                )
                incoming.video_prompt = self.compile_h3_prompt(project, incoming, assets)
                incoming.h3_prompt_source_revision = incoming.content_revision
            incoming.approval_status = ApprovalStatus.DRAFT
        elif seedance_reference_mode_changed and not seedance_authored:
            assets = self.store.list_assets(project_id)
            incoming.seedance_prompt = self.compile_seedance_prompt(project, incoming, assets)
            incoming.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
            incoming.seedance_prompt_source_revision = incoming.content_revision
        if h3_authored:
            incoming.video_prompt_source = incoming.video_prompt
            incoming.h3_prompt_skill_output = incoming.video_prompt
            incoming.h3_prompt_source_revision = incoming.content_revision
            incoming.h3_director_version = H3_DIRECTOR_VERSION
        incoming.version = max(existing.version, incoming.version) + 1
        saved = self.store.save_shot(incoming)
        self.invalidate_storyboard(project_id)
        return saved

    def ensure_keyframe_idle(self, shot_id: str) -> None:
        if shot_id in self._active_keyframes:
            raise KeyframeBusyError("本镜首帧正在生成，请等本镜完成后再修改或重做；其他镜头可继续操作。")

    def update_keyframe_prompt(
        self,
        project_id: str,
        shot_id: str,
        keyframe_prompt: str,
        revision_suggestion: str | None = None,
        revision_mode: str | None = None,
        reference_asset_ids: list[str] | None = None,
        reference_selection_supplied: bool = False,
    ) -> Shot:
        """Patch only keyframe fields, preserving concurrently generated video prompts."""
        shot = self.require_shot(shot_id)
        if shot.project_id != project_id:
            raise KeyError("Shot not found")
        self.ensure_keyframe_idle(shot_id)
        if not keyframe_prompt.strip():
            raise ValueError("请填写首帧 Prompt")
        if revision_mode is not None and revision_mode not in {"fresh", "iterate"}:
            raise ValueError(f"Unsupported keyframe revision mode: {revision_mode}")
        shot.keyframe_prompt = keyframe_prompt.strip()
        shot.keyframe_prompt_source_revision = shot.content_revision
        if revision_suggestion is not None:
            shot.keyframe_revision_suggestion_draft = revision_suggestion.strip()
        if revision_mode is not None:
            shot.keyframe_revision_mode = revision_mode
        if reference_selection_supplied:
            if reference_asset_ids is None:
                shot.keyframe_reference_asset_ids = None
            else:
                image_ids = {
                    asset.id
                    for asset in self.store.list_assets(project_id)
                    if asset.type == AssetType.IMAGE
                }
                unknown = set(reference_asset_ids) - image_ids
                if unknown:
                    raise ValueError("所选首帧参考图已不存在，请刷新页面后重试")
                shot.keyframe_reference_asset_ids = list(
                    dict.fromkeys(reference_asset_ids)
                )[:20]
        project = self.require_project(project_id)
        assets = self.store.list_assets(project_id)
        character_ids = self.keyframe_character_ids(project, shot, shot.keyframe_prompt)
        shot.keyframe_prompt = self.apply_fixed_prop_anchor(
            shot.keyframe_prompt,
            self.keyframe_reference_assets(
                project,
                shot,
                assets,
                character_ids,
            ),
        )
        shot.version += 1
        saved = self.store.save_shot(shot)
        self.invalidate_storyboard(project_id)
        return saved

    async def revise_shot_with_ai(
        self,
        project_id: str,
        shot_id: str,
        user_suggestions: str,
        prompt_targets: list[str] | None = None,
        h3_skill_id: str = "h3-prompt-writing",
    ) -> Shot:
        """Redo one shot while locking its identity and ordinal, not its duration."""
        project = self.require_project(project_id)
        shot = self.require_shot(shot_id)
        suggestions = user_suggestions.strip()
        if not suggestions:
            raise ValueError("请填写本镜修改建议")
        characters = [
            {"id": item.id, "name": item.name, "description": item.description, "wardrobe": item.wardrobe}
            for item in project.characters
        ]
        generator = create_llm_generator(settings.LLM_PROVIDER)
        payload = await generator.generate_json(
            "你是影视分镜导演。只重做指定单镜；镜头身份和镜号是硬约束，时长必须按重做后的有效内容重新判断。",
            f"""重做下面一个分镜。保持 shot_id、ordinal 原值；人物、对白、场景、动作、构图和 duration_seconds 都可按用户建议重新设计。

项目剧情：{project.brief.story}
统一风格：{self.project_style_text(project)}
可用角色：{json.dumps(characters, ensure_ascii=False)}
原镜头：{json.dumps(shot.model_dump(mode='json'), ensure_ascii=False)}
用户建议（最高优先级）：{suggestions}

先根据重做后的信息量选择 2–15 秒的自然镜头时长，不得为了沿用原来的 {shot.duration_seconds:g} 秒而加入静止等待、重复表情或无新信息留白；也不得用过短时长挤掉必要台词。visual_beats 从 0 秒连续覆盖所选 duration_seconds，每 2–4 秒有新的可见变化；每段只含一个主动作和一种主要运镜。系统播报、旁白、画外音须作为非 character 的 voice_event 且 lip_sync=false。voice_events 是唯一声音时序来源，dialogue 留空；整镜所有可朗读文字按正常语速每秒约 3.4 个有效字符编排，声音事件不重叠，只保留剧情确实需要的短反应和环境声。默认 text_policy=post_overlay。

严格返回紧凑 JSON（不要输出 visual_prompt、keyframe_prompt、motion_prompt 等系统本地编译字段）：
{{"duration_seconds":8,"event":"本镜完整事件","opening_state":"0 秒静态起始状态","dialogue":"","dialogue_speaker":"","character_names":[],"shot_size":"","camera_angle":"","lens":"","camera_motion":"","visual_beats":[{{"start_seconds":0,"end_seconds":3,"purpose":"","subject_action":"","environment_action":"","shot_size":"","camera_angle":"","camera_motion":"","sound_cue":""}}],"voice_events":[{{"kind":"character/system_vo/narration/offscreen","speaker_name":"","text":"","start_seconds":0.5,"end_seconds":3.5,"lip_sync":false}}],"text_policy":"post_overlay"}}""",
        )
        original_duration = shot.duration_seconds
        try:
            proposed_duration = float(payload.get("duration_seconds", original_duration))
        except (TypeError, ValueError):
            proposed_duration = original_duration
        proposed_duration = max(0.25, min(15.0, proposed_duration))
        authored_duration = proposed_duration
        raw_voice_payloads = payload.get("voice_events")
        if isinstance(raw_voice_payloads, list):
            spoken_size = sum(
                spoken_character_count(str(item.get("text") or ""))
                for item in raw_voice_payloads
                if isinstance(item, dict)
            )
            if spoken_size:
                proposed_duration = min(15.0, max(proposed_duration, spoken_size / 3.4 + 0.01))
        shot.duration_seconds = round(proposed_duration, 2)
        names = [str(value) for value in payload.get("character_names", [])]
        event_text = str(payload.get("event") or payload.get("narrative") or "").strip()
        opening_state = str(
            payload.get("opening_state")
            or payload.get("scene_description")
            or payload.get("keyframe_prompt")
            or event_text
        ).strip()
        speaker_id = self._resolve_dialogue_speaker_id(
            project,
            str(payload.get("dialogue_speaker_name") or payload.get("dialogue_speaker") or ""),
            event_text,
            str(payload.get("dialogue") or ""),
        )
        speaker = next((character for character in project.characters if character.id == speaker_id), None)
        for field in (
            "title",
            "narrative",
            "dialogue",
            "scene_description",
            "shot_size",
            "camera_angle",
            "lens",
            "camera_motion",
            "subject_motion",
            "transition",
            "audio_design",
            "visual_prompt",
        ):
            value = str(payload.get(field) or "").strip()
            if value:
                setattr(shot, field, value)
        if event_text and not str(payload.get("narrative") or "").strip():
            shot.narrative = event_text
        if opening_state and not str(payload.get("scene_description") or "").strip():
            shot.scene_description = opening_state
        shot.character_ids = self.resolve_character_ids(
            project,
            shot.title,
            shot.narrative,
            shot.scene_description,
            shot.subject_motion,
            shot.visual_prompt,
            explicit_names=names,
            speaker_ids=[speaker_id],
        )
        if not str(payload.get("visual_prompt") or "").strip():
            scene_profile = self.scene_profile_for_shot(project, shot)
            shot.visual_prompt = self.compile_visual_prompt(
                project,
                shot.scene_description or shot.narrative,
                shot.character_ids,
                shot.character_appearance_ids,
                scene_profile=scene_profile,
            )
        shot.dialogue_speaker_id = speaker.id if speaker else None
        shot.dialogue_turns = (
            [DialogueTurn(speaker_id=speaker.id, text=shot.dialogue)]
            if speaker and shot.dialogue.strip()
            else []
        )
        raw_beats = payload.get("visual_beats")
        parsed_beats: list[VisualBeat] = []
        if isinstance(raw_beats, list):
            for item in raw_beats:
                try:
                    parsed_beats.append(VisualBeat.model_validate(item))
                except Exception:
                    continue
        shot.visual_beats = self._scale_visual_beats(
            parsed_beats,
            authored_duration,
            shot.duration_seconds,
            shot_size=shot.shot_size,
            camera_angle=shot.camera_angle,
            camera_motion=shot.camera_motion or "固定镜头",
            fallback_action=shot.subject_motion,
        )
        if not str(payload.get("subject_motion") or "").strip() and parsed_beats:
            shot.subject_motion = "；".join(
                beat.subject_action for beat in shot.visual_beats if beat.subject_action
            )
        shot.subject_motion = self._normalize_motion_duration(
            shot.subject_motion or shot.narrative,
            shot.duration_seconds,
        )
        raw_voice_events = payload.get("voice_events")
        parsed_voice_events: list[VoiceEvent] = []
        if isinstance(raw_voice_events, list):
            for item in raw_voice_events:
                try:
                    parsed_voice_events.append(VoiceEvent.model_validate(item))
                except Exception:
                    continue
        if not parsed_voice_events and shot.dialogue.strip():
            speaker_label = str(payload.get("dialogue_speaker_name") or payload.get("dialogue_speaker") or "")
            kind = self._voice_kind_from_label(speaker_label)
            parsed_voice_events = [
                VoiceEvent(
                    kind=kind,
                    speaker_id=speaker_id if kind == "character" else None,
                    speaker_name=speaker_label,
                    text=shot.dialogue,
                    start_seconds=0.35,
                    end_seconds=min(shot.duration_seconds, max(1.5, len(shot.dialogue) / 4.2)),
                    lip_sync=kind == "character" and bool(speaker_id),
                )
            ]
        shot.voice_events = self._resolve_voice_events(
            project,
            parsed_voice_events,
            shot.duration_seconds,
            shot.duration_seconds,
        )
        text_policy = str(payload.get("text_policy") or "post_overlay")
        shot.text_policy = text_policy if text_policy in {"none", "post_overlay", "reference_locked"} else "post_overlay"
        # The update service compares against persisted data. Preserve these
        # hard constraints even if the model returned other values.
        shot.id = shot_id
        shot.project_id = project_id
        saved = self.update_shot(project_id, shot)
        requested_targets = ["seedance"] if prompt_targets is None else prompt_targets
        targets = [target for target in dict.fromkeys(requested_targets) if target in {"h3", "seedance"}]
        if "seedance" in targets:
            self.generate_seedance_prompts(project_id, [shot_id])
        if "h3" in targets:
            await self.generate_h3_prompts(project_id, [shot_id], h3_skill_id, suggestions)
        return self.require_shot(shot_id)

    async def preview_shot_split(
        self,
        project_id: str,
        shot_id: str,
        user_suggestions: str = "",
        segment_count: int | None = None,
    ) -> ShotSplitPreview:
        """Generate an editable proposal without mutating the saved storyboard."""
        if segment_count is not None and segment_count not in {2, 3, 4}:
            raise ValueError("拆分数量只能是 2、3 或 4 镜")
        project = self.require_project(project_id)
        source = self.require_shot(shot_id)
        if source.project_id != project_id:
            raise KeyError("Shot not found")
        source_version = source.version
        shots = self.store.list_shots(project_id)
        previous = next((item for item in shots if item.ordinal == source.ordinal - 1), None)
        following = next((item for item in shots if item.ordinal == source.ordinal + 1), None)
        characters = [
            {
                "name": item.name,
                "description": item.description,
                "wardrobe": item.wardrobe,
            }
            for item in project.characters
        ]
        scene_profiles = [
            {
                "name": profile.name,
                "description": profile.description,
                "continuity_notes": profile.continuity_notes,
            }
            for profile in project.scene_profiles
        ]
        count_instruction = (
            f"严格拆成 {segment_count} 镜"
            if segment_count is not None
            else "根据动作、场景或对白节拍自行选择拆成 2–4 镜"
        )
        suggestions = user_suggestions.strip()
        generator = create_llm_generator(settings.LLM_PROVIDER)
        payload = await generator.generate_json(
            "你是影视分镜导演。把一个过载镜头拆成连续、可独立生产的小镜头，只返回结构化 JSON。",
            f"""把下面的第 {source.ordinal} 镜拆分为多个连续镜头。{count_instruction}。这一步只生成可编辑预览，不要生成大段提示词。

项目剧情：{self._compact_prompt_text(project.brief.story, 3000)}
统一风格：{self.project_rendering_style_text(project)}
可用角色：{json.dumps(characters, ensure_ascii=False)}
可用场景档案：{json.dumps(scene_profiles, ensure_ascii=False)}
上一镜：{json.dumps(previous.model_dump(mode='json') if previous else {}, ensure_ascii=False)}
待拆原镜：{json.dumps(source.model_dump(mode='json'), ensure_ascii=False)}
下一镜：{json.dumps(following.model_dump(mode='json') if following else {}, ensure_ascii=False)}
用户本次要求（高优先级）：{suggestions or '保持原镜剧情、角色和风格，只优化为可生产的小镜头'}

要求：
1. 不丢失原镜事件因果，不重复上下镜已经完成的内容。
2. 每镜 0.25–15 秒，每镜只承担一个主动作或一个明确叙事转折；默认尽量保持原镜总时长。
3. scene_description 只写 0.00 秒首帧静态状态，narrative 写该镜完整事件。
4. visual_beats 从 0 秒无缝覆盖到该镜结尾；voice_events 是唯一声音时序来源。
5. scene_profile_name 优先逐字使用已有场景档案名称；确有新场景时才填新名称和档案说明。
6. character_names 只列画面中真正出现的已有角色。

严格返回：
{{"rationale":"拆分依据","segments":[{{"title":"","duration_seconds":4,"narrative":"","dialogue":"","scene_description":"","scene_profile_name":"","scene_profile_description":"","scene_continuity_notes":"","character_names":[],"shot_size":"","camera_angle":"","lens":"","camera_motion":"","subject_motion":"","transition":"硬切","audio_design":"","visual_beats":[{{"start_seconds":0,"end_seconds":4,"purpose":"","subject_action":"","environment_action":"","shot_size":"","camera_angle":"","camera_motion":"","sound_cue":""}}],"voice_events":[],"text_policy":"post_overlay"}}]}}""",
        )
        raw_segments = payload.get("segments") or payload.get("shots")
        if not isinstance(raw_segments, list) or not 2 <= len(raw_segments) <= 4:
            raise ValueError(f"AI 拆分结果需要包含 2–4 个镜头{_raw_response_hint(generator)}")
        if segment_count is not None and len(raw_segments) != segment_count:
            raise ValueError(f"AI 未按要求返回 {segment_count} 个镜头，请重试")

        source_profiles = self.scene_profiles_for_shot(project, source)
        segments: list[ShotSplitSegment] = []
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise ValueError(f"拆分预览的第 {index + 1} 镜格式不正确")
            try:
                duration = max(0.25, min(15.0, float(raw.get("duration_seconds") or 4.0)))
            except (TypeError, ValueError):
                duration = 4.0
            narrative = str(raw.get("narrative") or raw.get("event") or "").strip()
            opening = str(
                raw.get("scene_description")
                or raw.get("opening_state")
                or narrative
            ).strip()
            if not narrative:
                raise ValueError(f"拆分预览的第 {index + 1} 镜缺少剧情内容")
            fallback_profile = None
            if source_profiles:
                fallback_profile = source_profiles[-1] if len(source_profiles) > 1 and index > 0 else source_profiles[0]
            beats: list[VisualBeat] = []
            for beat in raw.get("visual_beats") or []:
                try:
                    beats.append(VisualBeat.model_validate(beat))
                except Exception:
                    continue
            beats = self._scale_visual_beats(
                beats,
                max((beat.end_seconds for beat in beats), default=duration),
                duration,
                shot_size=str(raw.get("shot_size") or source.shot_size or "中景").strip(),
                camera_angle=str(raw.get("camera_angle") or source.camera_angle or "平视").strip(),
                camera_motion=str(raw.get("camera_motion") or source.camera_motion or "固定镜头").strip(),
                fallback_action=str(raw.get("subject_motion") or narrative).strip(),
            )
            voices: list[VoiceEvent] = []
            for event in raw.get("voice_events") or []:
                try:
                    voices.append(VoiceEvent.model_validate(event))
                except Exception:
                    continue
            voices = self._resolve_voice_events(
                project,
                voices,
                max((event.end_seconds for event in voices), default=duration),
                duration,
            )
            policy = str(raw.get("text_policy") or source.text_policy)
            if policy not in {"none", "post_overlay", "reference_locked"}:
                policy = "post_overlay"
            raw_character_names = raw.get("character_names") or []
            if isinstance(raw_character_names, str):
                raw_character_names = re.split(r"[,，、;/；]", raw_character_names)
            if not isinstance(raw_character_names, list):
                raw_character_names = []
            segments.append(ShotSplitSegment(
                title=str(raw.get("title") or f"{source.title or f'镜头 {source.ordinal}'} · {index + 1}").strip(),
                duration_seconds=duration,
                narrative=narrative,
                dialogue=str(raw.get("dialogue") or "").strip(),
                scene_description=opening,
                scene_profile_name=str(raw.get("scene_profile_name") or (fallback_profile.name if fallback_profile else "")).strip(),
                scene_profile_description=str(raw.get("scene_profile_description") or "").strip(),
                scene_continuity_notes=str(raw.get("scene_continuity_notes") or "").strip(),
                character_names=[str(name).strip() for name in raw_character_names if str(name).strip()],
                shot_size=str(raw.get("shot_size") or source.shot_size or "中景").strip(),
                camera_angle=str(raw.get("camera_angle") or source.camera_angle or "平视").strip(),
                lens=str(raw.get("lens") or source.lens or "标准镜头").strip(),
                camera_motion=str(raw.get("camera_motion") or source.camera_motion or "固定镜头").strip(),
                subject_motion=str(raw.get("subject_motion") or narrative).strip(),
                transition=str(raw.get("transition") or (source.transition if index == len(raw_segments) - 1 else "硬切")).strip(),
                audio_design=str(raw.get("audio_design") or source.audio_design).strip(),
                visual_beats=beats,
                voice_events=voices,
                text_policy=policy,
            ))

        latest = self.require_shot(shot_id)
        if latest.project_id != project_id or latest.version != source_version:
            raise ShotVersionConflictError("本镜在拆分预览生成期间已更新，请重新生成预览")
        return ShotSplitPreview(
            source_shot_id=shot_id,
            source_shot_version=source_version,
            source_ordinal=source.ordinal,
            original_duration_seconds=source.duration_seconds,
            proposed_duration_seconds=round(sum(segment.duration_seconds for segment in segments), 2),
            rationale=str(payload.get("rationale") or "").strip(),
            segments=segments,
        )

    async def confirm_shot_split(
        self,
        project_id: str,
        shot_id: str,
        preview: ShotSplitPreview,
        prompt_targets: list[str] | None = None,
        h3_skill_id: str = "h3-prompt-writing",
    ) -> list[Shot]:
        """Compile edited split segments and atomically replace the source shot."""
        project = self.require_project(project_id)
        source = self.require_shot(shot_id)
        if source.project_id != project_id:
            raise KeyError("Shot not found")
        if preview.source_shot_id != shot_id:
            raise ValueError("拆分预览与当前镜头不匹配")
        if preview.source_ordinal != source.ordinal or abs(preview.original_duration_seconds - source.duration_seconds) > 0.001:
            raise ShotVersionConflictError("镜号或原镜时长已变化，请重新生成拆分预览")
        if preview.source_shot_version != source.version:
            raise ShotVersionConflictError("本镜已有新的保存结果，请重新生成拆分预览")
        active_statuses = {
            JobStatus.QUEUED,
            JobStatus.SUBMITTING,
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
        }
        if any(job.shot_id == shot_id and job.status in active_statuses for job in self.store.list_jobs(project_id)):
            raise ShotSplitBusyError("本镜仍有视频生成任务在进行，请等待完成或取消后再确认拆分")
        try:
            self.ensure_keyframe_idle(shot_id)
        except KeyframeBusyError as exc:
            raise ShotSplitBusyError(str(exc)) from exc

        assets = self.store.list_assets(project_id)
        source_profiles = self.scene_profiles_for_shot(project, source)
        excluded_generated_assets = {source.keyframe_asset_id, source.last_frame_asset_id}
        inherited_references = [
            asset_id
            for asset_id in source.reference_asset_ids
            if asset_id and asset_id not in excluded_generated_assets
        ]
        replacements: list[Shot] = []
        profile_by_name = {profile.name.strip().casefold(): profile for profile in project.scene_profiles}
        used_profile_ids: dict[str, list[str]] = {}
        for index, segment in enumerate(preview.segments):
            narrative = segment.narrative.strip()
            if not narrative:
                raise ValueError(f"第 {index + 1} 个拆分镜头的剧情内容不能为空")
            duration = max(0.25, min(15.0, float(segment.duration_seconds)))
            requested_profile_name = segment.scene_profile_name.strip()
            scene_profile = profile_by_name.get(requested_profile_name.casefold()) if requested_profile_name else None
            if scene_profile is None and not requested_profile_name and source_profiles:
                scene_profile = source_profiles[-1] if len(source_profiles) > 1 and index > 0 else source_profiles[0]
            if scene_profile is None and requested_profile_name:
                scene_profile = SceneProfile(
                    name=requested_profile_name,
                    description=segment.scene_profile_description.strip() or segment.scene_description.strip() or narrative,
                    continuity_notes=segment.scene_continuity_notes.strip() or "保持空间布局、固定陈设、材质、主色与光源方向",
                )
                project.scene_profiles.append(scene_profile)
                profile_by_name[requested_profile_name.casefold()] = scene_profile

            source_beat_duration = max((beat.end_seconds for beat in segment.visual_beats), default=duration)
            source_voice_duration = max((event.end_seconds for event in segment.voice_events), default=duration)
            voices = self._resolve_voice_events(
                project,
                segment.voice_events,
                source_voice_duration,
                duration,
            )
            speaker_ids = [event.speaker_id for event in voices if event.kind == "character" and event.speaker_id]
            character_ids = self.resolve_character_ids(
                project,
                segment.title,
                narrative,
                segment.scene_description,
                segment.subject_motion,
                explicit_names=segment.character_names,
                speaker_ids=speaker_ids,
            )
            character_appearance_ids = {
                character_id: appearance_id
                for character_id, appearance_id in source.character_appearance_ids.items()
                if character_id in character_ids
            }
            previous_replacement = replacements[-1] if replacements else None
            same_profile_as_previous = bool(
                previous_replacement
                and scene_profile
                and previous_replacement.scene_profile_id == scene_profile.id
            )
            shot = Shot(
                project_id=project_id,
                ordinal=source.ordinal + index,
                title=segment.title.strip() or f"{source.title or f'镜头 {source.ordinal}'} · {index + 1}",
                narrative=narrative,
                dialogue=segment.dialogue.strip(),
                duration_seconds=duration,
                scene_description=segment.scene_description.strip() or narrative,
                scene_profile_ids=[scene_profile.id] if scene_profile else [],
                scene_profile_id=scene_profile.id if scene_profile else None,
                use_scene_profile=bool(scene_profile),
                continuity_mode=(
                    source.continuity_mode
                    if index == 0
                    else ShotContinuityMode.SAME_SCENE
                    if same_profile_as_previous
                    else ShotContinuityMode.INDEPENDENT
                ),
                continuity_source_shot_id=source.continuity_source_shot_id if index == 0 else None,
                seedance_reference_mode=SeedanceReferenceMode.AUTO,
                first_frame_completeness=FirstFrameCompleteness.UNKNOWN,
                character_ids=character_ids,
                character_appearance_ids=character_appearance_ids,
                shot_size=segment.shot_size.strip() or source.shot_size,
                camera_angle=segment.camera_angle.strip() or source.camera_angle,
                lens=segment.lens.strip() or source.lens,
                camera_motion=segment.camera_motion.strip() or source.camera_motion,
                subject_motion=segment.subject_motion.strip() or narrative,
                transition=segment.transition.strip() or (source.transition if index == len(preview.segments) - 1 else "硬切"),
                audio_design=segment.audio_design.strip() or source.audio_design,
                visual_beats=self._scale_visual_beats(
                    segment.visual_beats,
                    source_beat_duration,
                    duration,
                    shot_size=segment.shot_size.strip() or source.shot_size,
                    camera_angle=segment.camera_angle.strip() or source.camera_angle,
                    camera_motion=segment.camera_motion.strip() or source.camera_motion,
                    fallback_action=segment.subject_motion.strip() or narrative,
                ),
                voice_events=voices,
                text_policy=segment.text_policy,
                negative_prompt=source.negative_prompt,
                generation_mode=source.generation_mode,
                ref_image_size=source.ref_image_size,
                reference_asset_ids=list(inherited_references),
                h3_width=source.h3_width,
                h3_height=source.h3_height,
                h3_turbo=source.h3_turbo,
                h3_model_profile=source.h3_model_profile,
                h3_text_encoder_profile=source.h3_text_encoder_profile,
                h3_steps=source.h3_steps,
                h3_scheduler=source.h3_scheduler,
                h3_denoise=source.h3_denoise,
                h3_lora_strength=source.h3_lora_strength,
                h3_low_vram=source.h3_low_vram,
                h3_shift_video=source.h3_shift_video,
                h3_shift_audio=source.h3_shift_audio,
                render_frames=h3_frames_for_seconds(duration),
            )
            shot.dialogue_speaker_id = speaker_ids[0] if len(set(speaker_ids)) == 1 else None
            shot.dialogue_turns = [
                DialogueTurn(speaker_id=event.speaker_id, text=event.text)
                for event in voices
                if event.kind == "character" and event.text.strip()
            ]
            self.synchronize_shot_cast(project, shot)
            shot.visual_prompt = self.compile_visual_prompt(
                project,
                shot.scene_description or shot.narrative,
                shot.character_ids,
                shot.character_appearance_ids,
                scene_profile=scene_profile,
            )
            shot.keyframe_prompt = self.compile_keyframe_prompt(
                project,
                shot.scene_description or shot.narrative,
                shot.character_ids,
                shot.character_appearance_ids,
                shot_size=shot.shot_size,
                camera_angle=shot.camera_angle,
                lens=shot.lens,
                text_policy=shot.text_policy,
                scene_profile=scene_profile,
            )
            shot.keyframe_prompt = self.apply_fixed_prop_anchor(
                shot.keyframe_prompt,
                self.keyframe_reference_assets(project, shot, assets, shot.character_ids),
            )
            shot.keyframe_prompt_source_revision = shot.content_revision
            shot.video_prompt_source = self.compile_base_video_prompt(project, shot.subject_motion, shot.narrative)
            shot.video_prompt = self.compile_h3_prompt(project, shot, assets)
            shot.h3_director_version = H3_DIRECTOR_VERSION
            shot.h3_prompt_source_revision = shot.content_revision
            shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
            shot.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
            shot.seedance_prompt_source_revision = shot.content_revision
            replacements.append(shot)
            if scene_profile:
                used_profile_ids.setdefault(scene_profile.id, []).append(shot.id)

        try:
            saved = self.store.replace_shot_with_many(
                project_id,
                shot_id,
                replacements,
                expected_version=preview.source_shot_version,
            )
        except ValueError as exc:
            if "新的保存结果" in str(exc):
                raise ShotVersionConflictError(str(exc)) from exc
            raise
        for profile in project.scene_profiles:
            profile.source_shot_ids = [item for item in profile.source_shot_ids if item != shot_id]
            profile.source_shot_ids = list(dict.fromkeys([
                *profile.source_shot_ids,
                *used_profile_ids.get(profile.id, []),
            ]))
        self.store.save_project(project)
        self.invalidate_storyboard(project_id)

        targets = [target for target in dict.fromkeys(prompt_targets or []) if target in {"h3", "seedance"}]
        if "h3" in targets:
            await self.generate_h3_prompts(
                project_id,
                [shot.id for shot in saved],
                h3_skill_id,
                f"从原镜头 {source.ordinal} 拆分后保持剧情连续",
            )
        return [self.require_shot(shot.id) for shot in saved]

    async def insert_shot_with_ai(
        self,
        project_id: str,
        after_shot_id: str | None,
        user_suggestions: str,
        prompt_targets: list[str] | None = None,
        h3_skill_id: str = "h3-prompt-writing",
    ) -> Shot:
        """Generate and transactionally insert one new shot without replacing the board."""
        project = self.require_project(project_id)
        suggestions = user_suggestions.strip()
        if not suggestions:
            raise ValueError("请填写新增镜头的剧情和画面要求")
        existing = self.store.list_shots(project_id)
        if after_shot_id is None:
            anchor = existing[-1] if existing else None
            next_shot = None
            intended_ordinal = len(existing) + 1
        else:
            anchor = next((item for item in existing if item.id == after_shot_id), None)
            if anchor is None:
                raise KeyError(f"Shot not found: {after_shot_id}")
            next_shot = next((item for item in existing if item.ordinal == anchor.ordinal + 1), None)
            intended_ordinal = anchor.ordinal + 1
        characters = [
            {
                "name": item.name,
                "description": item.description,
                "wardrobe": item.wardrobe,
            }
            for item in project.characters
        ]
        scene_profiles = [
            {
                "name": profile.name,
                "description": profile.description,
                "continuity_notes": profile.continuity_notes,
            }
            for profile in project.scene_profiles
        ]
        payload = await create_llm_generator(settings.LLM_PROVIDER).generate_json(
            "你是影视分镜导演。只补写一个可独立生产的新镜头，严格承接前后镜头，不重写任何已有镜头。",
            f"""在现有分镜中新增且只新增一个镜头，预期插入为第 {intended_ordinal} 镜。

项目剧情：{self._compact_prompt_text(project.brief.story, 3000)}
统一渲染风格：{self.project_rendering_style_text(project)}
可用角色：{json.dumps(characters, ensure_ascii=False)}
可用场景档案：{json.dumps(scene_profiles, ensure_ascii=False)}
上一镜：{json.dumps(anchor.model_dump(mode='json') if anchor else {}, ensure_ascii=False)}
下一镜：{json.dumps(next_shot.model_dump(mode='json') if next_shot else {}, ensure_ascii=False)}
用户对新增镜头的要求（最高优先级）：{suggestions}

要求：
1. 只设计这个新增镜头，不能复述或改写已有镜头；时长 4–15 秒。
2. opening_state 只能描述 0.00 秒静态画面，event 与 visual_beats 描述随后发生的动作。
3. visual_beats 从 0 秒无缝覆盖至结尾，每 2–4 秒一个新的可见变化，每段一个主动作和一种主要运镜。
4. 已有场景档案能覆盖画面时，scene_profile_name 必须逐字选择其名称；新场景则填写一个简洁的新场景名称，并给出 scene_profile_description 和 scene_continuity_notes，避免套用其他空间的灯光与陈设。
5. character_names 只列真正出现在画面中的既有角色；无名群演不要伪造成主角。
6. voice_events 是唯一声音时序来源；无对白就返回空数组。默认 text_policy=post_overlay。

严格返回紧凑 JSON，不要 Markdown，也不要输出系统本地编译的 Prompt：
{{"title":"","duration_seconds":8,"event":"","opening_state":"","scene_profile_name":"","scene_profile_description":"新场景的固定空间、专属色彩与光影；已有场景留空","scene_continuity_notes":"后续机位变化仍需保持的布局、陈设和光线；已有场景留空","character_names":[],"shot_size":"","camera_angle":"","lens":"","camera_motion":"","subject_motion":"","transition":"硬切","audio_design":"","visual_beats":[{{"start_seconds":0,"end_seconds":3,"purpose":"","subject_action":"","environment_action":"","shot_size":"","camera_angle":"","camera_motion":"","sound_cue":""}}],"voice_events":[],"text_policy":"post_overlay"}}""",
        )
        try:
            duration = max(4.0, min(15.0, float(payload.get("duration_seconds") or 8.0)))
        except (TypeError, ValueError):
            duration = 8.0
        event = str(payload.get("event") or payload.get("narrative") or suggestions).strip()
        opening = str(
            payload.get("opening_state")
            or payload.get("scene_description")
            or event
        ).strip()
        names = [str(value) for value in payload.get("character_names", [])]
        shot = Shot(
            project_id=project_id,
            ordinal=intended_ordinal,
            title=str(payload.get("title") or f"镜头 {intended_ordinal} · {event[:28]}").strip(),
            duration_seconds=duration,
            narrative=event,
            scene_description=opening,
            shot_size=str(payload.get("shot_size") or "全景").strip(),
            camera_angle=str(payload.get("camera_angle") or "平视").strip(),
            lens=str(payload.get("lens") or "标准镜头").strip(),
            camera_motion=str(payload.get("camera_motion") or "固定镜头").strip(),
            subject_motion=str(payload.get("subject_motion") or event).strip(),
            transition=str(payload.get("transition") or "硬切").strip(),
            audio_design=str(payload.get("audio_design") or "").strip(),
            h3_turbo=False,
            h3_steps=25,
        )
        shot.character_ids = self.resolve_character_ids(
            project,
            shot.title,
            shot.narrative,
            shot.scene_description,
            shot.subject_motion,
            explicit_names=names,
        )
        raw_beats = payload.get("visual_beats")
        parsed_beats: list[VisualBeat] = []
        if isinstance(raw_beats, list):
            for item in raw_beats:
                try:
                    parsed_beats.append(VisualBeat.model_validate(item))
                except Exception:
                    continue
        shot.visual_beats = self._scale_visual_beats(
            parsed_beats,
            duration,
            duration,
            shot_size=shot.shot_size,
            camera_angle=shot.camera_angle,
            camera_motion=shot.camera_motion,
            fallback_action=shot.subject_motion,
        )
        raw_voices = payload.get("voice_events")
        parsed_voices: list[VoiceEvent] = []
        if isinstance(raw_voices, list):
            for item in raw_voices:
                try:
                    parsed_voices.append(VoiceEvent.model_validate(item))
                except Exception:
                    continue
        shot.voice_events = self._resolve_voice_events(project, parsed_voices, duration, duration)
        policy = str(payload.get("text_policy") or "post_overlay")
        shot.text_policy = policy if policy in {"none", "post_overlay", "reference_locked"} else "post_overlay"

        requested_profile_name = str(payload.get("scene_profile_name") or "").strip()
        scene_profile = next(
            (profile for profile in project.scene_profiles if profile.name == requested_profile_name),
            None,
        )
        if requested_profile_name and scene_profile is None:
            scene_profile = SceneProfile(
                name=requested_profile_name,
                description=str(payload.get("scene_profile_description") or opening).strip(),
                continuity_notes=str(
                    payload.get("scene_continuity_notes")
                    or "保持本镜的空间结构、主色、光源方向、排队动线和固定陈设"
                ).strip(),
            )
            project.scene_profiles.append(scene_profile)
        if scene_profile:
            shot.scene_profile_ids = [scene_profile.id]
            shot.scene_profile_id = scene_profile.id
            shot.use_scene_profile = True
        self.synchronize_shot_cast(project, shot)
        shot.visual_prompt = self.compile_visual_prompt(
            project,
            shot.scene_description or shot.narrative,
            shot.character_ids,
            shot.character_appearance_ids,
            scene_profile=scene_profile,
        )
        shot.keyframe_prompt = self.compile_keyframe_prompt(
            project,
            shot.scene_description or shot.narrative,
            shot.character_ids,
            shot.character_appearance_ids,
            shot_size=shot.shot_size,
            camera_angle=shot.camera_angle,
            lens=shot.lens,
            text_policy=shot.text_policy,
            scene_profile=scene_profile,
        )
        shot.video_prompt_source = self.compile_base_video_prompt(project, shot.subject_motion, shot.narrative)
        assets = self.store.list_assets(project_id)
        shot.video_prompt = self.compile_h3_prompt(project, shot, assets)
        shot.h3_director_version = H3_DIRECTOR_VERSION
        shot.h3_prompt_source_revision = shot.content_revision
        shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
        shot.seedance_prompt_version = SEEDANCE_PROMPT_VERSION
        shot.seedance_prompt_source_revision = shot.content_revision
        shot.keyframe_prompt_source_revision = shot.content_revision
        inserted = self.store.insert_shot(shot, after_shot_id)
        if scene_profile and inserted.id not in scene_profile.source_shot_ids:
            scene_profile.source_shot_ids.append(inserted.id)
            self.store.save_project(project)
        self.invalidate_storyboard(project_id)

        requested_targets = ["seedance"] if prompt_targets is None else prompt_targets
        targets = [target for target in dict.fromkeys(requested_targets) if target in {"h3", "seedance"}]
        if "h3" in targets:
            await self.generate_h3_prompts(project_id, [inserted.id], h3_skill_id, suggestions)
        return self.require_shot(inserted.id)

    async def generate_h3_prompts(
        self,
        project_id: str,
        shot_ids: list[str] | None,
        skill_id: str,
        user_suggestions: str = "",
    ) -> list[Shot]:
        """Generate and persist H3 prompts with the official prompt schema plus a style Skill."""
        project = self.require_project(project_id)
        skill = get_h3_prompt_skill(skill_id)
        all_shots = self.store.list_shots(project_id)
        shot_map = {shot.id: shot for shot in all_shots}
        selected_ids = list(dict.fromkeys(shot_ids or [shot.id for shot in all_shots]))
        unknown = [shot_id for shot_id in selected_ids if shot_id not in shot_map]
        if unknown:
            raise ValueError(f"项目中不存在这些镜头: {', '.join(unknown)}")
        if not selected_ids:
            raise ValueError("请至少选择一个镜头")

        for shot_id in selected_ids:
            shot = shot_map[shot_id]
            self.synchronize_shot_cast(project, shot)
            if not shot.visual_beats:
                shot.visual_beats = self._scale_visual_beats(
                    [],
                    shot.duration_seconds,
                    shot.duration_seconds,
                    shot_size=shot.shot_size,
                    camera_angle=shot.camera_angle,
                    camera_motion=shot.camera_motion or "固定镜头",
                    fallback_action=shot.subject_motion or shot.video_prompt_source,
                )
            effective_voice_events = self._resolve_voice_events(
                project,
                self._effective_voice_events(shot),
                shot.duration_seconds,
                shot.duration_seconds,
            )
            if effective_voice_events:
                shot.voice_events = effective_voice_events
                if all(event.kind != "character" for event in effective_voice_events):
                    shot.dialogue_speaker_id = None
                    shot.dialogue_turns = []
            shot_map[shot_id] = self.store.save_shot(shot)

        assets = self.store.list_assets(project_id)
        asset_map = {asset.id: asset for asset in assets}
        provider = settings.LLM_PROVIDER
        llm = create_llm_generator(provider)
        suggestions = user_suggestions.strip()
        semaphore = asyncio.Semaphore(3)

        async def generate_one(shot: Shot) -> Shot:
            mode = self.resolve_generation_mode(shot, assets)
            references = self._reference_assets_in_shot_order(project, shot, assets)
            counters = {AssetType.IMAGE: 0, AssetType.VIDEO: 0, AssetType.AUDIO: 0}
            reference_manifest: list[str] = []
            has_character_voice = any(
                event.kind == "character" and event.speaker_id for event in shot.voice_events
            )
            if mode == GenerationMode.R2V and shot.dialogue.strip() and (not shot.voice_events or has_character_voice):
                counters[AssetType.AUDIO] = 1
                speaker = next(
                    (character for character in project.characters if character.id == shot.dialogue_speaker_id),
                    None,
                )
                reference_manifest.append(
                    f"<Audio 1>: clean Chinese dialogue track for {speaker.name if speaker else 'the designated speaker'}; "
                    "only that speaker may move their lips"
                )
            for asset in references:
                if asset.type not in counters:
                    continue
                counters[asset.type] += 1
                label = {
                    AssetType.IMAGE: "Picture",
                    AssetType.VIDEO: "Video",
                    AssetType.AUDIO: "Audio",
                }[asset.type]
                reference_manifest.append(
                    f"<{label} {counters[asset.type]}>: {asset.role.value} reference named "
                    f"{self._compact_prompt_text(asset.name, 64)}; use only for its declared reference role"
                )

            visible_characters = []
            for character in project.characters:
                if character.id not in shot.character_ids:
                    continue
                appearance = next(
                    (
                        item
                        for item in character.appearance_profiles
                        if item.id == shot.character_appearance_ids.get(character.id)
                    ),
                    None,
                )
                character_ref_ids = appearance.reference_asset_ids if appearance else character.reference_asset_ids
                visible_characters.append(
                    {
                        "id": character.id,
                        "name": character.name,
                        "time_context": appearance.time_context if appearance else "base appearance",
                        "appearance": appearance.description if appearance else character.description,
                        "wardrobe": appearance.wardrobe if appearance else character.wardrobe,
                        "voice": character.voice_description,
                        "reference_assets": [
                            asset_map[asset_id].name for asset_id in character_ref_ids if asset_id in asset_map
                        ],
                        "speaks_in_this_shot": character.id == shot.dialogue_speaker_id,
                    }
                )
            voice_anchor_manifest: list[dict[str, str]] = []
            seen_voice_sources: set[tuple[str, str]] = set()
            effective_voice_events = self._resolve_voice_events(
                project,
                self._effective_voice_events(shot),
                shot.duration_seconds,
                shot.duration_seconds,
            )
            for event in effective_voice_events:
                speaker = next(
                    (character for character in project.characters if character.id == event.speaker_id),
                    None,
                )
                source_id = speaker.id if speaker else (event.speaker_name or event.kind)
                source_key = (event.kind, source_id)
                if source_key in seen_voice_sources:
                    continue
                seen_voice_sources.add(source_key)
                voice_anchor_manifest.append(
                    {
                        "source": speaker.name if speaker else (event.speaker_name or event.kind),
                        "kind": event.kind,
                        "voice_anchor": (
                            self.character_voice_anchor(speaker)
                            if speaker
                            else self._voice_anchor_for_event(project, event)
                        ),
                    }
                )
            prompt = f"""Create one final MiniMax H3 prompt for the following approved shot.

PROJECT
- title: {project.brief.title}
- aspect ratio: {project.brief.aspect_ratio}
- reusable rendering style: {self._compact_prompt_text(self.project_rendering_style_text(project), 300) or 'derive only from the selected style Skill'}
- authoritative scene context: {self._compact_prompt_text(self.shot_style_text(project, shot), 700) or 'none'}
- global negative constraints: {project.brief.negative_prompt or 'none'}

SHOT
- ordinal/title: {shot.ordinal} / {shot.title}
- duration: {shot.duration_seconds:.2f} seconds
- generation mode: {mode.value.upper()}
- narrative event: {shot.narrative}
- opening/scene state: {shot.scene_description}
- shot size / angle / lens / camera: {shot.shot_size} / {shot.camera_angle} / {shot.lens} / {shot.camera_motion}
- subject action: {shot.subject_motion}
- exact dialogue or narration: {shot.dialogue or 'none'}
- hard speech budget: at most {speech_budget_for_duration(shot.duration_seconds)} total spoken Chinese characters across all voice events; natural speed; no overlapping voice; leave opening and ending reaction/ambience
- timed visual beats: {json.dumps([beat.model_dump(mode='json') for beat in shot.visual_beats], ensure_ascii=False)}
- timed voice events: {json.dumps([event.model_dump(mode='json') for event in shot.voice_events], ensure_ascii=False)}
- approved voice anchors: {json.dumps(voice_anchor_manifest, ensure_ascii=False)}
- text policy: {shot.text_policy} (this policy controls H3 on-screen copy independently from still-frame and Seedance prompts)
- sound design: {shot.audio_design or 'natural diegetic sound only'}
- transition context: {shot.transition}
- visible characters: {json.dumps(visible_characters, ensure_ascii=False)}
- available references in exact API order: {json.dumps(reference_manifest, ensure_ascii=False)}
- current approved first-frame prompt: {self._compact_prompt_text(shot.keyframe_prompt, 500) or 'none'}
- current general storyboard prompt: {self._compact_prompt_text(shot.visual_prompt, 400) or 'none'}
- user instructions for this generation: {suggestions or 'none'}

Do not copy the full project bible or every character into the result. The authoritative scene context is the only source for this shot's environment, lighting, palette and fixed set dressing; discard conflicting environment details from older storyboard text or broad style analysis. Include only details needed for this shot. Follow the authored timed visual beats as the shot's action arc; keep one main action and one main camera move inside each beat. Keep exactly the authored character count, voice-event kind, speaker ownership, lip-sync flag, dialogue and reference labels. For every spoken event, translate its approved Chinese voice anchor into compact English silent production metadata placed before the dialogue tag; preserve age, gender presentation, timbre, pitch range, accent, pacing, and emotional delivery, but never copy or quote the Chinese voice-description prose into the final prompt. Each authored voice-event text appears in exactly one dialogue tag; only text inside dialogue tags is spoken. Never vocalize or repeat names, labels, timings, parentheses, metadata, acting directions, voice descriptions, beat summaries, soundscape prose or music prose. system_vo, narration and offscreen events never belong to a visible character. For I2V, <Picture 1> is the approved first frame. For R2V, use only labels listed in the manifest. For text_policy=post_overlay, H3 may render exact on-screen copy explicitly authored by the shot at its specified time; preserve every character and do not invent, translate or paraphrase. For text_policy=reference_locked, use only readable text already clear in the locked reference. For text_policy=none, render no readable text. The no-text post_overlay rule applies to still-frame and Seedance compilers, not this H3 prompt. Return the final prompt in the required JSON field."""
            async with semaphore:
                payload = await llm.generate_json(
                    h3_prompt_system_instruction(skill, mode),
                    prompt,
                )
            output = str(payload.get("video_prompt") or "").strip()
            if not output:
                raise ValueError(f"镜头 {shot.ordinal} 的 H3 Prompt 生成结果为空")
            # Chinese voice-profile prose next to a dialogue tag is often
            # interpreted by H3 as more dialogue. Remove legacy anchor blocks
            # and append a controlled English-only, explicitly silent contract.
            output = self.strip_h3_spoken_voice_metadata(output)
            voice_contract = self.h3_silent_voice_contract(project, effective_voice_events)
            if voice_contract:
                output = f"{output.rstrip()}\n{voice_contract}"
            # Preserve the paid result even if validation or a concurrent
            # storyboard edit prevents making it the active prompt.
            generated = shot.model_copy(deep=True)
            generated.h3_prompt_skill_id = skill.id
            generated.h3_prompt_skill_version = skill.version
            generated.h3_director_version = H3_DIRECTOR_VERSION
            generated.h3_prompt_skill_output = output
            generated.video_prompt = output
            generated.h3_prompt_source_revision = shot.content_revision
            self.store.archive_h3_prompt(generated)
            if len(output) > 6000:
                raise ValueError(f"镜头 {shot.ordinal} 的 H3 Prompt 超过 6000 字符；原文已保存在 H3 历史中")
            required = (
                ("subject_definitions:", "detailed_description:")
                if mode == GenerationMode.R2V
                else ("integrated_multimodal_description:", "overall_soundscape:")
            )
            if not all(section in output for section in required):
                raise ValueError(f"镜头 {shot.ordinal} 的生成结果不符合 {mode.value.upper()} 官方结构；原文已保存在 H3 历史中")
            # Persist each completed shot immediately. Previously the whole
            # batch was saved only after every provider call had succeeded, so
            # one late failure or a dropped HTTP request discarded the useful
            # results that had already finished.
            latest = self.require_shot(shot.id)
            if latest.content_revision != shot.content_revision:
                raise ValueError(
                    f"镜头 {shot.ordinal} 在 H3 Prompt 生成期间已被编辑；本次结果已保存在 H3 历史中，未覆盖新内容"
                )
            latest.h3_prompt_skill_id = skill.id
            latest.h3_prompt_skill_version = skill.version
            latest.h3_director_version = H3_DIRECTOR_VERSION
            latest.h3_prompt_skill_output = output
            latest.video_prompt_source = output
            latest.video_prompt = output
            latest.h3_prompt_source_revision = latest.content_revision
            latest.resolved_generation_mode = self.resolve_generation_mode(latest, assets)
            latest.version += 1
            return self.store.save_shot(latest)

        return list(await asyncio.gather(*(generate_one(shot_map[shot_id]) for shot_id in selected_ids)))

    async def generate_keyframes(
        self,
        project_id: str,
        shot_ids: list[str] | None = None,
        image_provider: str | None = None,
        image_model: str | None = None,
        revision_mode: str = "fresh",
        user_suggestions: str = "",
    ) -> list[Shot]:
        if revision_mode not in {"fresh", "iterate"}:
            raise ValueError(f"Unsupported keyframe revision mode: {revision_mode}")
        project = self.require_project(project_id)
        user_suggestions = user_suggestions.strip()
        shots = self.store.list_shots(project_id)
        if shot_ids:
            selected = set(shot_ids)
            unknown = selected - {shot.id for shot in shots}
            if unknown:
                raise ValueError(f"项目中不存在这些镜头: {', '.join(sorted(unknown))}")
            shots = [shot for shot in shots if shot.id in selected]
        for shot in shots:
            self.ensure_keyframe_idle(shot.id)
            self.synchronize_shot_cast(project, shot)
        assets = self.store.list_assets(project_id)
        image_gen = create_image_generator(image_provider=image_provider, model=image_model)
        image_dir = self.project_dir(project_id) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        fallback_style_reference = next(
            (str(resolve_media_path(asset.path)) for asset in assets if asset.type == AssetType.IMAGE and asset.role == AssetRole.STYLE),
            None,
        )
        asset_map = {asset.id: asset for asset in assets}
        semaphore = asyncio.Semaphore(max(1, settings.WORKFLOW_CONCURRENCY))

        # Save the draft before any remote request. A failed generation keeps
        # the user's exact suggestion and revision mode for one-click retry.
        for shot in shots:
            shot.keyframe_revision_suggestion_draft = user_suggestions
            shot.keyframe_revision_mode = revision_mode
            # Include queued shots so refreshes and duplicate requests see them.
            shot.image_status = "processing"
            self.store.save_shot(shot)
        self._active_keyframes.update(shot.id for shot in shots)

        async def generate(shot: Shot) -> None:
            async with semaphore:
                previous_keyframe_id = shot.keyframe_asset_id
                previous_keyframe = asset_map.get(previous_keyframe_id) if previous_keyframe_id else None
                reference_paths: list[str] = []
                if (
                    revision_mode == "iterate"
                    and previous_keyframe is not None
                    and previous_keyframe.type == AssetType.IMAGE
                ):
                    reference_paths.append(str(resolve_media_path(previous_keyframe.path)))
                effective_prompt = (
                    shot.keyframe_prompt.strip()
                    or shot.scene_description.strip()
                    or shot.visual_prompt.strip()
                )
                if not shot.keyframe_prompt.strip():
                    shot.keyframe_prompt = effective_prompt
                effective_character_ids = self.keyframe_character_ids(project, shot, effective_prompt)
                keyframe_references = self.keyframe_reference_assets(
                    project,
                    shot,
                    assets,
                    effective_character_ids,
                )
                reference_paths.extend(
                    str(resolve_media_path(asset.path))
                    for asset in keyframe_references
                    if asset.id != previous_keyframe_id
                )
                reference_paths = list(dict.fromkeys(reference_paths))
                if not reference_paths and fallback_style_reference:
                    reference_paths.append(fallback_style_reference)

                suggestion_is_duplicate = bool(
                    user_suggestions
                    and self.merge_keyframe_suggestions(effective_prompt, user_suggestions) == effective_prompt
                )
                if not effective_prompt.startswith("【首帧画面】"):
                    profile = self.scene_profile_for_shot(project, shot)
                    effective_prompt = self.compile_keyframe_prompt(
                        project,
                        effective_prompt,
                        effective_character_ids,
                        shot.character_appearance_ids,
                        shot_size=shot.shot_size,
                        camera_angle=shot.camera_angle,
                        lens=shot.lens,
                        text_policy=shot.text_policy,
                        scene_profile=profile,
                    )
                effective_prompt = self.apply_fixed_prop_anchor(
                    effective_prompt,
                    keyframe_references,
                )
                effective_prompt = self.merge_keyframe_suggestions(
                    effective_prompt,
                    "" if suggestion_is_duplicate else user_suggestions,
                )
                profile = self.scene_profile_for_shot(project, shot)
                if profile and "【场景档案｜本镜最高优先级】" not in effective_prompt:
                    effective_prompt = (
                        f"{effective_prompt}\n\n【场景档案｜本镜最高优先级】{profile.name}："
                        f"{profile.description}；{profile.continuity_notes}；"
                        "本镜环境、光线、色彩和陈设只按此档案执行，忽略其他冲突描述"
                    ).strip()
                # ``effective_prompt`` is the complete provider prompt. Keep a
                # single visible source of truth and do not inject cast/style a
                # second time inside the image adapter.
                shot.keyframe_prompt = effective_prompt
                logger.info(
                    "Keyframe request: shot=%s prompt_chars=%s characters=%s references=%s revision=%s",
                    shot.ordinal,
                    len(effective_prompt),
                    len(effective_character_ids),
                    len(reference_paths),
                    revision_mode,
                )

                # A queued shot may have received its H3 prompt while waiting
                # for a slot. Never save the request's old whole-shot snapshot.
                latest = self.require_shot(shot.id)
                latest.keyframe_prompt = effective_prompt
                latest.version += 1
                self.store.save_shot(latest)
                scene = Scene(
                    id=shot.ordinal,
                    narrative=shot.narrative,
                    visual_prompt=effective_prompt,
                    motion_prompt=shot.video_prompt or shot.subject_motion,
                    duration=max(4, min(15, round(shot.duration_seconds))),
                )
                try:
                    # Each attempt gets its own directory so a successful revision
                    # never overwrites the image that was used as its reference.
                    attempt_dir = image_dir / f"{shot.id}-{uuid4().hex}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    output = await image_gen.generate_image(
                        scene,
                        str(attempt_dir),
                        ",".join(reference_paths) or None,
                        seed=self._stable_seed(project_id, shot.id),
                        character_description="",
                        image_style="",
                        aspect_ratio=project.brief.aspect_ratio,
                    )
                    path = Path(output)
                    asset = self.register_existing_asset(
                        project_id,
                        path,
                        role=AssetRole.KEYFRAME,
                        name=(
                            f"镜头 {shot.ordinal} 首帧（基于上一版修改）"
                            if revision_mode == "iterate" and previous_keyframe is not None
                            else f"镜头 {shot.ordinal} 首帧"
                        ),
                        description=effective_prompt,
                    )
                    latest = self.require_shot(shot.id)
                    latest.keyframe_asset_id = asset.id
                    latest.image_path = str(path)
                    latest.image_status = "completed"
                    if latest.video_reference_asset_ids is not None and previous_keyframe_id:
                        latest.video_reference_asset_ids = [
                            asset.id if asset_id == previous_keyframe_id else asset_id
                            for asset_id in latest.video_reference_asset_ids
                        ]
                        latest.video_reference_asset_ids = list(
                            dict.fromkeys(latest.video_reference_asset_ids)
                        )
                    latest.keyframe_revision_last_suggestion = user_suggestions
                    latest.keyframe_revision_suggestion_draft = ""
                    latest.keyframe_prompt_source_revision = shot.content_revision
                    latest.version += 1
                    self.store.save_shot(latest)
                except Exception:
                    logger.exception(
                        "镜头 %s（%s）首帧生成失败: provider=%s model=%s",
                        shot.ordinal,
                        shot.id,
                        image_provider or settings.IMAGE_PROVIDER,
                        image_model or (
                            settings.GRSAI_IMAGE_MODEL
                            if (image_provider or settings.IMAGE_PROVIDER) == "grsai"
                            else settings.ARK_IMAGE_MODEL
                        ),
                    )
                    raise

        async def guarded_generate(shot: Shot) -> None:
            try:
                await generate(shot)
            finally:
                # Also handle prompt-preparation errors and cancellation. Only
                # update image status; preserve prompts and the previous image.
                try:
                    latest = self.store.get_shot(shot.id)
                    if latest is not None and latest.image_status == "processing":
                        latest.image_status = "failed"
                        self.store.save_shot(latest)
                finally:
                    self._active_keyframes.discard(shot.id)

        results = await asyncio.gather(*(guarded_generate(shot) for shot in shots), return_exceptions=True)
        failures = [
            f"镜头 {shot.ordinal}: {result}"
            for shot, result in zip(shots, results, strict=False)
            if isinstance(result, BaseException)
        ]
        project = self.require_project(project_id)
        project.status = ProjectStatus.KEYFRAMES_REVIEW
        self.store.save_project(project)
        if failures:
            raise RuntimeError("部分分镜图生成失败；成功结果已保留。" + "；".join(failures))
        return self.store.list_shots(project_id)

    @staticmethod
    def _cover_reference_assets(
        project: Project,
        shots: list[Shot],
        assets: list[Asset],
    ) -> list[Asset]:
        """Pick a small, stable identity/style set for a project cover."""
        assets_by_id = {asset.id: asset for asset in assets}
        ordered_ids: list[str] = []

        # Lead characters are ranked by storyboard appearances so a large cast
        # does not crowd the cover or exhaust the provider's reference limit.
        appearances = {
            character.id: sum(character.id in shot.character_ids for shot in shots)
            for character in project.characters
        }
        for character in sorted(
            project.characters,
            key=lambda item: (-appearances[item.id], project.characters.index(item)),
        )[:4]:
            candidates = [
                *character.reference_asset_ids,
                *[
                    asset.id
                    for asset in assets
                    if asset.role == AssetRole.CHARACTER and asset.character_id == character.id
                ],
            ]
            ordered_ids.extend(candidates[:1])

        style_ids = [
            *(project.style_profile.reference_asset_ids if project.style_profile and project.style_profile.approved else []),
            *(
                asset.id
                for asset in assets
                if asset.role == AssetRole.STYLE and asset.type == AssetType.IMAGE
            ),
        ]
        ordered_ids.extend(list(dict.fromkeys(style_ids))[:2])

        # Theme-defining props (for example an overloaded power strip) deserve
        # their own identity reference instead of being inferred from prose.
        prop_assets = [
            asset
            for asset in assets
            if asset.role == AssetRole.PROP and asset.type == AssetType.IMAGE
        ]
        ordered_ids.extend(
            asset.id
            for asset in sorted(
                prop_assets,
                key=lambda asset: -sum(
                    asset.id in shot.reference_asset_ids
                    or asset.id in (shot.keyframe_reference_asset_ids or [])
                    or asset.id in (shot.video_reference_asset_ids or [])
                    for shot in shots
                ),
            )[:2]
        )
        scene_ids = [
            asset_id
            for profile in project.scene_profiles
            if profile.approved
            for asset_id in profile.reference_asset_ids[:1]
        ]
        ordered_ids.extend(list(dict.fromkeys(scene_ids))[:1])
        ordered_ids.extend(
            shot.keyframe_asset_id
            for shot in sorted(shots, key=lambda item: (-len(item.character_ids), item.ordinal))
            if shot.keyframe_asset_id
        )

        selected: list[Asset] = []
        for asset_id in dict.fromkeys(ordered_ids):
            asset = assets_by_id.get(asset_id)
            if (
                asset
                and asset.type == AssetType.IMAGE
                and asset.role not in {AssetRole.COVER, AssetRole.COVER_REFERENCE}
                and resolve_media_path(asset.path).is_file()
            ):
                selected.append(asset)
            if len(selected) >= 10:
                break
        return selected

    @classmethod
    def compile_project_cover_prompt(
        cls,
        project: Project,
        shots: list[Shot],
        references: list[Asset],
        user_suggestions: str = "",
        aspect_ratio: str = "16:9",
        reference_mode: str = "none",
    ) -> str:
        title = project.brief.title.strip()
        story = project.brief.story.strip()[:2400]
        style = cls.project_style_text(project).strip()[:1800] or "沿用项目已有分镜的统一视觉风格"
        character_lines = []
        for character in project.characters[:6]:
            details = "；".join(
                item.strip()
                for item in (character.description, character.wardrobe)
                if item.strip()
            )
            character_lines.append(f"- {character.name}：{details[:500] or '沿用角色参考图中的固定身份与造型'}")
        shot_clues = "\n".join(
            f"- 镜头 {shot.ordinal}《{shot.title or '未命名'}》：{(shot.narrative or shot.scene_description)[:220]}"
            for shot in shots[:16]
            if shot.narrative.strip() or shot.scene_description.strip()
        )[:2800]
        role_labels = {
            AssetRole.CHARACTER: "人物形象",
            AssetRole.STYLE: "画风",
            AssetRole.SCENE: "场景",
            AssetRole.KEYFRAME: "分镜首帧",
            AssetRole.PROP: "关键道具",
            AssetRole.COVER_REFERENCE: "用户上传的封面参考",
            AssetRole.COVER: "上一版生成封面",
        }
        reference_lines = "\n".join(
            f"- 参考图 {index}：{asset.name}（{role_labels.get(asset.role, '项目视觉参考')}）"
            for index, asset in enumerate(references, start=1)
        ) or "- 当前没有可用参考图，严格依据项目设定绘制"
        cover_reference_instruction = {
            "none": "不提交封面构图参考，根据项目设定全新构图。后续项目人设、画风、道具和分镜参考仍用于保持一致性。",
            "uploaded": "参考图 1 是用户上传的封面参考；借鉴它的构图、信息层级和氛围，但人物身份、故事内容和准确文字以本项目设定及用户建议为准。",
            "previous": "参考图 1 是上一版生成封面；保留用户建议未要求改动的人物、构图、色彩和视觉层级，重点执行本次建议。",
        }.get(reference_mode, "根据项目设定全新构图。")
        suggestion_block = user_suggestions.strip() or "无额外建议，由系统选择故事冲突最强的瞬间"
        return f"""【任务】为本视频项目生成一张可直接发布的高点击率中文主封面，不是普通分镜截图。
【封面比例】{aspect_ratio}，横竖构图必须严格适配该比例，重要人物、标题和危险物件都留在安全区。
【主标题】画面上方使用最大字号、强对比、粗描边的中文标题。若用户建议没有另行指定标题，文字必须准确写成“{title}”；若用户明确指定了新标题，以用户建议为准并准确呈现。
【故事核心】{story}
【主要角色】
{chr(10).join(character_lines) or '- 本项目没有已建档角色，按故事中的主角关系设计'}
【可用镜头线索】
{shot_clues or '- 尚无分镜，从故事核心提炼视觉冲突'}
【统一视觉风格】{style}
【参考图顺序】
{reference_lines}
【封面参考方式】{cover_reference_instruction}
【本次用户建议｜最高优先级】{suggestion_block}
【封面设计语言】参考成熟中文科普短片/剧情短视频海报：顶部大标题占约 20%，中心用主角的大幅强情绪反应形成第一视觉焦点，关键配角在侧后方形成态度对照；把最能说明主题的风险物件或事件放在前景并适度夸张，背景用环境细节补足故事世界。采用前景暖色风险光与背景冷色空间光的层次对比，轮廓清楚、构图饱满、信息一眼可读，兼具戏剧性、科普感和点击吸引力。
【硬性限制】角色身份、年龄、发型和服装必须服从本项目人物参考图与设定；用户上传的封面参考只用于构图、信息层级和氛围，不得用它覆盖本项目的人物身份或故事。只允许出现上述准确主标题，除非用户建议明确要求，否则不要生成小字、标签、榜单、乱码、Logo、水印、边框或界面元素；不要把封面做成角色设定板、九宫格、拼贴或电影截图。""".strip()

    async def generate_project_cover(
        self,
        project_id: str,
        user_suggestions: str = "",
        aspect_ratio: str = "16:9",
        reference_mode: str = "none",
        reference_asset_id: str | None = None,
        image_provider: str | None = None,
        image_model: str | None = None,
    ) -> Asset:
        supported_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4"}
        supported_reference_modes = {"none", "uploaded", "previous"}
        if aspect_ratio not in supported_ratios:
            raise ValueError(f"不支持的封面比例: {aspect_ratio}")
        if reference_mode not in supported_reference_modes:
            raise ValueError(f"不支持的封面参考方式: {reference_mode}")
        if project_id in self._active_project_covers:
            raise ProjectCoverBusyError("本项目封面正在生成，请等待完成后再提交下一版")

        project = self.require_project(project_id)
        shots = self.store.list_shots(project_id)
        assets = self.store.list_assets(project_id)
        cover_reference: Asset | None = None
        if reference_mode == "uploaded":
            if not reference_asset_id:
                raise ValueError("请先上传并选择一张封面参考图")
            cover_reference = next(
                (
                    asset
                    for asset in assets
                    if asset.id == reference_asset_id
                    and asset.role == AssetRole.COVER_REFERENCE
                    and asset.type == AssetType.IMAGE
                ),
                None,
            )
            if cover_reference is None or not resolve_media_path(cover_reference.path).is_file():
                raise ValueError("选中的封面参考图不存在或不是可用图片")
        elif reference_mode == "previous":
            cover_reference = next(
                (
                    asset
                    for asset in reversed(assets)
                    if asset.role == AssetRole.COVER
                    and asset.type == AssetType.IMAGE
                    and "generated-cover" in asset.tags
                    and resolve_media_path(asset.path).is_file()
                ),
                None,
            )
            if cover_reference is None:
                raise ValueError("当前还没有可作为参考的上一版生成封面")

        project_references = self._cover_reference_assets(project, shots, assets)
        references = list(dict.fromkeys(
            asset.id for asset in ([cover_reference] if cover_reference else []) + project_references
        ))
        assets_by_id = {asset.id: asset for asset in assets}
        references = [assets_by_id[asset_id] for asset_id in references if asset_id in assets_by_id][:10]
        prompt = self.compile_project_cover_prompt(
            project,
            shots,
            references,
            user_suggestions,
            aspect_ratio,
            reference_mode,
        )
        output_dir = self.project_dir(project_id) / "covers" / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = random.randint(1, 2**31 - 1)
        scene = Scene(
            id=1,
            narrative=project.brief.story[:1200],
            visual_prompt=prompt,
            motion_prompt="",
            duration=5,
        )
        image_gen = create_image_generator(image_provider=image_provider, model=image_model)
        self._active_project_covers.add(project_id)
        try:
            output = await image_gen.generate_image(
                scene,
                str(output_dir),
                ",".join(str(resolve_media_path(asset.path)) for asset in references) or None,
                seed=seed,
                character_description="",
                image_style="",
                aspect_ratio=aspect_ratio,
            )
            asset = self.register_existing_asset(
                project_id,
                Path(output),
                role=AssetRole.COVER,
                name=f"{project.brief.title} · 项目封面 {aspect_ratio}",
                description=prompt,
            )
            asset.tags = [
                "generated-cover",
                f"cover-ratio:{aspect_ratio}",
                f"seed:{seed}",
                f"cover-reference-mode:{reference_mode}",
            ]
            if cover_reference:
                asset.tags.append(f"cover-reference-asset:{cover_reference.id}")
            return self.store.save_asset(asset)
        finally:
            self._active_project_covers.discard(project_id)

    def register_existing_asset(
        self,
        project_id: str,
        path: Path,
        role: AssetRole,
        name: str,
        description: str = "",
    ) -> Asset:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            asset_type = AssetType.IMAGE
        elif suffix in {".mp4", ".mov", ".webm", ".mkv"}:
            asset_type = AssetType.VIDEO
        elif suffix in {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}:
            asset_type = AssetType.AUDIO
        elif suffix in {".srt", ".vtt", ".ass"}:
            asset_type = AssetType.SUBTITLE
        else:
            asset_type = AssetType.DOCUMENT
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        asset = Asset(
            project_id=project_id,
            type=asset_type,
            role=role,
            name=name,
            path=str(path.resolve()),
            mime_type="application/octet-stream",
            description=description,
            sha256=digest.hexdigest(),
            size_bytes=path.stat().st_size,
        )
        return self.store.save_asset(asset)

    def list_legacy_sessions(self) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        output_root = settings.OUTPUT_DIR.resolve()
        for script_path in sorted(output_root.glob("*/script.json"), reverse=True):
            try:
                payload = json.loads(script_path.read_text(encoding="utf-8"))
                scenes = payload.get("scenes") if isinstance(payload, dict) else []
                sessions.append(
                    {
                        "session_id": script_path.parent.name,
                        "topic": str(payload.get("topic") or script_path.parent.name),
                        "scene_count": len(scenes) if isinstance(scenes, list) else 0,
                        "has_images": any((script_path.parent / "images").glob("*")),
                        "has_videos": any((script_path.parent / "videos").glob("*")),
                    }
                )
            except (OSError, ValueError, TypeError):
                continue
        return sessions

    def import_legacy_session(self, session_id: str) -> Project:
        safe_id = Path(session_id).name
        source_dir = (settings.OUTPUT_DIR / safe_id).resolve()
        output_root = settings.OUTPUT_DIR.resolve()
        if output_root not in source_dir.parents:
            raise ValueError("Invalid legacy session id")
        script_path = source_dir / "script.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Legacy session not found: {safe_id}")
        payload = json.loads(script_path.read_text(encoding="utf-8"))
        scenes = payload.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Legacy session contains no scenes")
        durations = [max(0.25, min(15.0, float(scene.get("duration") or 5.0))) for scene in scenes]
        project = self.create_project(
            ProjectBrief(
                title=str(payload.get("topic") or f"导入项目 {safe_id}"),
                story=str(payload.get("topic") or "从旧版 VideoWorkflow 导入"),
                target_duration_seconds=sum(durations),
            )
        )
        destination = self.project_dir(project.id)
        imported_shots: list[Shot] = []
        for ordinal, (scene, duration) in enumerate(zip(scenes, durations, strict=False), start=1):
            event = str(
                scene.get("event")
                or scene.get("story_beat")
                or scene.get("narrative")
                or ""
            ).strip()
            opening_state = str(
                scene.get("opening_state")
                or scene.get("keyframe_prompt")
                or scene.get("visual_prompt")
                or event
            ).strip()
            beat_actions = [
                str(item.get("subject_action") or item.get("action") or "").strip()
                for item in (scene.get("visual_beats") or [])
                if isinstance(item, dict) and str(item.get("subject_action") or item.get("action") or "").strip()
            ]
            base_video_prompt = str(scene.get("motion_prompt") or "；".join(beat_actions) or "").strip()
            shot = Shot(
                project_id=project.id,
                ordinal=ordinal,
                title=f"镜头 {ordinal}",
                narrative=event,
                duration_seconds=duration,
                scene_description=opening_state,
                visual_prompt=str(scene.get("visual_prompt") or opening_state),
                keyframe_prompt=str(scene.get("keyframe_prompt") or opening_state),
                subject_motion=base_video_prompt,
                video_prompt_source=base_video_prompt,
                video_prompt=base_video_prompt,
                image_status=str(scene.get("image_status") or "pending"),
                video_status=str(scene.get("video_status") or "pending"),
                render_frames=h3_frames_for_seconds(duration),
                h3_turbo=False,
                h3_steps=25,
            )
            for key, folder, role in (
                ("image_path", "images", AssetRole.KEYFRAME),
                ("video_path", "videos", AssetRole.OUTPUT),
            ):
                raw = scene.get(key)
                if not raw:
                    continue
                source = Path(str(raw))
                if not source.is_absolute():
                    candidates = [(Path.cwd() / source).resolve(), (source_dir / source.name).resolve(), (source_dir / folder / source.name).resolve()]
                    source = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
                if not source.exists():
                    continue
                target_dir = destination / folder / shot.id
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / source.name
                shutil.copy2(source, target)
                if key == "image_path":
                    asset = self.register_existing_asset(project.id, target, role, f"镜头 {ordinal} 旧版首帧")
                    shot.keyframe_asset_id = asset.id
                    shot.image_path = str(target.resolve())
                else:
                    self.register_existing_asset(project.id, target, role, f"镜头 {ordinal} 旧版视频")
                    shot.video_path = str(target.resolve())
            imported_shots.append(shot)
        self.store.replace_shots(project.id, imported_shots)
        project.status = ProjectStatus.CLIPS_REVIEW if all(shot.video_path for shot in imported_shots) else ProjectStatus.KEYFRAMES_REVIEW
        return self.store.save_project(project)

    @staticmethod
    def _stable_seed(project_id: str, *identity_parts: str) -> int:
        """Return a deterministic seed for any project-scoped entity identity.

        Keeping the helper variadic preserves the existing ``project:shot``
        seeds while allowing distinct character appearance profiles to derive
        their own repeatable seeds.
        """
        digest = hashlib.sha256(":".join((project_id, *identity_parts)).encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def approve_storyboard(self, project_id: str, approved: bool, comment: str = "") -> Project:
        project = self.require_project(project_id)
        project.status = ProjectStatus.STORYBOARD_APPROVED if approved else ProjectStatus.STORYBOARD_DRAFT
        for shot in self.store.list_shots(project_id):
            shot.approval_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.CHANGES_REQUESTED
            self.store.save_shot(shot)
        return self.store.save_project(project)
