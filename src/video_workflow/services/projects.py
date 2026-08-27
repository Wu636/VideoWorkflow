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
from src.video_workflow.core.orchestrator import WorkflowOrchestrator, create_llm_generator
from src.video_workflow.domain import (
    ApprovalStatus,
    Asset,
    AssetRole,
    AssetType,
    CharacterAppearanceProfile,
    DialogueTurn,
    GenerationMode,
    Project,
    ProjectAnalysisDraft,
    ProjectBrief,
    ProjectStatus,
    SceneConsistencyMode,
    SceneProfile,
    ScriptRewriteDraft,
    Shot,
    ShotContinuityMode,
    StyleAnalysisDraft,
    StyleProfile,
    new_id,
)
from src.video_workflow.generators.image import create_image_generator
from src.video_workflow.h3_prompt_skills import (
    get_h3_prompt_skill,
    h3_prompt_system_instruction,
)
from src.video_workflow.integrations.comfyui import h3_frames_for_seconds
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.storage import ProjectStore
from src.video_workflow.types import Scene

logger = logging.getLogger(__name__)


class ProjectService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def create_project(self, brief: ProjectBrief) -> Project:
        project = Project(brief=brief)
        self.project_dir(project.id).mkdir(parents=True, exist_ok=True)
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
    def project_style_text(project: Project) -> str:
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
            return "；".join(item.strip() for item in fields if item and item.strip())
        return project.brief.visual_style or project.style_bible

    @staticmethod
    def scene_profile_for_shot(project: Project, shot: Shot) -> SceneProfile | None:
        if not shot.use_scene_profile or not shot.scene_profile_id:
            return None
        return next(
            (profile for profile in project.scene_profiles if profile.id == shot.scene_profile_id),
            None,
        )

    def update_project(self, project: Project) -> Project:
        existing = self.store.get_project(project.id)
        style_changed = bool(
            existing
            and (
                existing.style_profile != project.style_profile
                or existing.style_bible != project.style_bible
                or existing.brief.visual_style != project.brief.visual_style
                or existing.brief.negative_prompt != project.brief.negative_prompt
            )
        )
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
        if style_changed:
            assets = self.store.list_assets(project.id)
            for shot in self.store.list_shots(project.id):
                shot.content_revision += 1
                character_ids = self.keyframe_character_ids(saved_project, shot, shot.scene_description)
                shot.visual_prompt = self.compile_visual_prompt(
                    saved_project,
                    shot.scene_description or shot.narrative,
                    character_ids,
                    shot.character_appearance_ids,
                )
                shot.keyframe_prompt = self.compile_keyframe_prompt(
                    saved_project,
                    shot.scene_description or shot.narrative,
                    character_ids,
                    shot.character_appearance_ids,
                    shot_size=shot.shot_size,
                    camera_angle=shot.camera_angle,
                    lens=shot.lens,
                )
                shot.keyframe_prompt_source_revision = shot.content_revision
                shot.video_prompt_source = self.compile_base_video_prompt(
                    saved_project,
                    shot.subject_motion,
                    shot.narrative,
                )
                shot.h3_prompt_skill_output = ""
                shot.video_prompt = self.compile_h3_prompt(saved_project, shot, assets)
                shot.h3_prompt_source_revision = shot.content_revision
                shot.seedance_prompt = self.compile_seedance_prompt(saved_project, shot, assets)
                shot.seedance_prompt_source_revision = shot.content_revision
                shot.approval_status = ApprovalStatus.DRAFT
                shot.version += 1
                self.store.save_shot(shot)
        return saved_project

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
镜头数量要求高于“单镜建议 4–8 秒”的一般建议。本项目目标时长为 {project.brief.target_duration_seconds:g} 秒，平均每镜约 {average_duration:.2f} 秒；请在单镜不超过 15 秒的范围内调整每镜信息密度和 duration。
第 1 镜必须承担故事开端，第 {count} 镜必须完整呈现原剧情结局；中间 {max(0, count - 2)} 镜覆盖关键因果、转折与必要对白，确保压缩后仍是完整故事。
输出 JSON 前先自行计数校验，只有 scenes 恰好 {count} 项才能提交。"""
        base_suggestions = "\n\n".join(item for item in [user_suggestions, count_constraint] if item)
        storyboard = None
        actual_count = 0
        # Validation never edits the returned scenes. If a provider ignores the
        # requested count, ask it to re-plan the whole story so the ending and
        # causal chain are preserved instead of truncating/padding the result.
        for attempt in range(3):
            retry_note = ""
            if attempt:
                retry_note = (
                    f"\n\n【数量校验未通过，第 {attempt + 1} 次完整重构】"
                    f"上一次返回了 {actual_count} 个分镜，不符合严格 {count} 镜。"
                    f"请从完整剧情重新规划全部 {count} 镜，不要删掉数组尾部来凑数，也不要复制补镜；"
                    "通过合并或拆分叙事单元，让开端、关键因果和结局都保留。"
                )
            storyboard = await orchestrator.llm.generate_storyboard(
                topic=project.brief.story,
                count=count,
                reference_image=reference,
                include_dialogue=True,
                character_description=self.character_bible(project),
                image_style=self.project_style_text(project),
                user_suggestions=base_suggestions + retry_note,
            )
            actual_count = len(storyboard.scenes)
            if actual_count == count:
                break
        if storyboard is None or actual_count != count:
            raise ValueError(
                f"AI 连续 3 次未遵守严格 {count} 镜要求（最后返回 {actual_count} 镜）。"
                "本次结果未保存，原有分镜保持不变；请调整模型配置或精简本次建议后重试。"
            )
        durations = self._allocate_durations(
            [float(scene.duration) for scene in storyboard.scenes],
            project.brief.target_duration_seconds,
        )
        shots: list[Shot] = []
        sizes = ["全景", "中景", "近景", "特写", "中景"]
        for ordinal, (scene, duration) in enumerate(zip(storyboard.scenes, durations, strict=False), start=1):
            speaker_id = self._resolve_dialogue_speaker_id(
                project,
                scene.dialogue_speaker,
                scene.story_beat,
                scene.narrative,
                scene.dialogue,
            )
            scene_text = " ".join(
                item for item in [scene.story_beat, scene.narrative, scene.visual_prompt, scene.dialogue] if item
            )
            scene_character_ids = [
                character.id
                for character in project.characters
                if character.name and character.name in scene_text
            ]
            if speaker_id and speaker_id not in scene_character_ids:
                scene_character_ids.append(speaker_id)
            normalized_motion = self._normalize_motion_duration(scene.motion_prompt, duration)
            base_video_prompt = self.compile_base_video_prompt(
                project,
                normalized_motion,
                scene.story_beat or scene.narrative,
            )
            shot = Shot(
                project_id=project_id,
                ordinal=ordinal,
                title=f"镜头 {ordinal} · {(scene.story_beat or scene.visual_prompt)[:18]}",
                narrative=scene.story_beat or scene.visual_prompt,
                dialogue=scene.dialogue,
                dialogue_speaker_id=speaker_id,
                duration_seconds=round(duration, 3),
                scene_description=scene.visual_prompt,
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
                visual_prompt=self.compile_visual_prompt(project, scene.visual_prompt, scene_character_ids),
                keyframe_prompt=self.compile_keyframe_prompt(
                    project,
                    scene.keyframe_prompt or scene.visual_prompt,
                    scene_character_ids,
                    shot_size=scene.shot_size or sizes[(ordinal - 1) % len(sizes)],
                    camera_angle=scene.camera_angle or "平视",
                    lens=scene.lens or ("35mm 广角" if ordinal == 1 else "50mm 标准镜头"),
                ),
                video_prompt_source=base_video_prompt,
                video_prompt=base_video_prompt,
                negative_prompt=project.brief.negative_prompt,
                h3_turbo=False,
                h3_steps=25,
                render_frames=h3_frames_for_seconds(duration),
            )
            shots.append(shot)
        normalized_targets = [
            target for target in dict.fromkeys(prompt_targets or []) if target in {"h3", "seedance"}
        ]
        # The LLM call above can take a while. Merge the storyboard fields onto
        # the newest project instead of saving the stale pre-call snapshot and
        # accidentally erasing scene/character assets created in parallel.
        latest_project = self.require_project(project_id)
        if prompt_targets is not None:
            latest_project.preferred_prompt_targets = normalized_targets
        latest_project.storyboard_count_mode = "ai" if count_mode == "ai" else "manual"
        latest_project.manual_shot_count = count if count_mode == "manual" else None
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
        fallback = max(minimum, min(500, math.ceil(project.brief.target_duration_seconds / 8.0)))
        provider = settings.SHOT_COUNT_PROVIDER
        if provider == "auto":
            provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        orchestrator = WorkflowOrchestrator(provider)
        prompt = f"""分析下面的剧情脚本，为总时长 {project.brief.target_duration_seconds:g} 秒的视频决定分镜数量。
每个生成片段不得超过 15 秒，通常建议 4–8 秒；但用户建议中明确限制镜头总数时，只要不突破 15 秒上限就优先尊重。每镜只保留一个核心动作和一名发言者。
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
        prompt = f"""根据项目剧本、角色设定和客户分镜表，补全每行缺失的制作字段。
必须严格返回 {len(source_rows)} 行，顺序不变，不合并、不拆分、不新增分镜；客户已填写的值原样保留。
每镜时长不超过 15 秒，全部时长适配目标成片 {project.brief.target_duration_seconds:g} 秒。
字段为：{fields}。
character_names 使用逗号分隔项目中已有角色名；generation_mode 只能是 auto、i2v、r2v。
严格返回 JSON：{{"rows": [{{上述全部字段}}]}}。

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
            character_ids = [
                character.id
                for character in project.characters
                if character.name in character_names
                or any(character.name and character.name in row.get(key, "") for key in ("narrative", "dialogue", "scene_description"))
            ]
            speaker_id = self._resolve_dialogue_speaker_id(
                project,
                row.get("dialogue_speaker", ""),
                row.get("narrative", ""),
                row.get("dialogue", ""),
            )
            if speaker_id and speaker_id not in character_ids:
                character_ids.append(speaker_id)
            scene = row.get("scene_description") or row.get("narrative") or f"镜头 {ordinal}"
            subject_motion = row.get("subject_motion") or "角色保持连贯、自然的小幅动作"
            try:
                generation_mode = GenerationMode(row.get("generation_mode") or "auto")
            except ValueError:
                generation_mode = GenerationMode.AUTO
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

        normalized_targets = [
            target for target in dict.fromkeys(prompt_targets or []) if target in {"h3", "seedance"}
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
        minimum = max(1, math.ceil(project.brief.target_duration_seconds / 8.0))
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

recommended_shot_count 必须不少于 {minimum}，确保任一镜头建议不超过 8 秒且只包含一个核心动作；角色描述必须具备跨镜头复用的一致性，不要改变参考图中的身份特征。
"""
        provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        analysis_llm = create_llm_generator(provider)
        if reference_paths:
            vision_provider = settings.REFERENCE_ANALYSIS_PROVIDER
            if vision_provider == "auto":
                vision_provider = "ark" if settings.ARK_API_KEY else "glm" if settings.GLM_API_KEY else provider
            analysis_llm = create_llm_generator(vision_provider)
        payload = await analysis_llm.generate_json(
            "你是资深影视策划、分镜导演和角色一致性设计师。把模糊客户需求整理成可直接用于 AI 视频生产的结构化参数。",
            prompt,
            reference_images=reference_paths,
        )
        payload["recommended_shot_count"] = max(
            minimum,
            min(500, int(payload.get("recommended_shot_count") or math.ceil(project.brief.target_duration_seconds / 6.5))),
        )
        draft = ProjectAnalysisDraft.model_validate(payload)
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
        provider = settings.REFERENCE_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = "ark" if settings.ARK_API_KEY else "glm" if settings.GLM_API_KEY else settings.LLM_PROVIDER
        llm = create_llm_generator(provider)
        prompt = f"""分析这些客户参考画面，识别可复用的 AI 视频视觉风格。项目剧情仅用于判断适配性，不要凭剧情改写画面观察。

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
        return StyleAnalysisDraft.model_validate(payload)

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
                    id=existing.id if existing else new_id("scene"),
                    name=name,
                    description=str(item.get("description") or ""),
                    continuity_notes=str(item.get("continuity_notes") or ""),
                    reference_asset_ids=list(existing.reference_asset_ids) if existing else [],
                    source_shot_ids=source_ids,
                    approved=bool(existing and existing.approved),
                )
            )
        latest_project.scene_profiles = profiles
        latest_project.scene_consistency_mode = SceneConsistencyMode.OPTIONAL if profiles else SceneConsistencyMode.OFF
        self.store.save_project(latest_project)
        # Bind the suggested profile without enabling it. Users opt in per shot
        # or switch the project to strict mode explicitly.
        profile_by_shot = {
            shot_id: profile.id
            for profile in profiles
            for shot_id in profile.source_shot_ids
        }
        for shot in current_shots:
            latest_shot = self.require_shot(shot.id)
            latest_shot.scene_profile_id = profile_by_shot.get(shot.id)
            latest_shot.use_scene_profile = False
            self.store.save_shot(latest_shot)
        return profiles

    async def generate_scene_reference(
        self,
        project_id: str,
        scene_profile_id: str,
        user_suggestions: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
    ) -> Asset:
        project = self.require_project(project_id)
        profile = next((item for item in project.scene_profiles if item.id == scene_profile_id), None)
        if profile is None:
            raise KeyError(f"Scene profile not found: {scene_profile_id}")
        prompt = (
            "影视场景设定母版，无人物。展示同一空间的主视角与清晰空间布局，固定墙面、门窗、桌椅、道具位置、材质、灯光和色彩。"
            f"场景：{profile.name}。{profile.description}。连续性要求：{profile.continuity_notes}。"
            f"统一风格：{self.project_style_text(project)}。{user_suggestions.strip()}"
        )
        image_gen = create_image_generator(image_provider=image_provider, model=image_model)
        output_dir = self.project_dir(project_id) / "scene_assets" / profile.id / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = Scene(id=1, duration=5, narrative=profile.description, visual_prompt=prompt, motion_prompt="")
        output = await image_gen.generate_image(
            scene,
            str(output_dir),
            None,
            seed=self._stable_seed(project_id, profile.id),
            character_description="",
            image_style=self.project_style_text(project),
            aspect_ratio=project.brief.aspect_ratio,
        )
        asset = self.register_existing_asset(
            project_id,
            Path(output),
            role=AssetRole.SCENE,
            name=f"{profile.name} · 场景母版",
            description=prompt,
        )
        # Attach the image to the newest project snapshot. Other AI work may
        # have updated storyboard metadata or another scene while this image was
        # rendering, and those updates must survive.
        latest_project = self.require_project(project_id)
        latest_profile = next(
            (item for item in latest_project.scene_profiles if item.id == scene_profile_id),
            None,
        )
        if latest_profile is None:
            raise KeyError(f"Scene profile not found after generation: {scene_profile_id}")
        latest_profile.reference_asset_ids = list(
            dict.fromkeys([*latest_profile.reference_asset_ids, asset.id])
        )
        latest_profile.approved = True
        self.store.save_project(latest_project)
        return asset

    async def generate_character_references(
        self,
        project_id: str,
        character_ids: list[str] | None = None,
        user_suggestions: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
        reference_asset_ids: list[str] | None = None,
        appearance_profile_id: str | None = None,
    ) -> list[Asset]:
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
            bound_ids = list(
                dict.fromkeys(
                    reference_asset_ids
                    or (appearance.reference_asset_ids if appearance else character.reference_asset_ids)
                )
            )
            reference_paths = [
                str(resolve_media_path(assets_by_id[asset_id].path))
                for asset_id in bound_ids
                if asset_id in assets_by_id
                and assets_by_id[asset_id].type == AssetType.IMAGE
                and resolve_media_path(assets_by_id[asset_id].path).is_file()
            ][:10]
            appearance_label = appearance.label if appearance else "默认形象"
            appearance_time = appearance.time_context if appearance else ""
            appearance_description = appearance.description if appearance else character.description
            appearance_wardrobe = appearance.wardrobe if appearance else character.wardrobe
            prompt = (
                "单一角色设定图，纯净中性背景，同一人物的正面、四分之三侧面和侧面视图，固定脸型、五官、发型、体型、服装、配饰和颜色，"
                "不出现第二个人物，不出现文字标签。严格保留所选参考图中同一人物的身份特征，并按档案补齐缺失角度与细节。"
                f"角色：{character.name}。形象档案：{appearance_label}。剧情时期/状态：{appearance_time}。"
                f"外貌：{appearance_description}。服装：{appearance_wardrobe}。"
                f"项目统一风格：{self.project_style_text(project)}。{user_suggestions.strip()}"
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
                image_style=self.project_style_text(project),
                aspect_ratio="1:1",
            )
            asset = self.register_existing_asset(
                project_id,
                Path(output),
                role=AssetRole.CHARACTER,
                name=f"{character.name} · {appearance_label} · AI 角色设定图",
                description=prompt,
            )
            asset.character_id = character.id
            asset.approved = True
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
                latest_appearance.reference_asset_ids = list(
                    dict.fromkeys([*latest_appearance.reference_asset_ids, *bound_ids, asset.id])
                )
                latest_appearance.approved = True
            else:
                latest_character.reference_asset_ids = list(
                    dict.fromkeys([*latest_character.reference_asset_ids, *bound_ids, asset.id])
                )
            self.store.save_project(latest_project)
            created.append(asset)
        return created

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
        provider = settings.REFERENCE_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = "ark" if settings.ARK_API_KEY else "glm" if settings.GLM_API_KEY else settings.LLM_PROVIDER
        payload = await create_llm_generator(provider).generate_json(
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
        minimum_shots = max(1, math.ceil(target / 8.0))
        prompt = f"""请根据用户目标时长和改写建议，重新整理一份可直接进入 AI 视频分镜阶段的完整剧情脚本。

【项目约束】
- 目标成片时长：{target:g} 秒
- 画幅：{project.brief.aspect_ratio}
- 视觉风格：{project.brief.visual_style or '未填写'}
- 叙事节奏：{project.brief.pacing or '未填写'}
- 最少可拆分镜头数：{minimum_shots}（任一生成片段不得超过 15 秒；高质量片段建议 4–8 秒，每镜只保留一个核心动作）
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
        clean = name.strip()
        aliases = [clean]
        without_title = re.sub(r"(?:同志|先生|女士|老师)$", "", clean).strip()
        if without_title and without_title not in aliases:
            aliases.append(without_title)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", without_title):
            aliases.append(without_title[0])
        return aliases

    def _resolve_dialogue_speaker_id(self, project: Project, explicit: str, *context: str) -> str | None:
        """Resolve one speaking character instead of leaving the video model to guess."""
        explicit_text = (explicit or "").strip()
        if explicit_text and explicit_text not in {"旁白", "画外音", "无", "无对白"}:
            for character in project.characters:
                if any(alias and (explicit_text == alias or alias in explicit_text) for alias in self._character_aliases(character.name)):
                    return character.id

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
        text = re.sub(r"^[【\[][^】\]]{1,30}[】\]]\s*[:：]\s*", "", text)
        text = re.sub(r"^[^，。！？!?]{1,20}\s*[:：]\s*", "", text)
        text = text.strip("'\"“”‘’ ")
        text = re.sub(r"[（(][^（）()]{1,50}[）)]\s*$", "", text).strip()
        return text.strip("'\"“”‘’ ")

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
    ) -> str:
        style = ProjectService.project_style_text(project)
        character_text = ProjectService.character_bible(project, character_ids, appearance_ids)
        return "，".join(item.strip("，。 ") for item in [style, character_text, description] if item)

    @staticmethod
    def character_visual_bible(
        project: Project,
        character_ids: list[str] | None = None,
        appearance_ids: dict[str, str] | None = None,
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
    ) -> str:
        """Compile a still-image prompt that represents this shot's opening frame."""
        # A keyframe prompt needs the concise visual direction, not the full
        # production bible (which can also contain camera, audio and video-only
        # constraints).  The detailed bible remains available to the general
        # storyboard and H3 prompts.
        style = ProjectService._compact_prompt_text(ProjectService.project_style_text(project), 600)
        character_text = ProjectService.character_visual_bible(project, character_ids, appearance_ids)
        composition = "，".join(item.strip("，。 ") for item in [shot_size, camera_angle, lens] if item)
        parts = [
            "静态分镜首帧，只呈现本镜开场的一个清晰瞬间",
            f"画面内容：{description.strip('，。 ')}" if description else "",
            f"构图与机位：{composition}" if composition else "",
            f"本镜出场角色：{character_text}" if character_text else "",
            f"统一视觉风格：{style}" if style else "",
            "保持人物身份、脸型、发型、服装、配饰、空间关系和光影一致；画面中不出现动作过程、运镜、转场、时长、对白、配音或音效说明",
        ]
        return "。".join(part.strip("。 ") for part in parts if part)

    @staticmethod
    def keyframe_character_ids(project: Project, shot: Shot, prompt: str = "") -> list[str]:
        """Return only characters that should condition this still image.

        Older storyboards assigned every project character whenever the model
        did not mention a name.  That made title cards and empty shots submit
        every portrait and the entire character bible.  Explicit name/speaker
        matches take priority; a deliberately selected subset is preserved.
        """
        project_ids = [character.id for character in project.characters]
        project_id_set = set(project_ids)
        combined_text = " ".join(
            item
            for item in [prompt, shot.scene_description, shot.narrative, shot.dialogue]
            if item
        )
        resolved = [
            character.id
            for character in project.characters
            if character.name and character.name in combined_text
        ]
        speaker_ids = [
            speaker_id
            for speaker_id in [
                shot.dialogue_speaker_id,
                *(turn.speaker_id for turn in shot.dialogue_turns),
            ]
            if speaker_id in project_id_set
        ]
        for speaker_id in speaker_ids:
            if speaker_id not in resolved:
                resolved.append(speaker_id)
        if resolved:
            return resolved

        selected = [character_id for character_id in shot.character_ids if character_id in project_id_set]
        if selected and set(selected) != project_id_set:
            return selected

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
        if not normalized_revision or normalized_revision == normalized_base:
            return base
        return f"{base}\n\n【本次修改建议｜高优先级】{revision}".strip()

    @classmethod
    def keyframe_reference_assets(
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
        profile = cls.scene_profile_for_shot(project, shot)
        if (
            profile is None
            and project.scene_consistency_mode == SceneConsistencyMode.STRICT
            and shot.scene_profile_id
        ):
            profile = next(
                (item for item in project.scene_profiles if item.id == shot.scene_profile_id),
                None,
            )
        ordered_ids = [
            *(profile.reference_asset_ids if profile else []),
            *allowed_character_ref_list,
            *(project.style_profile.reference_asset_ids if project.style_profile and project.style_profile.approved else []),
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
        # Clean dialogue needs Ref2VA: the stock FL2VA/I2V node accepts image
        # anchors but has no audio-conditioning input. Ref2VA can keep the
        # approved group keyframe as Picture 1 while the exact clean TTS track
        # drives the named speaker's mouth. This does not change shot size.
        if (
            shot.dialogue.strip()
            and settings.H3_POSTPROCESS_AUDIO
            and settings.H3_AUDIO_MODE == "clean_tts"
            and settings.TTS_PROVIDER != "disabled"
            and (shot.keyframe_asset_id or shot.image_path or shot.reference_asset_ids)
        ):
            return GenerationMode.R2V
        # A composed first frame is a much stronger spatial/identity anchor than
        # several independent character portraits.  Prefer it whenever it is
        # available; R2V remains available when the user selects it explicitly
        # or when no approved first frame exists.
        if shot.keyframe_asset_id or shot.image_path:
            return GenerationMode.I2V
        references = [asset for asset in assets if asset.id in shot.reference_asset_ids]
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

    def _reference_assets_in_shot_order(
        self,
        project: Project,
        shot: Shot,
        assets: list[Asset],
    ) -> list[Asset]:
        asset_map = {asset.id: asset for asset in assets}
        profile = self.scene_profile_for_shot(project, shot)
        if (
            profile is None
            and project.scene_consistency_mode == SceneConsistencyMode.STRICT
            and shot.scene_profile_id
        ):
            profile = next(
                (item for item in project.scene_profiles if item.id == shot.scene_profile_id),
                None,
            )
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
            if project.style_profile and project.style_profile.approved
            else []
        )
        continuity_input_id = self.continuity_input_asset_id(project, shot)
        ordered_ids = [
            asset_id
            for asset_id in [
                continuity_input_id,
                shot.keyframe_asset_id,
                *(profile.reference_asset_ids if profile else []),
                *character_refs,
                *style_refs,
                *shot.reference_asset_ids,
            ]
            if asset_id
        ]
        return [asset_map[asset_id] for asset_id in dict.fromkeys(ordered_ids) if asset_id in asset_map]

    def seedance_reference_assets(
        self,
        shot: Shot,
        assets: list[Asset],
        project: Project | None = None,
    ) -> list[Asset]:
        """Return the exact numbered-material order used by Seedance.

        The approved full-shot keyframe is deliberately first, followed by
        explicitly bound character/style/motion references.  The compiler and
        API request call this same method so 图片1/视频1/音频1 can never drift.
        """
        project = project or self.require_project(shot.project_id)
        supported = {AssetType.IMAGE, AssetType.VIDEO, AssetType.AUDIO}
        # Seedance's strict first/last-frame mode cannot be mixed with any
        # reference image/video/audio.  A continuous shot must prioritize the
        # previous real tail frame, so expose that image as the sole input.  Its
        # character, wardrobe and scene constraints remain in the compiled text
        # prompt, while the actual frame carries the complete visual state.
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS:
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
        return [
            asset
            for asset in self._reference_assets_in_shot_order(project, shot, assets)
            if asset.type in supported
        ]

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
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS and continuity_input_id in asset_ids:
            return continuity_input_id
        if shot.keyframe_asset_id in asset_ids:
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
            manifests.append(f"{label}（{asset.description or asset.name}）")
            if asset.type == AssetType.IMAGE:
                image_numbers[asset.id] = number

        visible_characters = [
            character for character in project.characters if character.id in shot.character_ids
        ]
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
            if pictures:
                definitions.append(
                    f"将图片{pictures[0]}中{traits or character.name}的角色定义为“{character.name}”"
                )
            elif image_numbers:
                definitions.append(
                    f"将图片1中的“{character.name}”定义为“{character.name}”，固定特征为{traits or '保持图片1外观'}"
                )
            else:
                definitions.append(f"将{traits or character.name}的角色定义为“{character.name}”")

        style = self._compact_prompt_text(self.project_style_text(project), 260)
        scene = self._compact_prompt_text(shot.scene_description or shot.narrative, 360)
        motion = self._compact_prompt_text(self._compact_subject_motion(shot), 420)
        camera_motion = self._compact_prompt_text(shot.camera_motion or "固定镜头", 80)
        camera = "，".join(item for item in [shot.shot_size, shot.camera_angle, shot.lens, camera_motion] if item)
        audio_design = self._compact_prompt_text(shot.audio_design, 160)

        dialogue = ""
        if shot.dialogue_turns:
            turns = []
            for turn in shot.dialogue_turns:
                speaker = next((item for item in project.characters if item.id == turn.speaker_id), None)
                if turn.text.strip():
                    turns.append(f"{speaker.name if speaker else '画外音'}说：“{self.spoken_dialogue_text(turn.text)}”")
            if turns:
                dialogue = "对白按顺序发生：" + "；".join(turns) + "。每句话只由标明角色说出，其他人物闭嘴并自然聆听"
        elif shot.dialogue.strip():
            speaker = next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)
            if speaker:
                dialogue = (
                    f"唯一发言者是“{speaker.name}”，“{speaker.name}”自然说：“{self.spoken_dialogue_text(shot.dialogue)}”。"
                    "其余人物不说话、嘴唇闭合，只做克制的聆听反应；保持当前多人景别，不强制切单人近景"
                )
            else:
                dialogue = f"画外音说：“{self.spoken_dialogue_text(shot.dialogue)}”。画面内人物不做说话口型"

        reference_instructions: list[str] = []
        if image_numbers:
            first_frame_id = self.seedance_first_frame_asset_id(project, shot, assets)
            if first_frame_id and first_frame_id in image_numbers:
                if len(refs) == 1 and refs[0].id == first_frame_id:
                    reference_instructions.append(
                        "以图片1作为严格首帧，开场画面必须继承其中角色身份、人物数量、服装、站位、场景、景别和光影"
                    )
                else:
                    reference_instructions.append(
                        "参考图片1作为开场构图，开场优先继承其中角色身份、人物数量、服装、站位、场景、景别和光影"
                    )
            reference_instructions.append("参考上述图片锁定角色形象、服装、场景与视觉风格")
        if counters[AssetType.VIDEO]:
            reference_instructions.append("参考上述视频的主体动作、节奏或运镜，但不要复制无关内容")
        if counters[AssetType.AUDIO]:
            reference_instructions.append("参考上述音频的音色、对白或声音氛围")

        profile = self.scene_profile_for_shot(project, shot)
        continuity = ""
        if profile:
            continuity = (
                f"【场景连续性】本镜使用可选场景档案“{profile.name}”：{profile.description}；"
                f"必须保持：{profile.continuity_notes or '空间布局、固定陈设、材质与光线'}"
            )
        if shot.continuity_mode == ShotContinuityMode.CONTINUOUS:
            continuity = (
                continuity + "。" if continuity else "【连续长镜头】"
            ) + "图片1是上一镜真实尾帧；从该画面无跳切地继续动作、视线、站位、光线和声音，不复述上一镜事件"

        parts = [
            "【素材定义】" + "；".join(definitions) if definitions else "",
            "【可用素材】" + "；".join(manifests) if manifests else "",
            "【参考要求】" + "；".join(reference_instructions) if reference_instructions else "",
            continuity,
            f"【镜头事件】{shot.narrative or scene}",
            f"【场景与开场状态】{scene}" if scene else "",
            f"【主体动作】{motion}" if motion else "",
            f"【镜头语言】{camera}。本镜只使用一种主要运镜，动作衔接平缓连续",
            f"【对白与声音】{dialogue}。{audio_design or '生成与画面匹配的清晰环境声'}" if dialogue else f"【声音】{audio_design or '无对白，生成克制清晰的环境声'}",
            f"【视觉风格与画质】{style or '延续项目统一风格'}；高清，细节丰富，色彩自然，光影柔和，人物五官与肢体稳定",
            (
                "【约束】严格保持同一角色身份、脸型、发型、服装与人物数量；禁止人物复制、融合、重影、畸形肢体、"
                "身份互换、说话人错位、重复台词、乱码、字幕、Logo 和水印。"
                "按事件自然安排节奏，不在同一镜头叠加多次推拉摇移"
            ),
        ]
        return "\n".join(part for part in parts if part).strip()

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
            if shot.dialogue.strip() and not shot.dialogue_turns:
                turns = self.resolve_dialogue_turns(project, shot)
                shot.dialogue_turns = [DialogueTurn.model_validate(turn) for turn in turns]
                if len(turns) == 1:
                    shot.dialogue_speaker_id = turns[0]["speaker_id"]
            shot.seedance_prompt = self.compile_seedance_prompt(project, shot, assets)
            shot.seedance_prompt_version = "seedance-2.0-official-2026-08"
            shot.seedance_prompt_source_revision = shot.content_revision
            shot.version += 1
            saved.append(self.store.save_shot(shot))
        return saved

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
        if shot.h3_prompt_skill_output.strip():
            return shot.h3_prompt_skill_output.strip()
        mode = self.resolve_generation_mode(shot, assets)
        dialogue_audio_bound = bool(
            mode == GenerationMode.R2V
            and shot.dialogue.strip()
            and settings.H3_POSTPROCESS_AUDIO
            and settings.H3_AUDIO_MODE == "clean_tts"
            and settings.TTS_PROVIDER != "disabled"
        )
        style = self._compact_prompt_text(self.project_style_text(project), 240)
        narrative = self._compact_prompt_text(shot.narrative or shot.scene_description, 260)
        motion = self._compact_subject_motion(shot)
        camera = "，".join(item for item in [shot.shot_size, shot.camera_angle, shot.lens, shot.camera_motion] if item)
        audio_design = self._clean_audio_design(shot.audio_design)
        if shot.dialogue.strip():
            if len(shot.dialogue_turns) > 1 and not shot.dialogue_speaker_id:
                turn_summary = "；".join(
                    f"{next((item.name for item in project.characters if item.id == turn.speaker_id), '旁白')}说“{self._compact_prompt_text(turn.text, 80)}”"
                    for turn in shot.dialogue_turns
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
                    voice = f"，声音特征为{speaker.voice_description}" if speaker.voice_description else "，使用清晰标准普通话"
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
            audio = f"声音：无对白，{audio_design or '仅保留轻微自然环境声'}，无背景音乐"

        details = [
            (
                "纯净画面约束：画面内不出现任何字幕、台词文字、标题、标识、水印、数字、界面元素或乱码；"
                "不得把对白或声音转写成可见文字"
            ),
            f"画面风格：{style}" if style else "",
            f"场景与事件：{narrative}" if narrative else "",
            f"动作：{motion}" if motion else "",
            f"镜头：{camera}" if camera else "",
            audio,
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
                    tag_notes.append(f"<Picture {picture_index}> 用作{asset.role.value}参考：{asset.description or asset.name}")
                elif asset.type == AssetType.VIDEO:
                    video_index += 1
                    tag_notes.append(f"<Video {video_index}> 用作动作与镜头参考：{asset.description or asset.name}")
                elif asset.type == AssetType.AUDIO:
                    audio_index += 1
                    tag_notes.append(f"<Audio {audio_index}> 用作声音参考：{asset.description or asset.name}")
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
                dialogue_binding = (
                    f"<Audio 1> is the directly reused clean Chinese dialogue track for {speaker_label}. "
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
            issues.append(f"时长 {shot.duration_seconds:g}s 超过高稳定区间 8s，系统将自动拆段续帧生成")
        if len(action_terms) >= 3:
            issues.append(f"检测到 {len(action_terms)} 个动作节点，系统将按动作阶段自动拆段")
        if shot.dialogue.strip() and not shot.dialogue_speaker_id:
            issues.append("对白未手动绑定发言者，将按剧情动词推断；建议在分镜页确认")
        if shot.camera_motion and sum(keyword in shot.camera_motion for keyword in ("推", "拉", "摇", "移", "跟", "环绕")) > 1:
            issues.append("包含多次运镜，高质量生成会收敛为单向慢运镜")
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

    def h3_segment_plan(self, project: Project, shot: Shot, assets: list[Asset]) -> list[dict[str, object]]:
        """Plan stable sequential I2V segments while keeping one user-facing shot."""
        dialogue_turns = self.resolve_dialogue_turns(project, shot)
        longest_spoken_seconds = max(
            (len(str(turn["text"])) / 4.0 + 0.35 for turn in dialogue_turns),
            default=0.0,
        )
        if len(dialogue_turns) > 1 or longest_spoken_seconds > settings.H3_MAX_SEGMENT_SECONDS:
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
                if durations[longest_index] <= settings.H3_MAX_SEGMENT_SECONDS + 0.001:
                    break
                left, right = split_text(str(visual_turns[longest_index]["text"]))
                if not left or not right:
                    break
                original = visual_turns[longest_index]
                visual_turns[longest_index:longest_index + 1] = [
                    {"speaker_id": original.get("speaker_id"), "text": left},
                    {"speaker_id": original.get("speaker_id"), "text": right},
                ]

            weights = [max(1.0, len(str(turn["text"])) / 4.0 + 0.35) for turn in visual_turns]
            weight_total = sum(weights)
            durations = [shot.duration_seconds * weight / weight_total for weight in weights]
            durations[-1] += shot.duration_seconds - sum(durations)
            result: list[dict[str, object]] = []
            for turn, duration in zip(visual_turns, durations, strict=False):
                segment = shot.model_copy(deep=True)
                segment.h3_prompt_skill_output = ""
                delivery_duration = round(max(0.35, duration), 3)
                generation_duration = max(5.0, min(settings.H3_MAX_SEGMENT_SECONDS, delivery_duration))
                segment.duration_seconds = generation_duration
                segment.dialogue = str(turn["text"])
                segment.dialogue_speaker_id = turn.get("speaker_id")
                segment.dialogue_turns = [DialogueTurn.model_validate(turn)]
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
                "has_dialogue": bool(shot.dialogue.strip()),
                "dialogue": self.spoken_dialogue_text(shot.dialogue),
                "speaker_id": shot.dialogue_speaker_id,
            }]

        phases = self._motion_phases(shot.subject_motion or shot.narrative)
        action_count = len(re.findall(r"说|介绍|要求|询问|回答|起身|转身|走|跑|拿|掏|放|开门|关门|拉开|推开|坐下", shot.subject_motion or shot.narrative))
        duration_count = max(1, math.ceil(shot.duration_seconds / max(4.0, settings.H3_MAX_SEGMENT_SECONDS)))
        action_segments = min(len(phases), max(1, action_count)) if phases else 1
        segment_count = max(duration_count, action_segments)
        if not settings.H3_AUTO_SEGMENT_COMPLEX_SHOTS or segment_count == 1:
            return [{
                "duration": shot.duration_seconds,
                "generation_duration": max(5.0, shot.duration_seconds),
                "prompt": self.compile_h3_prompt(project, shot, assets),
                "motion": shot.subject_motion,
                "has_dialogue": bool(shot.dialogue.strip()),
                "dialogue": self.spoken_dialogue_text(shot.dialogue),
                "speaker_id": shot.dialogue_speaker_id,
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
                settings.H3_MAX_SEGMENT_SECONDS,
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
            segment.subject_motion = (
                f"当前短段只完成这一项动作：{phase}。动作幅度克制，人物数量、脸、服装和空间位置连续稳定；"
                "禁止增加新动作、转身、快速走位和第二次运镜；结尾保持清晰稳定姿态"
            )
            # Do not carry the whole-shot narrative into every continuation.
            # A preceding dialogue verb (for example "D介绍秦绍辉") causes H3
            # to invent another speaker and even burn pseudo-subtitles into a
            # later silent segment.  Each call must describe only its own beat.
            segment.narrative = f"本短段只表现：{phase}"
            segment.camera_motion = "固定镜头或极慢单向微移"
            if index != speaking_index:
                segment.dialogue = ""
                visible_names = [
                    item.name for item in project.characters if item.id in shot.character_ids
                ]
                silence_subjects = "、".join(visible_names) if visible_names else "画面内所有人物"
                segment.subject_motion = (
                    f"当前短段只完成这一项无对白动作：{phase}。{silence_subjects}全程嘴唇闭合，"
                    "不说话、不做说话口型、不抬手模拟发言，只做与当前动作直接相关的克制反应；"
                    "画面中不显示对白、字幕、标题、标识、水印或乱码；人物数量、脸、服装和空间位置"
                    "逐帧连续稳定；禁止增加新动作、快速走位、第二次运镜、角色互换、重影和复制人物；"
                    "结尾保持清晰稳定姿态"
                )
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
        if shot.dialogue.strip() and not shot.dialogue_turns:
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
            shot.scene_profile_id,
            shot.use_scene_profile,
            shot.continuity_mode,
            shot.continuity_source_shot_id,
            tuple(shot.character_ids),
            shot.shot_size,
            shot.camera_angle,
            shot.lens,
            shot.camera_motion,
            shot.subject_motion,
            shot.transition,
            shot.audio_design,
            shot.visual_prompt,
        )

    def update_shot(self, project_id: str, incoming: Shot) -> Shot:
        """Save an edited shot and immediately refresh all dependent prompts.

        Direct edits to a prompt field are preserved. When storyboard content
        changes, generated prompt variants are invalidated and rebuilt from the
        new source fields so the production page never shows stale content.
        """
        project = self.require_project(project_id)
        existing = self.require_shot(incoming.id)
        if incoming.project_id != project_id:
            raise ValueError("Shot project mismatch")
        content_changed = self._shot_content_signature(existing) != self._shot_content_signature(incoming)
        keyframe_authored = incoming.keyframe_prompt != existing.keyframe_prompt
        h3_authored = incoming.video_prompt != existing.video_prompt
        seedance_authored = incoming.seedance_prompt != existing.seedance_prompt
        if content_changed:
            incoming.content_revision = existing.content_revision + 1
            character_ids = self.keyframe_character_ids(project, incoming, incoming.scene_description)
            if not keyframe_authored:
                incoming.keyframe_prompt = self.compile_keyframe_prompt(
                    project,
                    incoming.scene_description or incoming.narrative,
                    character_ids,
                    incoming.character_appearance_ids,
                    shot_size=incoming.shot_size,
                    camera_angle=incoming.camera_angle,
                    lens=incoming.lens,
                )
            incoming.keyframe_prompt_source_revision = incoming.content_revision
            assets = self.store.list_assets(project_id)
            if not seedance_authored:
                incoming.seedance_prompt = self.compile_seedance_prompt(project, incoming, assets)
            incoming.seedance_prompt_source_revision = incoming.content_revision
            if not h3_authored:
                incoming.h3_prompt_skill_output = ""
                incoming.video_prompt_source = self.compile_base_video_prompt(
                    project,
                    incoming.subject_motion,
                    incoming.narrative,
                )
                incoming.video_prompt = self.compile_h3_prompt(project, incoming, assets)
            incoming.h3_prompt_source_revision = incoming.content_revision
            incoming.approval_status = ApprovalStatus.DRAFT
        elif h3_authored:
            incoming.video_prompt_source = incoming.video_prompt
        incoming.version = max(existing.version, incoming.version) + 1
        saved = self.store.save_shot(incoming)
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
        """Redo one shot while locking only its ordinal, id and duration."""
        project = self.require_project(project_id)
        shot = self.require_shot(shot_id)
        suggestions = user_suggestions.strip()
        if not suggestions:
            raise ValueError("请填写本镜修改建议")
        characters = [
            {"id": item.id, "name": item.name, "description": item.description, "wardrobe": item.wardrobe}
            for item in project.characters
        ]
        payload = await create_llm_generator(settings.LLM_PROVIDER).generate_json(
            "你是影视分镜导演。只重做指定单镜；镜号和时长是硬约束，人物、对白、场景、动作与构图可按建议改变。",
            f"""重做下面一个分镜。保持 shot_id、ordinal、duration_seconds 原值；其他内容按用户建议重新设计。

项目剧情：{project.brief.story}
统一风格：{self.project_style_text(project)}
可用角色：{json.dumps(characters, ensure_ascii=False)}
原镜头：{json.dumps(shot.model_dump(mode='json'), ensure_ascii=False)}
用户建议（最高优先级）：{suggestions}

严格返回 JSON：
{{"title":"","narrative":"","dialogue":"","dialogue_speaker_name":"","scene_description":"","character_names":[],"shot_size":"","camera_angle":"","lens":"","camera_motion":"","subject_motion":"","transition":"","audio_design":"","visual_prompt":""}}""",
        )
        by_name = {character.name: character for character in project.characters}
        names = [str(value) for value in payload.get("character_names", [])]
        speaker = by_name.get(str(payload.get("dialogue_speaker_name") or ""))
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
        shot.character_ids = [by_name[name].id for name in names if name in by_name]
        shot.dialogue_speaker_id = speaker.id if speaker else None
        shot.dialogue_turns = (
            [DialogueTurn(speaker_id=speaker.id, text=shot.dialogue)]
            if speaker and shot.dialogue.strip()
            else []
        )
        # The update service compares against persisted data. Preserve these
        # hard constraints even if the model returned other values.
        shot.id = shot_id
        shot.project_id = project_id
        saved = self.update_shot(project_id, shot)
        targets = [target for target in dict.fromkeys(prompt_targets or []) if target in {"h3", "seedance"}]
        if "seedance" in targets:
            self.generate_seedance_prompts(project_id, [shot_id])
        if "h3" in targets:
            await self.generate_h3_prompts(project_id, [shot_id], h3_skill_id, suggestions)
        return self.require_shot(shot_id)

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

        assets = self.store.list_assets(project_id)
        asset_map = {asset.id: asset for asset in assets}
        provider = settings.LLM_PROVIDER
        llm = create_llm_generator(provider)
        suggestions = user_suggestions.strip()
        semaphore = asyncio.Semaphore(3)

        async def generate_one(shot: Shot) -> tuple[Shot, str]:
            mode = self.resolve_generation_mode(shot, assets)
            references = self._reference_assets_in_shot_order(project, shot, assets)
            counters = {AssetType.IMAGE: 0, AssetType.VIDEO: 0, AssetType.AUDIO: 0}
            reference_manifest: list[str] = []
            if mode == GenerationMode.R2V and shot.dialogue.strip():
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
                    f"<{label} {counters[asset.type]}>: {asset.role.value} reference named {asset.name}; "
                    f"{asset.description or 'use only for its declared reference role'}"
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
            prompt = f"""Create one final MiniMax H3 prompt for the following approved shot.

PROJECT
- title: {project.brief.title}
- aspect ratio: {project.brief.aspect_ratio}
- approved visual style: {self.project_style_text(project) or 'derive only from the selected style Skill'}
- project continuity bible: {project.style_bible or 'none'}
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
- sound design: {shot.audio_design or 'natural diegetic sound only'}
- transition context: {shot.transition}
- visible characters: {json.dumps(visible_characters, ensure_ascii=False)}
- available references in exact API order: {json.dumps(reference_manifest, ensure_ascii=False)}
- current approved first-frame prompt: {shot.keyframe_prompt or 'none'}
- current general storyboard prompt: {shot.visual_prompt or 'none'}
- user instructions for this generation: {suggestions or 'none'}

Do not copy the full project bible or every character into the result. Include only details needed for this shot. Keep exactly the authored character count, speaker ownership, dialogue and reference labels. For I2V, <Picture 1> is the approved first frame. For R2V, use only labels listed in the manifest. Return the final prompt in the required JSON field."""
            async with semaphore:
                payload = await llm.generate_json(
                    h3_prompt_system_instruction(skill, mode),
                    prompt,
                )
            output = str(payload.get("video_prompt") or "").strip()
            if not output:
                raise ValueError(f"镜头 {shot.ordinal} 的 H3 Prompt 生成结果为空")
            if len(output) > 6000:
                raise ValueError(f"镜头 {shot.ordinal} 的 H3 Prompt 超过 6000 字符，请缩短建议后重试")
            required = (
                ("subject_definitions:", "detailed_description:")
                if mode == GenerationMode.R2V
                else ("integrated_multimodal_description:", "overall_soundscape:")
            )
            if not all(section in output for section in required):
                raise ValueError(f"镜头 {shot.ordinal} 的生成结果不符合 {mode.value.upper()} 官方结构")
            return shot, output

        generated = await asyncio.gather(*(generate_one(shot_map[shot_id]) for shot_id in selected_ids))
        saved: list[Shot] = []
        for shot, output in generated:
            shot.h3_prompt_skill_id = skill.id
            shot.h3_prompt_skill_version = skill.version
            shot.h3_prompt_skill_output = output
            shot.video_prompt_source = output
            shot.video_prompt = output
            shot.h3_prompt_source_revision = shot.content_revision
            shot.resolved_generation_mode = self.resolve_generation_mode(shot, assets)
            shot.version += 1
            saved.append(self.store.save_shot(shot))
        return saved

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
            shots = [shot for shot in shots if shot.id in selected]
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
            self.store.save_shot(shot)

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

                effective_prompt = self.merge_keyframe_suggestions(
                    effective_prompt,
                    user_suggestions,
                )
                profile = self.scene_profile_for_shot(project, shot)
                if profile:
                    effective_prompt = (
                        f"{effective_prompt}\n\n【场景档案｜必须保持】{profile.name}："
                        f"{profile.description}；{profile.continuity_notes}"
                    ).strip()
                character_description = self.character_visual_bible(
                    project,
                    effective_character_ids,
                    shot.character_appearance_ids,
                )
                image_style = self._compact_prompt_text(self.project_style_text(project), 600)
                logger.info(
                    "Keyframe request: shot=%s prompt_chars=%s characters=%s references=%s revision=%s",
                    shot.ordinal,
                    len(effective_prompt),
                    len(effective_character_ids),
                    len(reference_paths),
                    revision_mode,
                )

                shot.image_status = "processing"
                self.store.save_shot(shot)
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
                        character_description=character_description,
                        image_style=image_style,
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
                    shot.keyframe_asset_id = asset.id
                    shot.image_path = str(path)
                    shot.image_status = "completed"
                    shot.keyframe_revision_last_suggestion = user_suggestions
                    shot.keyframe_revision_suggestion_draft = ""
                    shot.keyframe_prompt_source_revision = shot.content_revision
                except Exception:
                    shot.image_status = "failed"
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
                finally:
                    self.store.save_shot(shot)

        results = await asyncio.gather(*(generate(shot) for shot in shots), return_exceptions=True)
        failures = [
            f"镜头 {shot.ordinal}: {result}"
            for shot, result in zip(shots, results, strict=False)
            if isinstance(result, BaseException)
        ]
        project.status = ProjectStatus.KEYFRAMES_REVIEW
        self.store.save_project(project)
        if failures:
            raise RuntimeError("部分分镜图生成失败；成功结果已保留。" + "；".join(failures))
        return self.store.list_shots(project_id)

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
            base_video_prompt = str(scene.get("motion_prompt") or "")
            shot = Shot(
                project_id=project.id,
                ordinal=ordinal,
                title=f"镜头 {ordinal}",
                narrative=str(scene.get("narrative") or ""),
                duration_seconds=duration,
                scene_description=str(scene.get("visual_prompt") or ""),
                visual_prompt=str(scene.get("visual_prompt") or ""),
                keyframe_prompt=str(scene.get("keyframe_prompt") or scene.get("visual_prompt") or ""),
                subject_motion=str(scene.get("motion_prompt") or ""),
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
