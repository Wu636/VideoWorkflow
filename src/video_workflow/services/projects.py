from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import shutil
from pathlib import Path
from uuid import uuid4

from src.video_workflow.config import settings
from src.video_workflow.core.orchestrator import WorkflowOrchestrator, create_llm_generator
from src.video_workflow.domain import (
    ApprovalStatus,
    Asset,
    AssetRole,
    AssetType,
    DialogueTurn,
    GenerationMode,
    Project,
    ProjectAnalysisDraft,
    ProjectBrief,
    ProjectStatus,
    ScriptRewriteDraft,
    Shot,
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

    def update_project(self, project: Project) -> Project:
        existing = self.store.get_project(project.id)
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
        return self.store.save_project(project)

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
    ) -> list[Shot]:
        project = self.require_project(project_id)
        user_suggestions = user_suggestions.strip()
        if count_mode == "ai":
            count = await self.recommend_shot_count(project_id, user_suggestions) if user_suggestions else (
                project.ai_recommended_shot_count or await self.recommend_shot_count(project_id)
            )
        else:
            count = shot_count or max(1, min(500, math.ceil(project.brief.target_duration_seconds / 8.0)))
        # Production-quality H3 clips are kept short.  The data model still
        # accepts up to 15 s for compatibility, but new storyboards target at
        # most 8 s per generated clip to avoid identity/body drift.
        count = max(count, math.ceil(project.brief.target_duration_seconds / 8.0))
        assets = self.store.list_assets(project_id)
        reference = next(
            (str(resolve_media_path(asset.path)) for asset in assets if asset.type == AssetType.IMAGE and asset.role in {AssetRole.CHARACTER, AssetRole.STYLE}),
            None,
        )
        orchestrator = WorkflowOrchestrator()
        storyboard = await orchestrator.llm.generate_storyboard(
            topic=project.brief.story,
            count=count,
            reference_image=reference,
            include_dialogue=True,
            character_description=self.character_bible(project),
            image_style=project.style_bible or project.brief.visual_style,
            user_suggestions=user_suggestions,
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
                h3_steps=20,
                render_frames=h3_frames_for_seconds(duration),
            )
            shots.append(shot)
        project.status = ProjectStatus.STORYBOARD_DRAFT
        project.storyboard_version += 1
        self.store.save_project(project)
        return self.store.replace_shots(project_id, shots)

    async def recommend_shot_count(self, project_id: str, user_suggestions: str = "") -> int:
        project = self.require_project(project_id)
        minimum = max(1, math.ceil(project.brief.target_duration_seconds / 8.0))
        fallback = max(minimum, min(500, math.ceil(project.brief.target_duration_seconds / 6.5)))
        provider = settings.SHOT_COUNT_PROVIDER
        if provider == "auto":
            provider = settings.BRIEF_ANALYSIS_PROVIDER
        if provider == "auto":
            provider = settings.LLM_PROVIDER
        orchestrator = WorkflowOrchestrator(provider)
        prompt = f"""分析下面的剧情脚本，为总时长 {project.brief.target_duration_seconds:g} 秒的视频决定分镜数量。
每个高质量生成片段建议 4–8 秒；每镜只保留一个核心动作和一名发言者。出现说话后走动、取放物品、开门、转身等连续动作时必须继续拆镜，不要机械平均。
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
        project.ai_recommended_shot_count = count
        self.store.save_project(project)
        return count

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
    def character_bible(project: Project, character_ids: list[str] | None = None) -> str:
        selected = set(character_ids) if character_ids is not None else None
        parts = []
        for character in project.characters:
            if selected is not None and character.id not in selected:
                continue
            details = "，".join(item for item in [character.description, character.wardrobe, character.voice_description] if item)
            parts.append(f"{character.name}：{details}" if details else character.name)
        return "；".join(parts)

    @staticmethod
    def compile_visual_prompt(project: Project, description: str, character_ids: list[str] | None = None) -> str:
        style = project.style_bible or project.brief.visual_style
        character_text = ProjectService.character_bible(project, character_ids)
        return "，".join(item.strip("，。 ") for item in [style, character_text, description] if item)

    @staticmethod
    def character_visual_bible(project: Project, character_ids: list[str] | None = None) -> str:
        """Return only appearance/wardrobe details suitable for image models."""
        selected = set(character_ids) if character_ids is not None else None
        parts: list[str] = []
        for character in project.characters:
            if selected is not None and character.id not in selected:
                continue
            details = "，".join(item for item in [character.description, character.wardrobe] if item)
            parts.append(f"{character.name}：{details}" if details else character.name)
        return "；".join(parts)

    @staticmethod
    def compile_keyframe_prompt(
        project: Project,
        description: str,
        character_ids: list[str] | None = None,
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
        style = ProjectService._compact_prompt_text(
            project.brief.visual_style or project.style_bible,
            600,
        )
        character_text = ProjectService.character_visual_bible(project, character_ids)
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

    @staticmethod
    def keyframe_reference_assets(
        project: Project,
        shot: Shot,
        assets: list[Asset],
        character_ids: list[str],
    ) -> list[Asset]:
        """Filter inherited character portraits to this shot's visible cast."""
        asset_map = {asset.id: asset for asset in assets}
        allowed_character_refs = {
            asset_id
            for character in project.characters
            if character.id in character_ids
            for asset_id in character.reference_asset_ids
        }
        known_character_refs = {
            asset_id
            for character in project.characters
            for asset_id in character.reference_asset_ids
        }
        selected: list[Asset] = []
        for asset_id in shot.reference_asset_ids:
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
            project.brief.visual_style,
            motion,
            f"画面叙事：{narrative}" if narrative else "",
        ]
        return "。".join(part.strip("。 ") for part in parts if part)

    def resolve_generation_mode(self, shot: Shot, assets: list[Asset]) -> GenerationMode:
        if shot.generation_mode != GenerationMode.AUTO:
            return shot.generation_mode
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

    @staticmethod
    def _visual_only_dialogue_phase(value: str) -> str:
        """Turn a spoken beat into a silent, unambiguous visual action.

        H3 may assign a quoted line to the wrong face or render fragments of the
        line as pseudo-subtitles.  Clean-audio delivery therefore sends speech
        to TTS only and gives the video model a silent gesture equivalent.
        """
        text = re.sub(r"[“\"『「][^”\"』」]*[”\"』」]", "", value or "")
        replacements = (
            (r"嘴唇(?:微动|张合)?(?:并)?说出[^，。；]*", "保持嘴唇闭合"),
            (r"开口(?:说|介绍|询问|回答|要求)?[^，。；]*", "保持嘴唇闭合"),
            (r"说出[^，。；]*", "做出克制示意"),
            (r"介绍(?:台词|对方|人物|情况)?", "抬手示意目标人物"),
            (r"询问[^，。；]*", "看向对方等待回应"),
            (r"回答[^，。；]*", "轻微点头回应"),
            (r"要求[^，。；]*", "抬手指示目标位置"),
            (r"说话|对白|台词|口型同步", "安静表演"),
        )
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"[，；、]\s*[，；、]+", "，", text)
        return text.strip("，；。 ") or "发言者做克制的介绍手势，其他人物安静聆听"

    @staticmethod
    def _reference_assets_in_shot_order(shot: Shot, assets: list[Asset]) -> list[Asset]:
        asset_map = {asset.id: asset for asset in assets}
        ordered_ids = [
            asset_id
            for asset_id in [shot.keyframe_asset_id, *shot.reference_asset_ids]
            if asset_id
        ]
        return [asset_map[asset_id] for asset_id in dict.fromkeys(ordered_ids) if asset_id in asset_map]

    def compile_h3_prompt(self, project: Project, shot: Shot, assets: list[Asset]) -> str:
        if shot.h3_prompt_skill_output.strip():
            return shot.h3_prompt_skill_output.strip()
        mode = self.resolve_generation_mode(shot, assets)
        style = self._compact_prompt_text(project.brief.visual_style or project.style_bible, 240)
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
                    audio = (
                        f"对白表演：唯一发言者明确为{speaker.name}{voice}，只由{speaker.name}嘴唇自然张合并说“{spoken}”；"
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
            refs = self._reference_assets_in_shot_order(shot, assets)
            if not refs and shot.keyframe_asset_id:
                keyframe = next((asset for asset in assets if asset.id == shot.keyframe_asset_id), None)
                if keyframe:
                    refs = [keyframe]
            picture_index = video_index = audio_index = 0
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
            details = [
                *tag_notes,
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
                visual_only = settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE == "clean_tts"
                if visual_only:
                    segment.subject_motion = (
                        f"{speaker_name}做克制的发言示意动作，画面内所有人物全程嘴唇闭合，"
                        "不说话、不做说话口型；人物站位、脸、服装和背景逐帧稳定；禁止显示对白、字幕、"
                        "标题、水印或乱码；禁止走位、转身、挥手、角色互换、重影、复制人物和额外肢体"
                    )
                    segment.narrative = f"本短段只表现{speaker_name}做克制示意，其他人物安静聆听"
                    segment.dialogue = ""
                    segment.dialogue_turns = []
                    segment.dialogue_speaker_id = None
                else:
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
                        # The H3 visual prompt is deliberately silent in
                        # clean_tts mode, but the delivery layer still needs
                        # the authored line and its speaker to synthesize and
                        # place the replacement dialogue track.
                        "dialogue": str(turn["text"]),
                        "speaker_id": turn.get("speaker_id"),
                    }
                )
            return result

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
            segment_has_dialogue = index == speaking_index and bool(shot.dialogue.strip())
            segment_dialogue = self.spoken_dialogue_text(shot.dialogue) if segment_has_dialogue else ""
            segment_speaker_id = shot.dialogue_speaker_id if segment_has_dialogue else None
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
            elif segment_has_dialogue and settings.H3_POSTPROCESS_AUDIO and settings.H3_AUDIO_MODE == "clean_tts":
                visual_phase = self._visual_only_dialogue_phase(phase)
                visible_names = [
                    item.name for item in project.characters if item.id in shot.character_ids
                ]
                silence_subjects = "、".join(visible_names) if visible_names else "画面内所有人物"
                segment.narrative = f"本短段只表现：{visual_phase}"
                segment.subject_motion = (
                    f"当前短段只完成这一项无对白动作：{visual_phase}。{silence_subjects}全程嘴唇闭合，"
                    "不说话、不做说话口型；D同志只做克制的介绍手势，秦绍辉与C同志安静聆听；"
                    "画面中不显示对白、字幕、标题、标识、水印或乱码；人物数量、脸、服装和空间位置"
                    "逐帧连续稳定；禁止抢动作、角色互换、重影、复制人物和额外肢体"
                )
                segment.dialogue = ""
                segment.dialogue_turns = []
                segment.dialogue_speaker_id = None
            result.append(
                {
                    "duration": delivery_duration,
                    "generation_duration": generation_duration,
                    "prompt": self.compile_h3_prompt(project, segment, assets),
                    "motion": phase,
                    "has_dialogue": segment_has_dialogue,
                    "dialogue": segment_dialogue,
                    "speaker_id": segment_speaker_id,
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
            references = self._reference_assets_in_shot_order(shot, assets)
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

            visible_characters = [
                {
                    "id": character.id,
                    "name": character.name,
                    "appearance": character.description,
                    "wardrobe": character.wardrobe,
                    "voice": character.voice_description,
                    "reference_assets": [
                        asset_map[asset_id].name for asset_id in character.reference_asset_ids if asset_id in asset_map
                    ],
                    "speaks_in_this_shot": character.id == shot.dialogue_speaker_id,
                }
                for character in project.characters
                if character.id in shot.character_ids
            ]
            prompt = f"""Create one final MiniMax H3 prompt for the following approved shot.

PROJECT
- title: {project.brief.title}
- aspect ratio: {project.brief.aspect_ratio}
- approved visual style: {project.brief.visual_style or project.style_bible or 'derive only from the selected style Skill'}
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
                character_description = self.character_visual_bible(
                    project,
                    effective_character_ids,
                )
                image_style = self._compact_prompt_text(
                    project.brief.visual_style or project.style_bible,
                    600,
                )
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
                    duration=max(4, min(12, round(shot.duration_seconds))),
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
                h3_steps=20,
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
    def _stable_seed(project_id: str, shot_id: str) -> int:
        digest = hashlib.sha256(f"{project_id}:{shot_id}".encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def approve_storyboard(self, project_id: str, approved: bool, comment: str = "") -> Project:
        project = self.require_project(project_id)
        project.status = ProjectStatus.STORYBOARD_APPROVED if approved else ProjectStatus.STORYBOARD_DRAFT
        for shot in self.store.list_shots(project_id):
            shot.approval_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.CHANGES_REQUESTED
            self.store.save_shot(shot)
        return self.store.save_project(project)
