from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.domain import CharacterProfile, ProjectBrief, Shot, VoiceEvent
from src.video_workflow.h3_prompt_skills import (
    OFFICIAL_H3_SKILLS_COMMIT,
    get_h3_prompt_skill,
    list_h3_prompt_skills,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


class H3PromptSkillsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.previous_projects_dir = settings.PROJECTS_DIR
        settings.PROJECTS_DIR = root / "projects"
        self.store = ProjectStore(root / "state.sqlite3")
        self.service = ProjectService(self.store)

    def tearDown(self) -> None:
        settings.PROJECTS_DIR = self.previous_projects_dir
        self.temp.cleanup()

    def test_catalog_exposes_general_and_eight_official_styles(self) -> None:
        skills = list_h3_prompt_skills()
        self.assertEqual(len(skills), 9)
        self.assertEqual(skills[0]["id"], "h3-prompt-writing")
        self.assertEqual(skills[0]["source_commit"], OFFICIAL_H3_SKILLS_COMMIT)
        self.assertEqual(get_h3_prompt_skill("paper-collage-explainer-generator").version, "0.3.9")
        with self.assertRaisesRegex(ValueError, "未知的 H3 Prompt Skill"):
            get_h3_prompt_skill("missing-style")

    def test_director_upgrade_keeps_saved_h3_text_for_review(self) -> None:
        project = self.service.create_project(ProjectBrief(title="旧 H3", story="人物操作设备"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            subject_motion="人物启动设备；指示灯亮起；人物确认结果",
            h3_prompt_skill_output="OLD_SLOW_PROMPT",
            h3_director_version="",
        )

        compiled = self.service.compile_h3_prompt(project, shot, [])

        self.assertEqual(compiled, "OLD_SLOW_PROMPT")
        self.assertEqual(shot.h3_director_version, "")

    async def test_selected_skill_generates_and_persists_h3_prompt(self) -> None:
        project = self.service.create_project(
            ProjectBrief(
                title="纸艺测试",
                story="一张纸折成小船。",
                target_duration_seconds=5,
                visual_style="温暖手作风",
            )
        )
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                title="纸船成形",
                narrative="纸张逐步折成一艘小船",
                subject_motion="纸张沿折痕逐帧折叠",
                keyframe_prompt="桌面上的白纸",
            )
        )
        captured: dict[str, str] = {}

        class FakeLLM:
            async def generate_json(self, system_prompt: str, user_prompt: str, reference_images=None):
                captured["system"] = system_prompt
                captured["user"] = user_prompt
                return {
                    "video_prompt": (
                        "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                        "(from [Shot 1]) is fully referenced.\n\n"
                        "integrated_multimodal_description: [Shot 1] Layered cardstock folds frame by frame.\n"
                        "overall_soundscape: Soft paper creases and a light tabletop tap.\n"
                        "non_diegetic_music: None."
                    )
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            generated = await self.service.generate_h3_prompts(
                project.id,
                [shot.id],
                "papercraft-stop-motion-explainer",
                "保持白纸颜色",
            )

        self.assertEqual(len(generated), 1)
        saved = self.service.require_shot(shot.id)
        self.assertEqual(saved.h3_prompt_skill_id, "papercraft-stop-motion-explainer")
        self.assertEqual(saved.h3_prompt_skill_version, "0.6.5")
        self.assertIn("integrated_multimodal_description:", saved.h3_prompt_skill_output)
        self.assertEqual(saved.video_prompt, saved.h3_prompt_skill_output)
        self.assertEqual(self.service.compile_h3_prompt(project, saved, []), saved.h3_prompt_skill_output)
        self.assertIn("handcrafted papercraft stop motion", captured["system"])
        self.assertIn("every 2–4 seconds", captured["system"])
        self.assertIn("system_vo, narration and offscreen", captured["system"])
        self.assertIn("hard speech budget", captured["system"])
        self.assertIn("exactly one <d>[Chinese]", captured["system"])
        self.assertIn("Translate each approved Chinese voice anchor", captured["system"])
        self.assertIn("Only the exact contents of <d> tags may be vocalized", captured["system"])
        self.assertNotIn("inside the exact approved voice anchors", captured["system"])
        self.assertIn("H3 is allowed to render authored", captured["system"])
        self.assertIn("text_policy=post_overlay does not prohibit H3", captured["system"])
        self.assertIn("hard speech budget: at most", captured["user"])
        self.assertIn("approved voice anchors", captured["user"])
        self.assertIn("timed visual beats", captured["user"])
        self.assertIn("text policy: post_overlay", captured["user"])
        self.assertIn("H3 may render exact on-screen copy explicitly authored", captured["user"])
        self.assertIn("no-text post_overlay rule applies to still-frame and Seedance", captured["user"])
        self.assertIn("保持白纸颜色", captured["user"])

    def test_h3_allows_authored_copy_while_seedance_remains_text_free(self) -> None:
        project = self.service.create_project(ProjectBrief(title="分引擎文字策略", story="电脑发布任务"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="电脑屏幕显示“紧急任务”，人物抬头确认",
            subject_motion="屏幕亮起后，人物抬头",
            text_policy="post_overlay",
        )

        h3_prompt = self.service.compile_h3_prompt(project, shot, [])
        seedance_prompt = self.service.compile_seedance_prompt(project, shot, [])

        self.assertIn("H3 可显示分镜明确写出的", h3_prompt)
        self.assertIn("逐字保留原文", h3_prompt)
        self.assertIn("紧急任务", h3_prompt)
        self.assertIn("画面内不生成任何可读文字", seedance_prompt)
        self.assertIn("所有信息文字统一后期叠加", seedance_prompt)

    def test_h3_none_policy_still_forbids_visible_text(self) -> None:
        project = self.service.create_project(ProjectBrief(title="H3 禁字策略", story="人物看向屏幕"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="人物看向屏幕",
            text_policy="none",
        )

        h3_prompt = self.service.compile_h3_prompt(project, shot, [])

        self.assertIn("画面内不出现任何字幕", h3_prompt)
        self.assertIn("不得把对白或声音转写成可见文字", h3_prompt)

    async def test_completed_prompts_remain_saved_when_later_batch_item_fails(self) -> None:
        project = self.service.create_project(ProjectBrief(title="批量持久化", story="先开门，再关灯。"))
        first = self.store.save_shot(Shot(project_id=project.id, ordinal=1, title="开门", narrative="人物打开门"))
        second = self.store.save_shot(Shot(project_id=project.id, ordinal=2, title="关灯", narrative="人物关闭灯"))

        class PartiallyFailingLLM:
            async def generate_json(self, _system_prompt: str, user_prompt: str, reference_images=None):
                if "ordinal/title: 2 /" in user_prompt:
                    await asyncio.sleep(0.02)
                    raise RuntimeError("second prompt failed")
                return {
                    "video_prompt": (
                        "For the target video, <Picture 1> is fully referenced.\n"
                        "integrated_multimodal_description: A person opens the door.\n"
                        "overall_soundscape: Door hinge and room tone.\n"
                        "non_diegetic_music: None."
                    )
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=PartiallyFailingLLM()):
            with self.assertRaisesRegex(RuntimeError, "second prompt failed"):
                await self.service.generate_h3_prompts(
                    project.id,
                    [first.id, second.id],
                    "h3-prompt-writing",
                )

        saved_first = self.service.require_shot(first.id)
        saved_second = self.service.require_shot(second.id)
        self.assertIn("integrated_multimodal_description:", saved_first.h3_prompt_skill_output)
        self.assertEqual(saved_first.video_prompt, saved_first.h3_prompt_skill_output)
        self.assertEqual(saved_first.h3_prompt_source_revision, saved_first.content_revision)
        self.assertEqual(saved_second.h3_prompt_skill_output, "")

    async def test_generation_finishing_after_source_change_is_archived(self) -> None:
        project = self.service.create_project(ProjectBrief(title="生成期间编辑", story="人物开门"))
        shot = self.store.save_shot(Shot(project_id=project.id, ordinal=1, narrative="人物开门"))
        service = self.service
        output = "integrated_multimodal_description: A person opens the door.\noverall_soundscape: Room tone."

        class LLM:
            async def generate_json(self, *args, **kwargs):
                latest = service.require_shot(shot.id)
                latest.content_revision += 1
                latest.narrative = "人物关门"
                service.store.save_shot(latest)
                return {"video_prompt": output}

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=LLM()):
            with self.assertRaisesRegex(ValueError, "已保存在 H3 历史"):
                await self.service.generate_h3_prompts(project.id, [shot.id], "h3-prompt-writing")
        self.assertEqual(self.service.require_shot(shot.id).narrative, "人物关门")
        history = self.store.list_h3_prompt_history(project.id, shot.id)
        self.assertEqual(history[0]["prompt"], output)
        self.assertEqual(history[0]["source_revision"], 1)

    async def test_paid_h3_output_replaces_spoken_chinese_anchor_with_silent_english_direction(self) -> None:
        project = self.service.create_project(ProjectBrief(title="声音锚点", story="李强抱怨"))
        speaker = CharacterProfile(
            name="李强",
            description="22岁男性",
            voice_description="年轻男性声线，毛躁不耐烦，节奏偏快",
        )
        project.characters = [speaker]
        self.store.save_project(project)
        shot = self.store.save_shot(Shot(
            project_id=project.id,
            ordinal=1,
            title="抱怨",
            narrative="李强敷衍检查",
            subject_motion="李强扫过杆身",
            character_ids=[speaker.id],
            voice_events=[VoiceEvent(
                kind="character",
                speaker_id=speaker.id,
                speaker_name=speaker.name,
                text="真是麻烦。",
                start_seconds=1,
                end_seconds=3,
                lip_sync=True,
            )],
        ))

        class FakeLLM:
            async def generate_json(self, system_prompt: str, user_prompt: str, reference_images=None):
                return {
                    "video_prompt": (
                        "For the target video, <Picture 1> is fully referenced.\n"
                        "integrated_multimodal_description: Li Qiang complains. "
                        f"<d>[Chinese] 真是麻烦。</d> (李强 voice anchor: {speaker.voice_description})\n"
                        "overall_soundscape: One young male voice.\n"
                        "non_diegetic_music: None.\n"
                        f"approved_voice_identity: 李强: {speaker.voice_description}; preserve this voice."
                    )
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            await self.service.generate_h3_prompts(project.id, [shot.id], "h3-prompt-writing")

        saved = self.service.require_shot(shot.id)
        self.assertNotIn("approved_voice_identity", saved.h3_prompt_skill_output)
        self.assertNotIn("voice anchor:", saved.h3_prompt_skill_output)
        self.assertNotIn(speaker.voice_description, saved.h3_prompt_skill_output)
        self.assertIn("speech_control (NON-SPOKEN PRODUCTION INSTRUCTION)", saved.h3_prompt_skill_output)
        self.assertIn("Only the exact text inside existing <d>...</d> tags may become speech", saved.h3_prompt_skill_output)
        self.assertIn("silent_voice_direction_1", saved.h3_prompt_skill_output)
        self.assertIn("young adult male voice", saved.h3_prompt_skill_output)
        self.assertIn("a restless, impatient edge", saved.h3_prompt_skill_output)
        self.assertEqual(saved.h3_prompt_skill_output.count("<d>[Chinese] 真是麻烦。</d>"), 1)

    def test_energetic_heroine_voice_direction_is_compact_english_metadata(self) -> None:
        project = self.service.create_project(ProjectBrief(title="活力女主", story="张小差接到任务"))
        speaker = CharacterProfile(
            name="张小差",
            description="20岁年轻女性",
            voice_description=(
                "清亮活泼的年轻女声，情绪起伏大：摸鱼时语气蔫蔫拖腔，看到高绩效任务时语速瞬间变快发亮，"
                "被怼时会卡顿语塞，劝人时诚恳急切带点咋呼，没有公职人员的架子，像普通邻家女生的语气"
            ),
        )
        project.characters = [speaker]
        event = VoiceEvent(
            kind="character",
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            text="高危？大单啊！",
            start_seconds=7.16,
            end_seconds=9.06,
            lip_sync=True,
        )

        contract = self.service.h3_silent_voice_contract(project, [event])

        self.assertIn("lively, energetic delivery", contract)
        self.assertIn("quicker, brighter delivery", contract)
        self.assertIn("sparkling, excited lift", contract)
        self.assertIn("informal, approachable girl-next-door", contract)
        self.assertIn("lip-sync only during the tagged words", contract)
        self.assertNotIn(speaker.voice_description, contract)


if __name__ == "__main__":
    unittest.main()
