from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.domain import (
    AssetRole,
    CharacterProfile,
    GenerationMode,
    ProjectBrief,
    SeedanceReferenceMode,
    Shot,
    VoiceEvent,
)
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

    def test_catalog_exposes_general_official_styles_and_short_ad_workflow(self) -> None:
        skills = list_h3_prompt_skills()
        self.assertEqual(len(skills), 10)
        self.assertEqual(skills[0]["id"], "h3-prompt-writing")
        self.assertEqual(skills[0]["source_commit"], OFFICIAL_H3_SKILLS_COMMIT)
        self.assertEqual(get_h3_prompt_skill("paper-collage-explainer-generator").version, "0.3.9")
        self.assertEqual(get_h3_prompt_skill("short-video-talking-ad-generator").name, "舞台互动型抓眼广告")
        self.assertEqual(skills[3]["source_url"], "")
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
        self.assertIn("用手工纸艺定格动画解释概念", captured["system"])
        self.assertIn("普通内容每2至4秒", captured["system"])
        self.assertIn("system_vo、narration和offscreen", captured["system"])
        self.assertIn("口播硬预算", captured["system"])
        self.assertIn("只在其指定时间出现一次 <d>[Chinese]", captured["system"])
        self.assertIn("声音锚点压缩成中文", captured["system"])
        self.assertIn("只有 <d> 标签内部文字可以发声", captured["system"])
        self.assertIn("H3可以渲染分镜明确要求", captured["system"])
        self.assertIn("text_policy=post_overlay不禁止H3", captured["system"])
        self.assertIn("口播硬预算：全部声音事件合计最多", captured["user"])
        self.assertIn("已确认声音锚点", captured["user"])
        self.assertIn("分段视觉时间轴", captured["user"])
        self.assertIn("文字策略：post_overlay", captured["user"])
        self.assertIn("H3可以在指定时间渲染", captured["user"])
        self.assertIn("最终提示词正文必须使用中文", captured["user"])
        self.assertIn("保持白纸颜色", captured["user"])

    async def test_explicit_h3_prompt_input_mode_selects_exclusive_template_and_submission_mode(self) -> None:
        project = self.service.create_project(ProjectBrief(title="输入模式", story="主持人口播"))
        first_path = Path(self.temp.name) / "first.png"
        last_path = Path(self.temp.name) / "last.png"
        character_path = Path(self.temp.name) / "host.png"
        for path in (first_path, last_path, character_path):
            path.write_bytes(b"image")
        first = self.service.register_existing_asset(project.id, first_path, AssetRole.KEYFRAME, "舞台首帧")
        last = self.service.register_existing_asset(project.id, last_path, AssetRole.LAST_FRAME, "人物收束尾帧")
        character = self.service.register_existing_asset(project.id, character_path, AssetRole.CHARACTER, "主持人角色图")
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                generation_mode=GenerationMode.R2V,
                narrative="主持人从舞台中央走向观众并完成口播",
                keyframe_asset_id=first.id,
                last_frame_asset_id=last.id,
                reference_asset_ids=[character.id],
            )
        )
        calls: list[tuple[str, str]] = []

        class FakeLLM:
            async def generate_json(self, system_prompt: str, user_prompt: str, reference_images=None):
                calls.append((system_prompt, user_prompt))
                if "严格使用以下字段顺序" in system_prompt:
                    return {
                        "video_prompt": (
                            "subject_definitions: Host from <Picture 1>.\n"
                            "summary: Stage pitch.\nretention_analysis: Strong hook.\n"
                            "detailed_description: The host crosses the stage.\n"
                            "overall_soundscape: Clear room tone.\nnon_diegetic_music: None."
                        )
                    }
                return {
                    "video_prompt": (
                        "For the target video, at 0.00 seconds, <Picture 1> is fully referenced.\n"
                        "integrated_multimodal_description: The host crosses the stage and lands on the supplied last frame.\n"
                        "overall_soundscape: Clear room tone.\nnon_diegetic_music: None."
                    )
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            frame_result = await self.service.generate_h3_prompts(
                project.id, [shot.id], "h3-prompt-writing", input_mode="frame"
            )
            reference_result = await self.service.generate_h3_prompts(
                project.id, [shot.id], "h3-prompt-writing", input_mode="reference"
            )

        self.assertEqual(frame_result[0].generation_mode, GenerationMode.I2V)
        self.assertIn("API另行提供已确认的last_frame", calls[0][0])
        self.assertIn("仅使用帧：已确认的first_frame和last_frame", calls[0][1])
        self.assertIn("按API实际顺序排列的可用参考：[]", calls[0][1])
        self.assertNotIn("主持人角色图", calls[0][1])
        self.assertEqual(reference_result[0].generation_mode, GenerationMode.R2V)
        self.assertIn("严格使用以下字段顺序", calls[1][0])
        self.assertIn("仅使用参考：", calls[1][1])
        self.assertIn("主持人角色图", calls[1][1])
        saved = self.service.require_shot(shot.id)
        self.assertEqual(saved.generation_mode, GenerationMode.R2V)
        self.assertEqual(saved.resolved_generation_mode, GenerationMode.R2V)

    def test_explicit_seedance_prompt_input_mode_compiles_distinct_grammars(self) -> None:
        project = self.service.create_project(ProjectBrief(title="Seedance 输入模式", story="主持人口播"))
        first_path = Path(self.temp.name) / "seedance-first.png"
        style_path = Path(self.temp.name) / "seedance-style.png"
        first_path.write_bytes(b"image")
        style_path.write_bytes(b"image")
        first = self.service.register_existing_asset(project.id, first_path, AssetRole.KEYFRAME, "严格首帧")
        style = self.service.register_existing_asset(project.id, style_path, AssetRole.STYLE, "广告风格参考")
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                narrative="主持人走向观众",
                keyframe_asset_id=first.id,
                reference_asset_ids=[style.id],
            )
        )

        frame = self.service.generate_seedance_prompts(project.id, [shot.id], "frame")[0]
        self.assertEqual(frame.seedance_reference_mode, SeedanceReferenceMode.STRICT_FIRST_FRAME)
        self.assertIn("@图片1是严格首帧", frame.seedance_prompt)
        self.assertNotIn("广告风格参考", frame.seedance_prompt)

        reference = self.service.generate_seedance_prompts(project.id, [shot.id], "reference")[0]
        self.assertEqual(reference.seedance_reference_mode, SeedanceReferenceMode.MULTIMODAL_REFERENCE)
        self.assertNotIn("@图片1是严格首帧", reference.seedance_prompt)
        self.assertIn("广告风格参考", reference.seedance_prompt)

    async def test_short_ad_prompt_replaces_paraphrased_dialogue_with_exact_copy(self) -> None:
        advertising_copy = "最高保障三百万元，立即点击咨询。"
        project = self.service.create_project(
            ProjectBrief(
                title="快口播原文校验",
                story=advertising_copy,
                speech_pacing="short_ad",
            )
        )
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                generation_mode="r2v",
                duration_seconds=5,
                dialogue_rate_percent=55,
                voice_events=[
                    VoiceEvent(
                        kind="narration",
                        speaker_name="口播主持人",
                        text=advertising_copy,
                        start_seconds=0.2,
                        end_seconds=4.8,
                        lip_sync=False,
                    )
                ],
            )
        )

        class ParaphrasingLLM:
            async def generate_json(self, *_args, **_kwargs):
                return {
                    "video_prompt": (
                        "subject_definitions: One presenter.\n"
                        "detailed_description: The presenter says <d>[Chinese] 最高保障很多，快来咨询。</d>"
                    )
                }

        with patch(
            "src.video_workflow.services.projects.create_llm_generator",
            return_value=ParaphrasingLLM(),
        ):
            generated = await self.service.generate_h3_prompts(
                project.id,
                [shot.id],
                "short-video-talking-ad-generator",
            )

        prompt = generated[0].h3_prompt_skill_output
        self.assertNotIn("最高保障很多", prompt)
        self.assertEqual(prompt.count(advertising_copy), 1)
        self.assertIn(f"<d>[Chinese] {advertising_copy}</d>", prompt)
        self.assertIn("舞台广告多机位控制", prompt)
        self.assertIn("本段不要求一镜到底", prompt)
        self.assertIn("主持人不在画面内", prompt)

    def test_stage_ad_sequence_roles_balance_energy_and_continuity(self) -> None:
        roles = [self.service.short_ad_stage_role(ordinal, 7) for ordinal in range(1, 8)]

        self.assertEqual(
            [role["name"] for role in roles],
            [
                "开场空间钩子",
                "主持人中景跟拍",
                "主持人情绪近景",
                "观众肩后视角",
                "主持人与观众同框互动",
                "大屏证据点",
                "情绪高点与 CTA",
            ],
        )
        self.assertTrue(all("runtime" in role for role in roles))
        contract = self.service.short_ad_stage_storyboard_contract(7)
        self.assertIn("不等于一镜到底", contract)
        self.assertIn("纯观众中近景或近景", contract)
        self.assertIn("避免无动机跨越180度", contract)
        self.assertIn("首帧与尾帧", contract)

    def test_stage_ad_runtime_contract_refreshes_after_last_frame_is_added(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="舞台广告", story="主持人介绍保险", speech_pacing="short_ad")
        )
        first = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                scene_description="蓝色保险发布会舞台，主持人在中央，LED 大屏在身后",
            )
        )
        self.store.save_shot(Shot(project_id=project.id, ordinal=2))
        self.store.save_shot(Shot(project_id=project.id, ordinal=3))

        initial = self.service.apply_h3_stage_ad_contract(project, first, "base prompt")
        self.assertIn("从已提供首帧准确起步", initial)
        self.assertNotIn("本段为首尾帧I2V", initial)

        first.last_frame_asset_id = "last-frame-asset"
        refreshed = self.service.apply_h3_stage_ad_contract(project, first, initial)
        self.assertEqual(refreshed.count("舞台广告多机位控制"), 1)
        self.assertIn("本段为首尾帧I2V", refreshed)
        self.assertIn("准确落到已提供尾帧构图", refreshed)

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
                if "生成段序号/标题：2 /" in user_prompt:
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
        self.assertIn("声音控制（非口播制作指令）", saved.h3_prompt_skill_output)
        self.assertIn("只有现有<d>...</d>标签内部的逐字内容可以发声", saved.h3_prompt_skill_output)
        self.assertIn("声音表演说明1", saved.h3_prompt_skill_output)
        self.assertIn("青年男性声线", saved.h3_prompt_skill_output)
        self.assertIn("带一点毛躁和不耐烦", saved.h3_prompt_skill_output)
        self.assertEqual(saved.h3_prompt_skill_output.count("<d>[Chinese] 真是麻烦。</d>"), 1)

    def test_energetic_heroine_voice_direction_is_compact_chinese_metadata(self) -> None:
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

        self.assertIn("表达活泼有能量", contract)
        self.assertIn("立刻提高语速、音高和明亮度", contract)
        self.assertIn("自然亲近的邻家口吻", contract)
        self.assertIn("只在标签台词期间同步口型", contract)
        self.assertNotIn(speaker.voice_description, contract)


if __name__ == "__main__":
    unittest.main()
