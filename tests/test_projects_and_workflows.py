from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import (
    AssetRole,
    CharacterAppearanceProfile,
    CharacterProfile,
    GenerationMode,
    JobStatus,
    ProjectBrief,
    SceneProfile,
    Shot,
    ShotContinuityMode,
    StyleProfile,
)
from src.video_workflow.integrations.comfyui import (
    ComfyUIClient,
    H3WorkflowBuilder,
    H3WorkflowRequest,
    h3_diffusion_model,
    h3_frames_for_seconds,
    h3_text_encoder,
)
from src.video_workflow.generators.image import (
    GrsaiImageGenerator,
    _decode_grsai_response,
    _generated_image_suffix,
    grsai_image_aspect_ratio,
    grsai_image_endpoint,
)
from src.video_workflow.generators.llm import _build_user_suggestions_text
from src.video_workflow.media_paths import resolve_media_path
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.services.audio import DialogueAudioService
from src.video_workflow.storage import ProjectStore
from src.video_workflow.runtime_settings import RuntimeSettingsManager
from src.video_workflow.types import Scene, Storyboard


class ProjectAndWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = (settings.PROJECTS_DIR, settings.OUTPUT_DIR)
        settings.PROJECTS_DIR = self.root / "projects"
        settings.OUTPUT_DIR = self.root / "legacy"
        self.store = ProjectStore(self.root / "state.sqlite3")
        self.service = ProjectService(self.store)

    def tearDown(self) -> None:
        settings.PROJECTS_DIR, settings.OUTPUT_DIR = self.previous
        self.temp.cleanup()

    def test_duration_mapping_and_workflow_builders(self) -> None:
        self.assertEqual(_generated_image_suffix(b"\xff\xd8\xff\xe0"), ".jpg")
        self.assertEqual(_generated_image_suffix(b"\x89PNG\r\n\x1a\n"), ".png")
        self.assertEqual(h3_frames_for_seconds(1), 124)
        self.assertEqual(h3_frames_for_seconds(5), 124)
        self.assertEqual(h3_frames_for_seconds(15), 362)
        self.assertEqual((h3_frames_for_seconds(9) - 5) % 17, 0)
        self.assertEqual(grsai_image_endpoint("gpt-image-2"), "/v1/draw/completions")
        self.assertEqual(grsai_image_endpoint("nano-banana-fast"), "/v1/draw/nano-banana")
        self.assertEqual(grsai_image_aspect_ratio("gpt-image-2", "16:9"), "1672x941")
        self.assertEqual(grsai_image_aspect_ratio("nano-banana-fast", "16:9"), "16:9")

        builder = H3WorkflowBuilder()
        requirements = builder.preflight_requirements()
        self.assertTrue({"LoadVideo", "GetVideoComponents", "LoadAudio"}.issubset(requirements["nodes"]))
        self.assertTrue(requirements["optional_models"])
        self.assertTrue(all(name not in requirements["models"] for name in requirements["optional_models"]))
        i2v = builder.build(
            H3WorkflowRequest(
                mode=GenerationMode.I2V,
                prompt="camera pushes in",
                width=1344,
                height=768,
                frames=362,
                seed=42,
                output_prefix="test/i2v",
                first_frame_filename="inputs/first.png",
                turbo=False,
                steps=8,
                scheduler="karras",
                denoise=0.85,
                lora_strength=0.9,
                low_vram=True,
                shift_video=10.5,
                shift_audio=2.5,
            )
        )
        self.assertEqual(i2v["7"]["inputs"]["image"], "inputs/first.png")
        self.assertEqual(i2v["8"]["inputs"]["length"], 362)
        self.assertEqual(i2v["11"]["inputs"]["noise_seed"], 42)
        self.assertEqual(i2v["10"]["inputs"], {"model": ["1", 0], "scheduler": "karras", "steps": 8, "denoise": 0.85})
        self.assertNotIn("17", i2v)
        self.assertNotIn("5", i2v)
        self.assertEqual(i2v["6"], {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}})

        precise_i2v = builder.build(
            H3WorkflowRequest(
                mode=GenerationMode.I2V,
                prompt="controlled model A/B",
                width=1344,
                height=768,
                frames=124,
                seed=43,
                output_prefix="test/precise-i2v",
                first_frame_filename="inputs/first.png",
                turbo=False,
                steps=25,
                diffusion_model=h3_diffusion_model("pruned_bf16", GenerationMode.I2V),
                text_encoder_model=h3_text_encoder("int8"),
            )
        )
        self.assertEqual(precise_i2v["1"]["inputs"]["unet_name"], "minimax_h3_fl2va_pruned_bf16.safetensors")
        self.assertEqual(precise_i2v["2"]["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        self.assertNotIn("17", precise_i2v)

        r2v = builder.build(
            H3WorkflowRequest(
                mode=GenerationMode.R2V,
                prompt="<Picture 1> and <Picture 2>, use <Video 1>",
                width=768,
                height=1344,
                frames=243,
                seed=7,
                output_prefix="test/r2v",
                reference_images=["a.png", "b.png"],
                reference_videos=["motion.mp4"],
                reference_audios=["voice.wav"],
            )
        )
        self.assertEqual(r2v["8"]["inputs"]["ref_image_1"], ["101", 0])
        self.assertEqual(r2v["8"]["inputs"]["ref_image_size"], "match")
        self.assertIn("ref_video_1", r2v["8"]["inputs"])
        self.assertIn("ref_audio_1", r2v["8"]["inputs"])
        self.assertEqual(r2v["123"]["class_type"], "GetVideoComponents")
        self.assertEqual(r2v["17"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(r2v["17"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(r2v["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(r2v["6"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(
            self.service._normalize_motion_duration("总时长约7秒。角色向前走，停留2秒。", 4.667),
            "总时长约4.67秒。角色向前走，停留2秒。",
        )

    def test_grsai_gpt_image_success_payload_and_event_stream_are_parsed(self) -> None:
        success_payload = {
            "id": "l1-f7915a15-c9d4-4b38-942e-21237c76f672",
            "task_id": "",
            "url": "",
            "progress": 100,
            "status": "succeeded",
            "failure_reason": "",
            "error": "",
            "results": [
                {
                    "url": "https://files.example.invalid/generated.png",
                    "width": 0,
                    "height": 0,
                }
            ],
        }
        generator = object.__new__(GrsaiImageGenerator)

        parsed_json = _decode_grsai_response(httpx.Response(200, json=success_payload), "test")
        self.assertEqual(generator._extract_task_state(parsed_json)["status"], "succeeded")
        self.assertEqual(
            generator._extract_result_url(parsed_json),
            "https://files.example.invalid/generated.png",
        )

        stream_body = (
            'event: progress\ndata: {"id":"l1-test","status":"running","progress":50}\n\n'
            f"event: result\ndata: {json.dumps(success_payload)}\n\n"
            "data: [DONE]\n"
        )
        parsed_stream = _decode_grsai_response(
            httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream_body),
            "test",
        )
        self.assertEqual(parsed_stream["id"], success_payload["id"])
        self.assertEqual(
            generator._extract_result_url(parsed_stream),
            "https://files.example.invalid/generated.png",
        )

    def test_media_paths_survive_host_and_container_roots(self) -> None:
        media = settings.PROJECTS_DIR / "project_portable" / "deliveries" / "final.mp4"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"video")
        persisted_in_container = "/app/outputs/projects/project_portable/deliveries/final.mp4"
        self.assertEqual(resolve_media_path(persisted_in_container), media.resolve())

    def test_storyboard_generation_forwards_user_suggestions(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="导演建议", story="角色在雨夜寻找失踪的信件", target_duration_seconds=8)
        )
        captured: dict[str, object] = {}

        class FakeLLM:
            async def generate_storyboard(self, **kwargs: object) -> Storyboard:
                captured.update(kwargs)
                return Storyboard(
                    topic=str(kwargs["topic"]),
                    scenes=[
                        Scene(
                            id=1,
                            narrative="角色找到信件",
                            visual_prompt="雨夜近景",
                            motion_prompt="镜头缓慢推进，角色拾起信件",
                            duration=8,
                        )
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        suggestion = "开头先给信件特写，减少对白，结尾停在角色震惊的近景。"
        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 1, "manual", suggestion))

        self.assertIn(suggestion, str(captured["user_suggestions"]))
        self.assertIn("镜头数量最高优先级硬约束", str(captured["user_suggestions"]))
        self.assertEqual(len(shots), 1)
        prompt_text = _build_user_suggestions_text(suggestion)
        self.assertIn("用户本次分镜建议｜高优先级", prompt_text)
        self.assertIn(suggestion, prompt_text)

    def test_storyboard_generation_preserves_parallel_project_updates(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="并行分镜", story="人物进入固定谈话室", target_duration_seconds=8)
        )
        parallel_profile = SceneProfile(name="谈话室", description="固定米白墙面和深木桌")

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                latest = self_test.service.require_project(project.id)
                latest.scene_profiles = [parallel_profile]
                self_test.store.save_project(latest)
                return Storyboard(
                    topic="测试",
                    scenes=[
                        Scene(
                            id=1,
                            narrative="人物落座",
                            visual_prompt="谈话室中景",
                            motion_prompt="人物缓慢落座",
                            duration=8,
                        )
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        self_test = self
        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            asyncio.run(self.service.generate_storyboard(project.id, 1, "manual"))

        saved_project = self.service.require_project(project.id)
        self.assertEqual([profile.id for profile in saved_project.scene_profiles], [parallel_profile.id])

    def test_scene_profile_analysis_rejects_obsolete_storyboard_mapping(self) -> None:
        project = self.service.create_project(ProjectBrief(title="并行场景", story="谈话", target_duration_seconds=8))
        original = Shot(project_id=project.id, ordinal=1, narrative="旧分镜")
        self.store.save_shot(original)
        replacement = Shot(project_id=project.id, ordinal=1, narrative="并行生成的新分镜")

        class FakeLLM:
            async def generate_json(self, *_: object, **__: object) -> dict[str, object]:
                self_test.store.replace_shots(project.id, [replacement])
                return {
                    "scenes": [
                        {
                            "name": "旧场景",
                            "description": "不应绑定到新分镜",
                            "continuity_notes": "保持不变",
                            "shot_ordinals": [1],
                        }
                    ]
                }

        self_test = self
        with patch(
            "src.video_workflow.services.projects.create_llm_generator",
            return_value=FakeLLM(),
        ):
            with self.assertRaisesRegex(ValueError, "场景识别期间分镜已更新"):
                asyncio.run(self.service.generate_scene_profiles(project.id))

        saved = self.store.list_shots(project.id)
        self.assertEqual([shot.id for shot in saved], [replacement.id])
        self.assertEqual(self.service.require_project(project.id).scene_profiles, [])

    def test_scene_reference_preserves_parallel_project_updates(self) -> None:
        project = self.service.create_project(ProjectBrief(title="并行母版", story="谈话", target_duration_seconds=8))
        profile = SceneProfile(name="谈话室", description="米白墙面和深木桌")
        project.scene_profiles = [profile]
        self.store.save_project(project)

        class FakeImageGenerator:
            async def generate_image(self, scene: object, output_dir: str, *_: object, **__: object) -> str:
                latest = self_test.service.require_project(project.id)
                latest.ai_recommended_shot_count = 77
                self_test.store.save_project(latest)
                output = Path(output_dir) / "scene.png"
                output.write_bytes(b"scene")
                return str(output)

        self_test = self
        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            asset = asyncio.run(self.service.generate_scene_reference(project.id, profile.id))

        saved_project = self.service.require_project(project.id)
        saved_profile = next(item for item in saved_project.scene_profiles if item.id == profile.id)
        self.assertEqual(saved_project.ai_recommended_shot_count, 77)
        self.assertIn(asset.id, saved_profile.reference_asset_ids)
        self.assertTrue(saved_profile.approved)

    def test_character_reference_generation_accepts_selected_image_and_appearance(self) -> None:
        appearance = CharacterAppearanceProfile(
            label="少年期",
            time_context="十二岁",
            description="圆脸短发少年",
            wardrobe="浅青短袄",
        )
        character = CharacterProfile(
            name="阿明",
            description="成年形象",
            wardrobe="深色长袍",
            appearance_profiles=[appearance],
        )
        project = self.service.create_project(ProjectBrief(title="角色补全", story="阿明的一生"))
        project.characters = [character]
        self.store.save_project(project)
        reference_path = self.root / "uploaded-character.png"
        reference_path.write_bytes(b"selected-reference")
        reference = self.service.register_existing_asset(
            project.id,
            reference_path,
            AssetRole.CHARACTER,
            "用户上传参考图",
        )
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(
                self,
                _scene: object,
                output_dir: str,
                reference_images: str | None,
                **kwargs: object,
            ) -> str:
                captured["reference_images"] = reference_images
                captured["seed"] = kwargs.get("seed")
                latest = self_test.service.require_project(project.id)
                latest.ai_recommended_shot_count = 77
                self_test.store.save_project(latest)
                output = Path(output_dir) / "character.png"
                output.write_bytes(b"generated-character")
                return str(output)

        self_test = self
        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            assets = asyncio.run(
                self.service.generate_character_references(
                    project.id,
                    [character.id],
                    reference_asset_ids=[reference.id],
                    appearance_profile_id=appearance.id,
                )
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(captured["reference_images"], str(reference_path.resolve()))
        self.assertEqual(
            captured["seed"],
            self.service._stable_seed(project.id, character.id, appearance.id),
        )
        saved_project = self.service.require_project(project.id)
        saved_character = next(item for item in saved_project.characters if item.id == character.id)
        saved_appearance = next(
            item for item in saved_character.appearance_profiles if item.id == appearance.id
        )
        self.assertEqual(saved_project.ai_recommended_shot_count, 77)
        self.assertTrue(saved_appearance.approved)
        self.assertEqual(saved_appearance.reference_asset_ids, [reference.id, assets[0].id])

    def test_manual_storyboard_count_replans_instead_of_truncating(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="固定十镜", story="一百五十秒人物传记", target_duration_seconds=150)
        )
        calls: list[str] = []

        class FakeLLM:
            async def generate_storyboard(self, **kwargs: object) -> Storyboard:
                calls.append(str(kwargs.get("user_suggestions") or ""))
                count = 19 if len(calls) == 1 else 10
                return Storyboard(
                    topic="测试",
                    scenes=[
                        Scene(
                            id=index,
                            narrative=(f"错误的十九镜段落 {index}" if count == 19 else f"完整重构段落 {index}"),
                            story_beat=("完整结局" if count == 10 and index == 10 else ""),
                            visual_prompt=f"重构画面 {index}",
                            motion_prompt="人物做一个连续的小动作",
                            duration=15,
                        )
                        for index in range(1, count + 1)
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 10, "manual"))

        self.assertEqual(len(calls), 2)
        self.assertIn("上一次返回了 19 个分镜", calls[1])
        self.assertEqual(len(shots), 10)
        self.assertEqual([shot.ordinal for shot in shots], list(range(1, 11)))
        self.assertEqual(shots[-1].narrative, "完整结局")
        self.assertNotIn("错误的十九镜段落", " ".join(shot.narrative for shot in shots))
        saved_project = self.service.require_project(project.id)
        self.assertEqual(saved_project.storyboard_count_mode, "manual")
        self.assertEqual(saved_project.manual_shot_count, 10)

    def test_manual_storyboard_count_mismatch_preserves_existing_storyboard(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="保留旧分镜", story="完整人物故事", target_duration_seconds=30)
        )
        existing = Shot(
            project_id=project.id,
            ordinal=1,
            title="已经确认的旧分镜",
            narrative="旧分镜内容不得被错误结果覆盖",
            duration_seconds=10,
        )
        self.store.save_shot(existing)
        calls = 0

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                nonlocal calls
                calls += 1
                return Storyboard(
                    topic="始终数量错误",
                    scenes=[
                        Scene(
                            id=index,
                            narrative=f"错误结果 {index}",
                            visual_prompt=f"错误画面 {index}",
                            motion_prompt="轻微动作",
                            duration=5,
                        )
                        for index in range(1, 7)
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            with self.assertRaisesRegex(ValueError, "连续 3 次未遵守严格 5 镜要求"):
                asyncio.run(self.service.generate_storyboard(project.id, 5, "manual"))

        self.assertEqual(calls, 3)
        saved = self.store.list_shots(project.id)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].id, existing.id)
        self.assertEqual(saved[0].narrative, "旧分镜内容不得被错误结果覆盖")

    def test_import_storyboard_preserves_rows_and_fills_only_blank_fields(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="客户分镜表", story="人物进门并落座", target_duration_seconds=12)
        )
        rows = [
            {"title": "客户开场", "narrative": "人物推门", "duration_seconds": "5"},
            {"title": "客户收尾", "narrative": "人物落座", "duration_seconds": "7"},
        ]

        class FakeLLM:
            async def generate_json(self, *_: object, **__: object) -> dict[str, object]:
                return {
                    "rows": [
                        {"title": "AI 不得覆盖", "shot_size": "全景", "camera_angle": "平视"},
                        {"title": "AI 不得覆盖", "shot_size": "近景", "camera_angle": "侧面"},
                        {"title": "AI 多出的第三行"},
                    ]
                }

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.import_storyboard(project.id, rows, prompt_targets=[]))

        self.assertEqual(len(shots), 2)
        self.assertEqual([shot.title for shot in shots], ["客户开场", "客户收尾"])
        self.assertEqual([shot.shot_size for shot in shots], ["全景", "近景"])
        self.assertEqual(self.service.require_project(project.id).manual_shot_count, 2)

    def test_shot_appearance_profile_overrides_base_character_look(self) -> None:
        young = CharacterAppearanceProfile(
            label="少年期",
            time_context="12 岁",
            description="清瘦少年，圆脸",
            wardrobe="浅青色短袄",
            approved=True,
        )
        character = CharacterProfile(
            name="阿明",
            description="成年男子，健壮",
            wardrobe="深色长袍",
            appearance_profiles=[young],
        )
        project = self.service.create_project(ProjectBrief(title="年龄变化", story="阿明的一生"))
        project.characters = [character]
        self.store.save_project(project)

        prompt = self.service.compile_visual_prompt(
            project,
            "站在院中",
            [character.id],
            {character.id: young.id},
        )
        self.assertIn("12 岁", prompt)
        self.assertIn("浅青色短袄", prompt)
        self.assertNotIn("成年男子", prompt)

    def test_style_activation_refreshes_all_downstream_prompts(self) -> None:
        project = self.service.create_project(ProjectBrief(title="style", story="story"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="人物走入房间",
            scene_description="办公室中景",
            subject_motion="缓慢走入",
            visual_prompt="旧视觉",
            keyframe_prompt="旧首帧",
            video_prompt="旧 H3",
            seedance_prompt="旧 Seedance",
        )
        self.store.save_shot(shot)
        project.style_profile = StyleProfile(
            name="定格黏土",
            medium="手工黏土定格动画",
            palette="低饱和赭石与青灰",
            lighting="柔和侧光",
            analysis_summary="手工黏土定格动画，低饱和赭石与青灰",
            approved=True,
        )

        self.service.update_project(project)
        refreshed = self.service.require_shot(shot.id)
        self.assertEqual(refreshed.content_revision, 2)
        self.assertIn("手工黏土定格动画", refreshed.keyframe_prompt)
        self.assertIn("手工黏土定格动画", refreshed.video_prompt)
        self.assertIn("手工黏土定格动画", refreshed.seedance_prompt)
        self.assertEqual(refreshed.h3_prompt_source_revision, refreshed.content_revision)
        self.assertEqual(refreshed.seedance_prompt_source_revision, refreshed.content_revision)

    def test_optional_scene_profile_and_seedance_tail_continuity(self) -> None:
        project = self.service.create_project(ProjectBrief(title="continuity", story="三分钟谈话"))
        scene_path = self.root / "interview-room.png"
        tail_path = self.root / "tail.png"
        scene_path.write_bytes(b"scene")
        tail_path.write_bytes(b"tail")
        scene_asset = self.service.register_existing_asset(project.id, scene_path, AssetRole.SCENE, "谈话室母版")
        tail_asset = self.service.register_existing_asset(project.id, tail_path, AssetRole.LAST_FRAME, "上一镜尾帧")
        profile = SceneProfile(
            name="谈话室",
            description="米白墙面、深木桌、固定门窗位置",
            reference_asset_ids=[scene_asset.id],
            approved=True,
        )
        project.scene_profiles = [profile]
        self.store.save_project(project)
        first = Shot(project_id=project.id, ordinal=1, title="第一镜", last_frame_asset_id=tail_asset.id)
        second = Shot(
            project_id=project.id,
            ordinal=2,
            title="第二镜",
            scene_profile_id=profile.id,
            use_scene_profile=False,
            continuity_mode=ShotContinuityMode.CONTINUOUS,
        )
        self.store.save_shot(first)
        self.store.save_shot(second)
        assets = self.store.list_assets(project.id)

        without_scene = self.service.seedance_reference_assets(second, assets, project)
        self.assertNotIn(scene_asset.id, [asset.id for asset in without_scene])
        self.assertEqual(self.service.continuity_input_asset_id(project, second), tail_asset.id)
        self.assertEqual(self.service.seedance_first_frame_asset_id(project, second, assets), tail_asset.id)

        second.use_scene_profile = True
        with_scene = self.service.seedance_reference_assets(second, assets, project)
        self.assertEqual([asset.id for asset in with_scene], [tail_asset.id])
        prompt = self.service.compile_seedance_prompt(project, second, assets)
        self.assertIn("图片1是上一镜真实尾帧", prompt)
        self.assertNotIn("图片2", prompt)

    def test_single_shot_ai_redo_locks_only_identity_ordinal_and_duration(self) -> None:
        project = self.service.create_project(ProjectBrief(title="redo", story="原剧情"))
        shot = Shot(
            project_id=project.id,
            ordinal=3,
            duration_seconds=7.25,
            title="原镜头",
            scene_description="原房间",
            dialogue="原对白",
        )
        self.store.save_shot(shot)

        class FakeLLM:
            async def generate_json(self, *_: object, **__: object) -> dict[str, object]:
                return {
                    "title": "彻底重做",
                    "narrative": "角色来到室外",
                    "dialogue": "新的对白",
                    "dialogue_speaker_name": "",
                    "scene_description": "雨夜街道",
                    "character_names": [],
                    "shot_size": "全景",
                    "camera_angle": "低机位",
                    "lens": "24mm",
                    "camera_motion": "缓慢横移",
                    "subject_motion": "角色撑伞走过",
                    "transition": "硬切",
                    "audio_design": "雨声",
                    "visual_prompt": "雨夜电影感",
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            revised = asyncio.run(
                self.service.revise_shot_with_ai(project.id, shot.id, "场景和对白都换掉", [])
            )

        self.assertEqual(revised.id, shot.id)
        self.assertEqual(revised.ordinal, 3)
        self.assertEqual(revised.duration_seconds, 7.25)
        self.assertEqual(revised.title, "彻底重做")
        self.assertEqual(revised.scene_description, "雨夜街道")
        self.assertEqual(revised.dialogue, "新的对白")
        self.assertGreater(revised.content_revision, shot.content_revision)

    def test_comfyui_submit_uses_canonical_uuid_prompt_id(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            is_error = False

            @staticmethod
            def json() -> dict[str, object]:
                return {}

        class FakeAsyncClient:
            def __init__(self, **_: object) -> None:
                pass

            async def __aenter__(self) -> "FakeAsyncClient":
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            async def post(self, _: str, json: dict[str, object]) -> FakeResponse:
                captured.update(json)
                return FakeResponse()

        with patch("src.video_workflow.integrations.comfyui.httpx.AsyncClient", FakeAsyncClient):
            prompt_id = asyncio.run(ComfyUIClient("https://example.invalid").submit({"1": {}}))

        self.assertEqual(str(UUID(prompt_id)), prompt_id)
        self.assertEqual(captured["prompt_id"], prompt_id)

    def test_auto_routing_and_reference_tag_order(self) -> None:
        project = self.service.create_project(ProjectBrief(title="refs", story="story"))
        image_a = self.root / "a.png"
        image_b = self.root / "b.png"
        video = self.root / "motion.mp4"
        audio = self.root / "voice.wav"
        for path in (image_a, image_b, video, audio):
            path.write_bytes(path.name.encode())
        assets = [
            self.service.register_existing_asset(project.id, image_a, AssetRole.CHARACTER, "A"),
            self.service.register_existing_asset(project.id, video, AssetRole.MOTION, "Motion"),
            self.service.register_existing_asset(project.id, image_b, AssetRole.STYLE, "B"),
            self.service.register_existing_asset(project.id, audio, AssetRole.VOICE, "Voice"),
        ]
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            video_prompt="角色走入镜头",
            reference_asset_ids=[asset.id for asset in assets],
        )
        self.assertEqual(self.service.resolve_generation_mode(shot, assets), GenerationMode.R2V)
        prompt = self.service.compile_h3_prompt(project, shot, assets)
        self.assertLess(prompt.index("<Picture 1>"), prompt.index("<Video 1>"))
        self.assertLess(prompt.index("<Video 1>"), prompt.index("<Picture 2>"))
        self.assertIn("<Audio 1>", prompt)

        # Once a composed keyframe exists, auto mode must use it instead of
        # asking R2V to rebuild a multi-person composition from portraits.
        shot.keyframe_asset_id = assets[0].id
        shot.image_path = str(image_a)
        self.assertEqual(self.service.resolve_generation_mode(shot, assets), GenerationMode.I2V)
        i2v_prompt = self.service.compile_h3_prompt(project, shot, assets)
        self.assertNotIn("<Picture", i2v_prompt)
        self.assertIn("禁止新增或复制人物", i2v_prompt)

        self.store.save_shot(shot)
        first_plan = self.service.plan_shot(project.id, shot.id)
        second_plan = self.service.plan_shot(project.id, shot.id)
        self.assertEqual(first_plan.video_prompt, second_plan.video_prompt)
        self.assertEqual(second_plan.video_prompt.count("镜头："), 1)

    def test_speaker_binding_and_complex_shot_segmentation(self) -> None:
        project = self.service.create_project(ProjectBrief(title="谈话", story="C要求秦绍辉存放物品"))
        speaker = CharacterProfile(name="C同志", voice_description="中年女性，语气正式")
        subject = CharacterProfile(name="秦绍辉", voice_description="青年男性，语气平静")
        project.characters = [speaker, subject]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="C同志要求秦绍辉把手机和随身物品放入储物柜",
            dialogue="秦绍辉同志，请你把手机和随身携带的物品放入储物柜中。",
            duration_seconds=11.5,
            subject_motion=(
                "起势：C同志抬手指向储物柜并说出指令；"
                "发展：秦绍辉点头，从口袋拿出手机并走向储物柜；"
                "收束：秦绍辉拉开柜门，把物品放入柜中"
            ),
            character_ids=[speaker.id, subject.id],
        )
        self.store.save_shot(shot)
        planned = self.service.plan_shot(project.id, shot.id)
        self.assertEqual(planned.dialogue_speaker_id, speaker.id)
        self.assertIn("唯一发言者明确为C同志", planned.video_prompt)
        self.assertIn("秦绍辉全程闭嘴", planned.video_prompt)
        segments = self.service.h3_segment_plan(project, planned, [])
        self.assertEqual(len(segments), 3)
        self.assertEqual(sum(bool(segment["has_dialogue"]) for segment in segments), 1)
        self.assertAlmostEqual(sum(float(segment["duration"]) for segment in segments), 11.5, places=2)
        self.assertTrue(all("不出现任何字幕" in str(segment["prompt"]) for segment in segments))
        silent_segments = [segment for segment in segments if not segment["has_dialogue"]]
        self.assertTrue(silent_segments)
        self.assertTrue(all("全程嘴唇闭合" in str(segment["prompt"]) for segment in silent_segments))
        self.assertTrue(all("C同志要求秦绍辉" not in str(segment["prompt"]) for segment in silent_segments))
        speaking_segment = next(segment for segment in segments if segment["has_dialogue"])
        self.assertEqual(speaking_segment["dialogue"], shot.dialogue)
        self.assertEqual(speaking_segment["speaker_id"], speaker.id)
        self.assertIn("唯一发言者明确为C同志", str(speaking_segment["prompt"]))
        self.assertIn("秦绍辉全程闭嘴", str(speaking_segment["prompt"]))

    def test_dialogue_r2v_keeps_group_composition_and_binds_clean_audio(self) -> None:
        project = self.service.create_project(ProjectBrief(title="三人中景", story="D介绍秦绍辉"))
        subject = CharacterProfile(name="秦绍辉", description="男性，黑框眼镜")
        speaker = CharacterProfile(name="D同志", description="女性，深灰色西装")
        listener = CharacterProfile(name="C同志", description="女性，蓝框眼镜")
        project.characters = [subject, speaker, listener]
        self.store.save_project(project)
        keyframe_path = self.root / "group.png"
        qin_path = self.root / "qin.png"
        d_path = self.root / "d.png"
        c_path = self.root / "c.png"
        for path in (keyframe_path, qin_path, d_path, c_path):
            path.write_bytes(b"image")
        keyframe = self.service.register_existing_asset(project.id, keyframe_path, AssetRole.KEYFRAME, "三人中景")
        qin = self.service.register_existing_asset(project.id, qin_path, AssetRole.CHARACTER, "秦绍辉")
        d_ref = self.service.register_existing_asset(project.id, d_path, AssetRole.CHARACTER, "D同志")
        c_ref = self.service.register_existing_asset(project.id, c_path, AssetRole.CHARACTER, "C同志")
        subject.reference_asset_ids = [qin.id]
        speaker.reference_asset_ids = [d_ref.id]
        listener.reference_asset_ids = [c_ref.id]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            dialogue="同志你好，这是秦绍辉。",
            dialogue_speaker_id=speaker.id,
            duration_seconds=6.82,
            subject_motion=(
                "起势：D同志与秦绍辉进入房间；"
                "发展：D同志抬手介绍秦绍辉；"
                "收束：C同志点头回应"
            ),
            character_ids=[subject.id, speaker.id, listener.id],
            keyframe_asset_id=keyframe.id,
            reference_asset_ids=[qin.id, d_ref.id, c_ref.id],
            shot_size="三人中景",
        )
        previous_audio = (settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE, settings.TTS_PROVIDER)
        settings.H3_POSTPROCESS_AUDIO = True
        settings.H3_AUDIO_MODE = "clean_tts"
        settings.TTS_PROVIDER = "edge"
        try:
            self.assertEqual(self.service.resolve_generation_mode(shot, [keyframe, qin, d_ref, c_ref]), GenerationMode.R2V)
            prompt = self.service.compile_h3_prompt(project, shot, [keyframe, qin, d_ref, c_ref])
        finally:
            settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE, settings.TTS_PROVIDER = previous_audio
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Subject 2> (S1)", prompt)
        self.assertIn("<Audio 1> is the directly reused clean Chinese dialogue track", prompt)
        self.assertIn("<d>[Chinese] 同志你好，这是秦绍辉。</d>", prompt)
        self.assertEqual(prompt.count("<d>"), 1)
        self.assertEqual(prompt.count("</d>"), 1)
        self.assertIn("禁止重复台词", prompt)
        self.assertIn("Do not change to a single-person close-up", prompt)
        segments = self.service.h3_segment_plan(project, shot, [keyframe, qin, d_ref, c_ref])
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(float(segments[0]["generation_duration"]), 6.82, places=2)
        self.assertIn("Do not change to a single-person close-up", str(segments[0]["prompt"]))
        required = RenderQueue._required_assets(shot, [keyframe, qin, d_ref, c_ref], GenerationMode.R2V)
        self.assertEqual([asset.id for asset in required], [keyframe.id, qin.id, d_ref.id, c_ref.id])

    def test_dialogue_voice_follows_bound_speaker(self) -> None:
        project = self.service.create_project(ProjectBrief(title="声音", story="角色说话"))
        male = CharacterProfile(name="男主", voice_description="青年男性，低沉克制")
        project.characters = [male]
        shot = Shot(project_id=project.id, ordinal=1, dialogue="你好", dialogue_speaker_id=male.id)
        self.assertEqual(DialogueAudioService.voice_for(project, shot), settings.TTS_DEFAULT_MALE_VOICE)
        male.tts_voice = "zh-CN-YunyangNeural"
        self.assertEqual(DialogueAudioService.voice_for(project, shot), "zh-CN-YunyangNeural")
        male.tts_voice = ""
        male.voice_description = "中年男性，沉稳威严"
        self.assertEqual(DialogueAudioService.voice_for(project, shot), settings.TTS_DEFAULT_MATURE_MALE_VOICE)

    @unittest.skipUnless(shutil.which(settings.FFMPEG_BIN) and shutil.which(settings.FFPROBE_BIN), "ffmpeg required")
    def test_clean_audio_delivery_discards_native_h3_track(self) -> None:
        video = self.root / "native-noise.mp4"
        voice = self.root / "voice.wav"
        subprocess.run(
            [
                settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24",
                "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.8:r=32000",
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(video),
            ],
            check=True,
        )
        subprocess.run(
            [
                settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                str(voice),
            ],
            check=True,
        )
        previous_audio = (settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE)
        settings.H3_POSTPROCESS_AUDIO = True
        settings.H3_AUDIO_MODE = "clean_tts"
        try:
            delivery = RenderQueue._replace_generated_audio(video, voice, 0.2)
        finally:
            settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE = previous_audio
        self.assertTrue(delivery["applied"])
        self.assertEqual(delivery["native_h3_audio"], "discarded")
        self.assertEqual(delivery["delivered"], "independent_tts")
        probe = subprocess.run(
            [
                settings.FFPROBE_BIN, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels", "-of", "json", str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual(stream["sample_rate"], str(settings.FINAL_AUDIO_SAMPLE_RATE))
        self.assertEqual(stream["channels"], 2)
        qc = RenderQueue._validate_rendered_media(video, 320, 180, 2.0, True)
        self.assertTrue(qc["passed"], qc["issues"])
        bad_qc = RenderQueue._validate_rendered_media(video, 640, 360, 2.0, True)
        self.assertFalse(bad_qc["passed"])
        self.assertTrue(any("分辨率异常" in issue for issue in bad_qc["issues"]))

    @unittest.skipUnless(shutil.which(settings.FFMPEG_BIN) and shutil.which(settings.FFPROBE_BIN), "ffmpeg required")
    def test_dialogue_reference_and_delivery_share_exact_timing(self) -> None:
        source = self.root / "speech.wav"
        delivery = self.root / "delivery.wav"
        h3_reference = self.root / "h3-reference.wav"
        subprocess.run(
            [
                settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                str(source),
            ],
            check=True,
        )
        RenderQueue._prepare_dialogue_guide(source, delivery, 2.5, 0.35)
        RenderQueue._pad_dialogue_guide(delivery, h3_reference, 5.0)
        self.assertAlmostEqual(RenderQueue._media_duration(delivery), 2.5, delta=0.03)
        self.assertAlmostEqual(RenderQueue._media_duration(h3_reference), 5.0, delta=0.03)

    def test_multi_speaker_dialogue_is_split_into_role_locked_segments(self) -> None:
        project = self.service.create_project(ProjectBrief(title="问答", story="三人谈话"))
        interviewer = CharacterProfile(name="C同志", voice_description="中年女性")
        subject = CharacterProfile(name="秦绍辉", voice_description="青年男性")
        supervisor = CharacterProfile(name="严秉诚", voice_description="中年男性")
        project.characters = [interviewer, subject, supervisor]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="C同志追问是否汇报，秦绍辉否认，严秉诚说继续说。",
            dialogue="向党委或分管领导汇报过吗？没有，党委通过的指标没上浮。继续说。",
            duration_seconds=10.5,
            subject_motion="C同志提问；秦绍辉摇头回答；严秉诚说继续说。",
            character_ids=[interviewer.id, subject.id, supervisor.id],
        )
        self.store.save_shot(shot)

        planned = self.service.plan_shot(project.id, shot.id)
        self.assertIsNone(planned.dialogue_speaker_id)
        self.assertEqual(
            [turn.speaker_id for turn in planned.dialogue_turns],
            [interviewer.id, subject.id, supervisor.id],
        )
        segments = self.service.h3_segment_plan(project, planned, [])
        self.assertEqual([segment["speaker_id"] for segment in segments], [interviewer.id, subject.id, supervisor.id])
        self.assertTrue(all(float(segment["generation_duration"]) >= 5 for segment in segments))
        self.assertAlmostEqual(sum(float(segment["duration"]) for segment in segments), 10.5, places=2)
        self.assertIn("唯一发言者明确为严秉诚", str(segments[-1]["prompt"]))
        self.assertIn("C同志,秦绍辉全程闭嘴", str(segments[-1]["prompt"]))
        self.assertIn("继续说", str(segments[-1]["prompt"]))

    def test_legacy_import_copies_existing_outputs(self) -> None:
        session = settings.OUTPUT_DIR / "old-session"
        (session / "images").mkdir(parents=True)
        (session / "videos").mkdir(parents=True)
        image = session / "images" / "1.png"
        video = session / "videos" / "1.mp4"
        image.write_bytes(b"image")
        video.write_bytes(b"video")
        (session / "script.json").write_text(
            json.dumps(
                {
                    "topic": "旧项目",
                    "scenes": [
                        {
                            "id": 1,
                            "duration": 6,
                            "narrative": "叙事",
                            "visual_prompt": "画面",
                            "motion_prompt": "动作",
                            "image_path": str(image),
                            "video_path": str(video),
                            "image_status": "completed",
                            "video_status": "completed",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        project = self.service.import_legacy_session("old-session")
        shots = self.store.list_shots(project.id)
        self.assertEqual(len(shots), 1)
        self.assertTrue(Path(shots[0].image_path or "").exists())
        self.assertTrue(Path(shots[0].video_path or "").exists())
        self.assertEqual(len(self.store.list_assets(project.id)), 2)

    def test_keyframe_generation_uses_project_aspect_ratio(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="aspect", story="story", aspect_ratio="16:9")
        )
        shot = Shot(project_id=project.id, ordinal=1, visual_prompt="wide establishing shot")
        self.store.save_shot(shot)
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(self, scene: object, output_dir: str, *_: object, **kwargs: object) -> str:
                captured.update(kwargs)
                output = Path(output_dir) / "1_keyframe.png"
                output.write_bytes(b"image")
                return str(output)

        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            generated = asyncio.run(self.service.generate_keyframes(project.id, [shot.id]))

        self.assertEqual(captured["aspect_ratio"], "16:9")
        self.assertEqual(generated[0].image_status, "completed")

    def test_keyframe_revision_mode_controls_previous_image_reference(self) -> None:
        project = self.service.create_project(ProjectBrief(title="revision", story="story"))
        previous_path = self.root / "previous.png"
        character_path = self.root / "character.png"
        previous_path.write_bytes(b"previous")
        character_path.write_bytes(b"character")
        previous = self.service.register_existing_asset(project.id, previous_path, AssetRole.KEYFRAME, "旧首帧")
        character = self.service.register_existing_asset(project.id, character_path, AssetRole.CHARACTER, "角色参考")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            visual_prompt="办公室里的双人中景",
            reference_asset_ids=[character.id],
            keyframe_asset_id=previous.id,
            image_path=str(previous_path),
            image_status="completed",
        )
        self.store.save_shot(shot)
        captured: list[dict[str, object]] = []

        class FakeImageGenerator:
            async def generate_image(
                self,
                scene: object,
                output_dir: str,
                reference_image_path: str | None = None,
                **_: object,
            ) -> str:
                captured.append({"scene": scene, "references": reference_image_path})
                output = Path(output_dir) / "generated.png"
                output.write_bytes(f"generated-{len(captured)}".encode())
                return str(output)

        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            iterated = asyncio.run(
                self.service.generate_keyframes(
                    project.id,
                    [shot.id],
                    revision_mode="iterate",
                    user_suggestions="保留构图，把窗外改成雨夜",
                )
            )
            iterated_asset = self.store.get_asset(iterated[0].keyframe_asset_id or "")
            self.assertIsNotNone(iterated_asset)
            current_path = Path(iterated_asset.path)  # type: ignore[union-attr]
            fresh = asyncio.run(
                self.service.generate_keyframes(
                    project.id,
                    [shot.id],
                    revision_mode="fresh",
                    user_suggestions="改成俯拍构图",
                )
            )

        iterate_references = str(captured[0]["references"]).split(",")
        fresh_references = str(captured[1]["references"]).split(",")
        self.assertEqual(iterate_references[0], str(previous_path.resolve()))
        self.assertIn(str(character_path.resolve()), iterate_references)
        self.assertIn("保留构图，把窗外改成雨夜", captured[0]["scene"].visual_prompt)  # type: ignore[union-attr]
        self.assertIn("保留构图，把窗外改成雨夜", iterated_asset.description)  # type: ignore[union-attr]
        self.assertNotEqual(current_path.resolve(), previous_path.resolve())
        self.assertEqual(fresh_references, [str(character_path.resolve())])
        self.assertNotIn(str(current_path.resolve()), fresh_references)
        self.assertEqual(fresh[0].image_status, "completed")

    def test_failed_keyframe_regeneration_keeps_user_suggestion_for_retry(self) -> None:
        project = self.service.create_project(ProjectBrief(title="retry", story="story"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            keyframe_prompt="两人在固定谈话室交谈",
        )
        self.store.save_shot(shot)

        class FailingImageGenerator:
            async def generate_image(self, *_: object, **__: object) -> str:
                raise RuntimeError("temporary image provider error")

        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FailingImageGenerator(),
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary image provider error"):
                asyncio.run(
                    self.service.generate_keyframes(
                        project.id,
                        [shot.id],
                        revision_mode="fresh",
                        user_suggestions="保持谈话室布局，把机位改到门口",
                    )
                )

        failed = self.service.require_shot(shot.id)
        self.assertEqual(failed.image_status, "failed")
        self.assertEqual(failed.keyframe_revision_mode, "fresh")
        self.assertEqual(
            failed.keyframe_revision_suggestion_draft,
            "保持谈话室布局，把机位改到门口",
        )

    def test_render_queue_deduplicates_and_cancels_queued_shot(self) -> None:
        project = self.service.create_project(ProjectBrief(title="queue", story="story"))
        image = self.root / "first.png"
        image.write_bytes(b"image")
        asset = self.service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            keyframe_asset_id=asset.id,
            image_path=str(image),
            h3_width=608,
            h3_height=352,
            h3_steps=6,
            h3_seed=123456,
        )
        self.store.save_shot(shot)
        queue = RenderQueue(self.store, self.service)

        jobs = queue.enqueue(project.id, [shot.id])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].seed, 123456)
        self.assertEqual(jobs[0].input_snapshot["h3_parameters"]["width"], 608)
        self.assertTrue(jobs[0].input_snapshot["h3_parameters"]["turbo"])
        self.assertEqual(jobs[0].input_snapshot["h3_parameters"]["steps"], 6)
        self.assertEqual(jobs[0].input_snapshot["h3_parameters"]["model_profile"], "pruned_int8")
        self.assertEqual(jobs[0].input_snapshot["h3_parameters"]["text_encoder_profile"], "nvfp4")
        self.assertEqual(
            jobs[0].input_snapshot["h3_parameters"]["diffusion_models"]["i2v"],
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        )
        self.assertEqual(
            jobs[0].input_snapshot["h3_parameters"]["text_encoder_model"],
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        )
        self.assertEqual(queue.enqueue(project.id, [shot.id]), [])

        cancelled = asyncio.run(queue.cancel(jobs[0].id))
        self.assertEqual(cancelled.status, JobStatus.CANCELLED)
        self.assertEqual(self.store.get_shot(shot.id).video_status, "cancelled")  # type: ignore[union-attr]

    def test_render_queue_blocks_auto_multi_character_r2v_without_composed_keyframe(self) -> None:
        project = self.service.create_project(ProjectBrief(title="quality gate", story="story"))
        ref_a = self.root / "a.png"
        ref_b = self.root / "b.png"
        ref_a.write_bytes(b"image-a")
        ref_b.write_bytes(b"image-b")
        asset_a = self.service.register_existing_asset(project.id, ref_a, AssetRole.CHARACTER, "A")
        asset_b = self.service.register_existing_asset(project.id, ref_b, AssetRole.CHARACTER, "B")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            character_ids=["character-a", "character-b"],
            reference_asset_ids=[asset_a.id, asset_b.id],
            generation_mode=GenerationMode.AUTO,
        )
        self.store.save_shot(shot)

        with self.assertRaisesRegex(ValueError, "多人镜头.*完整首帧"):
            RenderQueue(self.store, self.service).enqueue(project.id, [shot.id])

        self.assertEqual(self.store.list_jobs(project.id), [])

    def test_runtime_settings_mask_secrets_and_apply_values(self) -> None:
        previous_key = settings.DEEPSEEK_API_KEY
        previous_model = settings.DEEPSEEK_MODEL
        manager = RuntimeSettingsManager()
        try:
            manager.update({"DEEPSEEK_API_KEY": "test-secret-1234", "DEEPSEEK_MODEL": "test-model"})
            payload = manager.public_payload()
            fields = {field["key"]: field for group in payload["groups"] for field in group["fields"]}
            self.assertEqual(fields["DEEPSEEK_API_KEY"]["value"], "")
            self.assertEqual(fields["DEEPSEEK_API_KEY"]["masked"], "••••1234")
            self.assertEqual(settings.DEEPSEEK_MODEL, "test-model")
            self.assertEqual(manager.path.stat().st_mode & 0o777, 0o600)
            routes = {route["id"]: route for route in payload["routes"]}
            self.assertEqual(routes["storyboard"]["setting_key"], "LLM_PROVIDER")
            self.assertEqual(routes["h3_video"]["effective_label"], "MiniMax H3 / ComfyUI")
            self.assertIsNone(routes["finalize"]["setting_key"])
        finally:
            settings.DEEPSEEK_API_KEY = previous_key
            settings.DEEPSEEK_MODEL = previous_model

    def test_brief_analysis_uses_bound_reference_images(self) -> None:
        project = self.service.create_project(ProjectBrief(title="analysis", story="小雨在旧车站等母亲", target_duration_seconds=30))
        reference = self.root / "xiaoyu.png"
        reference.write_bytes(b"image")
        asset = self.service.register_existing_asset(project.id, reference, AssetRole.CHARACTER, "小雨参考")
        project.characters = [CharacterProfile(name="小雨", reference_asset_ids=[asset.id])]
        self.store.save_project(project)
        captured: dict[str, object] = {}

        class FakeLLM:
            async def generate_json(self, _system: str, _prompt: str, reference_images: list[str] | None = None):
                captured["reference_images"] = reference_images
                return {
                    "visual_style": "电影写实",
                    "pacing": "克制",
                    "audience": "家庭观众",
                    "style_bible": "冷雨暖灯",
                    "negative_prompt": "角色漂移",
                    "delivery_notes": "保留环境声",
                    "recommended_shot_count": 4,
                    "shot_count_reason": "四个情绪节点",
                    "characters": [{"character_id": project.characters[0].id, "name": "小雨", "description": "短发女孩", "wardrobe": "黄色雨衣", "voice_description": "轻声", "reference_observations": "按参考图"}],
                    "analysis_notes": [],
                }

        with (
            patch.object(settings, "ARK_API_KEY", None),
            patch.object(settings, "GLM_API_KEY", None),
            patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()),
        ):
            draft = asyncio.run(self.service.analyze_brief(project.id))
        self.assertEqual(draft.recommended_shot_count, 4)
        self.assertEqual(draft.characters[0].character_id, project.characters[0].id)
        self.assertEqual(captured["reference_images"], [str(reference.resolve())])

    def test_character_reference_backfill_merges_with_latest_project_state(self) -> None:
        project = self.service.create_project(ProjectBrief(title="parallel backfill", story="甲与乙同场", target_duration_seconds=30))
        first = CharacterProfile(name="甲")
        second = CharacterProfile(name="乙", description="旧设定")
        project.characters = [first, second]
        self.store.save_project(project)
        reference = self.root / "first.png"
        reference.write_bytes(b"image")
        asset = self.service.register_existing_asset(project.id, reference, AssetRole.CHARACTER, "甲参考")

        class FakeLLM:
            async def generate_json(self, *_: object, **__: object) -> dict[str, object]:
                latest = project_service.require_project(project.id)
                latest.characters[1].description = "另一个并行任务刚保存的新设定"
                self_store.save_project(latest)
                return {
                    "name": "甲",
                    "description": "参考图中的甲",
                    "wardrobe": "青色长袍",
                    "voice_description": "平稳男声",
                }

        self_store = self.store
        project_service = self.service
        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            result = asyncio.run(self.service.analyze_character_references(project.id, first.id, [asset.id]))

        latest = self.service.require_project(project.id)
        self.assertEqual(result.description, "参考图中的甲")
        self.assertEqual(latest.characters[0].description, "参考图中的甲")
        self.assertEqual(latest.characters[1].description, "另一个并行任务刚保存的新设定")
        self.assertEqual(latest.characters[0].reference_asset_ids, [asset.id])

    def test_script_rewrite_uses_target_duration_and_user_suggestions(self) -> None:
        project = self.service.create_project(
            ProjectBrief(
                title="时长适配",
                story="角色先进入房间，进行了大量重复说明，最后发现关键证据。",
                target_duration_seconds=45,
            )
        )
        captured: dict[str, object] = {}

        class FakeLLM:
            async def generate_json(self, system_prompt: str, user_prompt: str, reference_images: list[str] | None = None):
                captured.update(system=system_prompt, prompt=user_prompt, reference_images=reference_images)
                return {
                    "rewritten_story": "角色进入房间，简短交代背景后发现关键证据，并在结尾作出选择。",
                    "rewrite_mode": "shorten",
                    "estimated_duration_seconds": 44,
                    "change_summary": "合并重复说明，保留发现证据的核心转折。",
                    "feasibility_notes": ["对白需要保持简洁"],
                }

        suggestion = "保留发现证据的反转，删掉重复背景说明。"
        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            draft = asyncio.run(self.service.rewrite_story(project.id, "shorten", suggestion))

        self.assertEqual(draft.rewrite_mode, "shorten")
        self.assertEqual(draft.estimated_duration_seconds, 44)
        self.assertIn("目标成片时长：45 秒", str(captured["prompt"]))
        self.assertIn(suggestion, str(captured["prompt"]))
        self.assertIn("任一生成片段不得超过 15 秒", str(captured["prompt"]))


if __name__ == "__main__":
    unittest.main()
