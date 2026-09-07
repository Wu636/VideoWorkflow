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
    Asset,
    AssetRole,
    AssetType,
    CharacterAppearanceProfile,
    CharacterProfile,
    GenerationMode,
    JobStatus,
    ProjectBrief,
    SceneProfile,
    SeedanceReferenceMode,
    Shot,
    ShotContinuityMode,
    ShotSplitPreview,
    ShotSplitSegment,
    StyleProfile,
    VoiceEvent,
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
from src.video_workflow.services.projects import ProjectService, ShotVersionConflictError, _normalize_project_analysis_payload
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

    def test_series_episode_inherits_style_selected_cast_and_reference_files(self) -> None:
        source = self.service.create_project(ProjectBrief(
            title="张小差安全科普",
            story="第一集",
            visual_style="中式奇幻轻喜剧插画",
            negative_prompt="禁止写实摄影",
        ))
        style_path = self.root / "style.png"
        character_path = self.root / "zhang.png"
        style_path.write_bytes(b"series-style")
        character_path.write_bytes(b"series-character")
        style_asset = self.store.save_asset(Asset(
            project_id=source.id,
            type=AssetType.IMAGE,
            role=AssetRole.STYLE,
            name="系列画风",
            path=str(style_path),
        ))
        zhang = CharacterProfile(name="张小差", reference_asset_ids=[])
        character_asset = self.store.save_asset(Asset(
            project_id=source.id,
            type=AssetType.IMAGE,
            role=AssetRole.CHARACTER,
            name="张小差四视图",
            path=str(character_path),
            character_id=zhang.id,
        ))
        zhang.reference_asset_ids = [character_asset.id]
        source.characters = [zhang, CharacterProfile(name="本集路人")]
        source.style_bible = "固定线稿、材质和角色比例"
        source.style_profile = StyleProfile(
            name="安全科普系列",
            medium="数字手绘",
            reference_asset_ids=[style_asset.id],
            approved=True,
        )
        self.store.save_project(source)

        series, source = self.service.save_project_as_series(source.id, "张小差科普宇宙")
        self.assertEqual(source.episode_number, 1)
        self.assertTrue(all(Path(asset.path).is_file() for asset in series.assets))
        self.assertNotIn(style_asset.id, series.style_profile.reference_asset_ids)

        inherited_character_id = next(item.id for item in series.characters if item.name == "张小差")
        episode = self.service.create_series_episode(
            series.id,
            ProjectBrief(title="电动车入户充电", story="张小差遇见王大爷"),
            [inherited_character_id],
        )
        episode_assets = self.store.list_assets(episode.id)
        self.assertEqual(episode.episode_number, 2)
        self.assertIn("数字手绘", episode.brief.visual_style)
        self.assertIn("固定线稿", source.style_bible)
        self.assertIn("逐集场景规则", episode.style_bible)
        self.assertIn("数字手绘", episode.style_bible)
        self.assertEqual(episode.brief.negative_prompt, source.brief.negative_prompt)
        self.assertEqual([item.name for item in episode.characters], ["张小差"])
        self.assertNotEqual(episode.characters[0].id, zhang.id)
        self.assertEqual(len(episode.characters[0].reference_asset_ids), 1)
        self.assertTrue(all(Path(asset.path).is_file() for asset in episode_assets))
        self.assertTrue(all(asset.project_id == episode.id for asset in episode_assets))
        self.assertEqual(
            next(asset.character_id for asset in episode_assets if asset.role == AssetRole.CHARACTER),
            episode.characters[0].id,
        )
        episode.characters.append(CharacterProfile(name="王大爷"))
        self.store.save_project(episode)
        refreshed_series, _ = self.service.save_project_as_series(episode.id)
        self.assertEqual(
            {item.name for item in refreshed_series.characters},
            {"张小差", "本集路人", "王大爷"},
        )
        self.assertEqual(
            next(item.id for item in refreshed_series.characters if item.name == "张小差"),
            inherited_character_id,
        )

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

    def test_split_preview_and_confirm_replaces_only_source_shot(self) -> None:
        project = self.service.create_project(ProjectBrief(
            title="拆分过载镜头",
            story="阿明发现电池冒烟，拔掉插头后退到门外。",
            target_duration_seconds=18,
        ))
        character = CharacterProfile(name="阿明", description="短发青年", wardrobe="蓝色工装")
        profile = SceneProfile(name="客厅", description="木地板，墙边插座", source_shot_ids=[])
        project.characters = [character]
        project.scene_profiles = [profile]
        self.store.save_project(project)
        first = self.store.save_shot(Shot(project_id=project.id, ordinal=1, title="开场", narrative="阿明进入客厅"))
        source = self.store.save_shot(Shot(
            project_id=project.id,
            ordinal=2,
            title="发现险情并处置",
            narrative="阿明发现电池冒烟，拔掉插头后退到门外",
            duration_seconds=8,
            scene_profile_ids=[profile.id],
            scene_profile_id=profile.id,
            use_scene_profile=True,
            character_ids=[character.id],
            keyframe_asset_id="asset_old_keyframe",
            reference_asset_ids=["asset_old_keyframe"],
            image_path="/tmp/old-first.png",
            video_path="/tmp/old-video.mp4",
            selected_video_job_id="job_old_video",
            image_status="completed",
            video_status="completed",
        ))
        profile.source_shot_ids = [source.id]
        self.store.save_project(project)
        tail = self.store.save_shot(Shot(
            project_id=project.id,
            ordinal=3,
            title="收尾",
            narrative="阿明在门外报警",
            video_path="/tmp/tail.mp4",
            video_status="completed",
        ))
        first_updated_at = first.updated_at
        captured: dict[str, str] = {}

        class FakeLLM:
            async def generate_json(self, system: str, prompt: str) -> dict[str, object]:
                captured["system"] = system
                captured["prompt"] = prompt
                return {
                    "rationale": "把发现和处置拆为两个单一动作",
                    "segments": [
                        {
                            "title": "发现冒烟",
                            "duration_seconds": 3,
                            "narrative": "阿明发现电池冒烟",
                            "scene_description": "客厅中，阿明望向墙边电池",
                            "scene_profile_name": "客厅",
                            "character_names": ["阿明"],
                            "shot_size": "中景",
                            "camera_angle": "平视",
                            "camera_motion": "缓慢推近",
                            "subject_motion": "阿明看到冒烟后停下",
                            "visual_beats": [{"start_seconds": 0, "end_seconds": 3, "subject_action": "阿明发现冒烟"}],
                        },
                        {
                            "title": "断电后退",
                            "duration_seconds": 5,
                            "narrative": "阿明拔掉插头并退到门外",
                            "scene_description": "阿明的手停在插头上方",
                            "scene_profile_name": "客厅",
                            "character_names": ["阿明"],
                            "shot_size": "近景",
                            "camera_angle": "俯拍",
                            "camera_motion": "固定镜头",
                            "subject_motion": "拔掉插头后快速后退",
                            "visual_beats": [{"start_seconds": 0, "end_seconds": 5, "subject_action": "拔掉插头后退"}],
                        },
                    ],
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            preview = asyncio.run(self.service.preview_shot_split(
                project.id,
                source.id,
                "动作拆开，总时长不变",
                2,
            ))

        self.assertIn("动作拆开", captured["prompt"])
        self.assertEqual(preview.source_shot_version, source.version)
        self.assertEqual(preview.proposed_duration_seconds, 8)
        self.assertEqual([shot.id for shot in self.store.list_shots(project.id)], [first.id, source.id, tail.id])

        preview.segments[0].title = "用户修改的发现镜头"
        preview.segments[0].duration_seconds = 3.5
        created = asyncio.run(self.service.confirm_shot_split(project.id, source.id, preview, []))
        board = self.store.list_shots(project.id)

        self.assertEqual([shot.id for shot in board], [first.id, created[0].id, created[1].id, tail.id])
        self.assertEqual([shot.ordinal for shot in board], [1, 2, 3, 4])
        self.assertEqual(board[0].updated_at, first_updated_at)
        self.assertEqual(board[-1].video_path, "/tmp/tail.mp4")
        self.assertEqual(board[-1].video_status, "completed")
        self.assertIsNone(self.store.get_shot(source.id))
        self.assertEqual(created[0].title, "用户修改的发现镜头")
        self.assertEqual(created[0].duration_seconds, 3.5)
        self.assertTrue(all(shot.id != source.id for shot in created))
        self.assertTrue(all(shot.keyframe_asset_id is None for shot in created))
        self.assertTrue(all(shot.selected_video_job_id is None for shot in created))
        self.assertTrue(all(shot.image_path is None and shot.video_path is None for shot in created))
        self.assertTrue(all(shot.image_status == "pending" and shot.video_status == "pending" for shot in created))
        self.assertTrue(all("asset_old_keyframe" not in shot.reference_asset_ids for shot in created))
        self.assertTrue(all(character.id in shot.character_ids for shot in created))
        self.assertTrue(all(shot.visual_prompt and shot.keyframe_prompt and shot.video_prompt and shot.seedance_prompt for shot in created))
        saved_profile = self.service.require_project(project.id).scene_profiles[0]
        self.assertNotIn(source.id, saved_profile.source_shot_ids)
        self.assertEqual(saved_profile.source_shot_ids, [shot.id for shot in created])

    def test_split_confirmation_rejects_stale_preview(self) -> None:
        project = self.service.create_project(ProjectBrief(title="过期预览", story="人物连续完成两个动作"))
        source = self.store.save_shot(Shot(project_id=project.id, ordinal=1, narrative="先开门，再落座", duration_seconds=6))
        preview = ShotSplitPreview(
            source_shot_id=source.id,
            source_shot_version=source.version,
            source_ordinal=source.ordinal,
            original_duration_seconds=source.duration_seconds,
            proposed_duration_seconds=6,
            segments=[
                ShotSplitSegment(title="开门", duration_seconds=3, narrative="人物开门"),
                ShotSplitSegment(title="落座", duration_seconds=3, narrative="人物落座"),
            ],
        )
        source.narrative = "页面之外已修改"
        source.version += 1
        self.store.save_shot(source)

        with self.assertRaisesRegex(ShotVersionConflictError, "新的保存结果"):
            asyncio.run(self.service.confirm_shot_split(project.id, source.id, preview, []))
        self.assertEqual([shot.id for shot in self.store.list_shots(project.id)], [source.id])

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
                captured["aspect_ratio"] = kwargs.get("aspect_ratio")
                captured["prompt"] = _scene.visual_prompt
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
                    user_suggestions="眼神更锐利，衣料增加磨损细节",
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
        self.assertEqual(captured["aspect_ratio"], "16:9")
        self.assertIn("大尺寸正脸近照", str(captured["prompt"]))
        self.assertIn("全身正面、标准侧面、全身背面", str(captured["prompt"]))
        self.assertIn("动物使用自然四足站姿", str(captured["prompt"]))
        self.assertIn("固定品种", str(captured["prompt"]))
        self.assertIn("眼神更锐利，衣料增加磨损细节", str(captured["prompt"]))
        saved_project = self.service.require_project(project.id)
        saved_character = next(item for item in saved_project.characters if item.id == character.id)
        saved_appearance = next(
            item for item in saved_character.appearance_profiles if item.id == appearance.id
        )
        self.assertEqual(saved_project.ai_recommended_shot_count, 77)
        self.assertTrue(saved_appearance.approved)
        self.assertEqual(saved_appearance.reference_asset_ids, [reference.id, assets[0].id])

    def test_missing_character_references_are_created_from_script_and_style(self) -> None:
        project = self.service.create_project(
            ProjectBrief(
                title="剧本生成人设",
                story="阿青带着弟弟小满穿过雨夜车站。",
                visual_style="低饱和电影写实，冷雨暖灯",
            )
        )
        style_path = self.root / "style.png"
        style_path.write_bytes(b"style")
        style_asset = self.service.register_existing_asset(project.id, style_path, AssetRole.STYLE, "客户风格图")
        project.style_profile = StyleProfile(
            name="冷雨电影写实",
            reference_asset_ids=[style_asset.id],
            analysis_summary="低饱和冷雨暖灯",
            approved=True,
        )
        self.store.save_project(project)
        image_calls: list[dict[str, object]] = []

        class FakeLLM:
            async def generate_json(self, _system: str, _prompt: str, reference_images: list[str] | None = None):
                return {
                    "visual_style": "冷雨电影写实", "pacing": "克制", "audience": "大众",
                    "style_bible": "冷雨暖灯", "negative_prompt": "身份漂移", "delivery_notes": "",
                    "recommended_shot_count": 4, "shot_count_reason": "四个叙事节点",
                    "characters": [
                        {"character_id": None, "name": "阿青", "description": "二十多岁短发女性", "wardrobe": "深蓝雨衣", "voice_description": "沉稳", "reference_observations": "按剧本推断"},
                        {"character_id": None, "name": "小满", "description": "十岁男孩", "wardrobe": "黄色雨衣", "voice_description": "稚嫩", "reference_observations": "按剧本推断"},
                    ],
                    "analysis_notes": [],
                }

        class FakeImageGenerator:
            async def generate_image(self, scene: Scene, output_dir: str, reference_images: str | None, **kwargs: object) -> str:
                image_calls.append({"prompt": scene.visual_prompt, "references": reference_images, "style": kwargs.get("image_style"), "aspect_ratio": kwargs.get("aspect_ratio")})
                output = Path(output_dir) / f"character-{len(image_calls)}.png"
                output.write_bytes(f"character-{len(image_calls)}".encode())
                return str(output)

        with (
            patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()),
            patch("src.video_workflow.services.projects.create_image_generator", return_value=FakeImageGenerator()),
        ):
            result = asyncio.run(self.service.generate_missing_character_references(
                project.id,
                mode="script_style",
                user_suggestions="所有角色保持统一雨衣材质，脸型必须有区分",
            ))

        saved = self.service.require_project(project.id)
        self.assertEqual(result["missing_count"], 2)
        self.assertEqual(len(result["assets"]), 2)
        self.assertEqual([character.name for character in saved.characters], ["阿青", "小满"])
        self.assertTrue(all(character.reference_asset_ids for character in saved.characters))
        self.assertTrue(all(call["references"] == str(style_path.resolve()) for call in image_calls))
        self.assertTrue(all("项目剧情依据：阿青带着弟弟小满" in str(call["prompt"]) for call in image_calls))
        self.assertTrue(all("禁止复制参考图人物身份" in str(call["prompt"]) for call in image_calls))
        self.assertTrue(all("所有角色保持统一雨衣材质" in str(call["prompt"]) for call in image_calls))
        self.assertTrue(all("全身正面、标准侧面、全身背面" in str(call["prompt"]) for call in image_calls))
        self.assertTrue(all(call["aspect_ratio"] == "16:9" for call in image_calls))
        self.assertTrue(all(style_asset.id not in character.reference_asset_ids for character in saved.characters))

    def test_partial_character_references_guide_missing_cast_without_identity_binding(self) -> None:
        first = CharacterProfile(name="姐姐", description="成年女性", wardrobe="深蓝外套")
        second = CharacterProfile(name="弟弟", description="少年", wardrobe="黄色外套")
        third = CharacterProfile(name="站长", description="中年男性", wardrobe="铁路制服")
        project = self.service.create_project(ProjectBrief(title="部分人设补齐", story="姐姐带弟弟向站长问路。", visual_style="电影写实"))
        reference_path = self.root / "sister.png"
        reference_path.write_bytes(b"sister")
        reference = self.service.register_existing_asset(project.id, reference_path, AssetRole.CHARACTER, "姐姐参考")
        first.reference_asset_ids = [reference.id]
        project.characters = [first, second, third]
        self.store.save_project(project)
        image_calls: list[dict[str, object]] = []

        class FakeLLM:
            async def generate_json(self, _system: str, _prompt: str, reference_images: list[str] | None = None):
                return {
                    "visual_style": "电影写实", "pacing": "自然", "audience": "大众",
                    "style_bible": "统一写实", "negative_prompt": "身份漂移", "delivery_notes": "",
                    "recommended_shot_count": 4, "shot_count_reason": "完整覆盖剧情",
                    "characters": [
                        {"character_id": first.id, "name": "姐姐", "description": first.description, "wardrobe": first.wardrobe, "voice_description": "", "reference_observations": "按参考图"},
                        {"character_id": second.id, "name": "弟弟", "description": second.description, "wardrobe": second.wardrobe, "voice_description": "", "reference_observations": "按剧本"},
                        {"character_id": third.id, "name": "站长", "description": third.description, "wardrobe": third.wardrobe, "voice_description": "", "reference_observations": "按剧本"},
                    ],
                    "analysis_notes": [],
                }

        class FakeImageGenerator:
            async def generate_image(self, scene: Scene, output_dir: str, reference_images: str | None, **_kwargs: object) -> str:
                image_calls.append({"prompt": scene.visual_prompt, "references": reference_images})
                output = Path(output_dir) / f"missing-{len(image_calls)}.png"
                output.write_bytes(f"missing-{len(image_calls)}".encode())
                return str(output)

        with (
            patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()),
            patch("src.video_workflow.services.projects.create_image_generator", return_value=FakeImageGenerator()),
        ):
            result = asyncio.run(self.service.generate_missing_character_references(
                project.id,
                mode="complete_missing",
                reference_asset_ids=[reference.id],
            ))

        saved_by_name = {character.name: character for character in self.service.require_project(project.id).characters}
        self.assertEqual(result["missing_count"], 2)
        self.assertEqual(len(result["assets"]), 2)
        self.assertTrue(all(call["references"] == str(reference_path.resolve()) for call in image_calls))
        self.assertEqual(saved_by_name["姐姐"].reference_asset_ids, [reference.id])
        self.assertNotIn(reference.id, saved_by_name["弟弟"].reference_asset_ids)
        self.assertNotIn(reference.id, saved_by_name["站长"].reference_asset_ids)
        self.assertTrue(all("禁止复制参考图人物身份" in str(call["prompt"]) for call in image_calls))

    def test_project_style_regeneration_uses_other_series_character_and_replaces_bad_binding(self) -> None:
        zhang = CharacterProfile(name="张小差", description="年轻阴差", wardrobe="炭黑工装")
        elder = CharacterProfile(name="王大爷", description="六十五岁农村老人", wardrobe="藏蓝中山装")
        project = self.service.create_project(
            ProjectBrief(
                title="同系列新角色",
                story="张小差劝王大爷不要把电动车推进家里充电。",
                visual_style="数字手绘，二次元写实融合渲染，人物为二次元萌感+中式写实特征",
            )
        )
        project.series_id = "series_test"
        zhang_path = self.root / "zhang-style-anchor.png"
        zhang_path.write_bytes(b"zhang")
        zhang_asset = self.service.register_existing_asset(project.id, zhang_path, AssetRole.CHARACTER, "张小差四视图")
        zhang_asset.character_id = zhang.id
        zhang_asset.tags = ["series:series_test"]
        zhang_asset.approved = True
        self.store.save_asset(zhang_asset)
        old_path = self.root / "elder-photoreal.png"
        old_path.write_bytes(b"old")
        old_asset = self.service.register_existing_asset(project.id, old_path, AssetRole.CHARACTER, "王大爷错误写实图")
        old_asset.character_id = elder.id
        old_asset.approved = True
        self.store.save_asset(old_asset)
        style_path = self.root / "generic-style.png"
        style_path.write_bytes(b"style")
        style_asset = self.service.register_existing_asset(project.id, style_path, AssetRole.STYLE, "普通风格图")
        project.style_profile = StyleProfile(
            name="中式轻喜剧插画",
            medium="数字手绘，二次元写实融合渲染",
            texture="人物为二次元萌感+中式写实特征，道具有真实磨损",
            reference_asset_ids=[style_asset.id],
            approved=True,
        )
        zhang.reference_asset_ids = [zhang_asset.id]
        elder.reference_asset_ids = [old_asset.id]
        project.characters = [zhang, elder]
        self.store.save_project(project)
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(self, scene: Scene, output_dir: str, reference_images: str | None, **kwargs: object) -> str:
                captured["references"] = reference_images
                captured["prompt"] = scene.visual_prompt
                captured["style"] = kwargs.get("image_style")
                output = Path(output_dir) / "elder-stylized.png"
                output.write_bytes(b"new")
                return str(output)

        with patch("src.video_workflow.services.projects.create_image_generator", return_value=FakeImageGenerator()):
            created = asyncio.run(self.service.generate_character_references(
                project.id,
                [elder.id],
                reference_strategy="project_style",
            ))

        self.assertEqual(
            str(captured["references"]).split(","),
            [str(zhang_path.resolve()), str(style_path.resolve())],
        )
        self.assertNotIn(str(old_path.resolve()), str(captured["references"]))
        self.assertIn("最高优先级画风锚点", str(captured["prompt"]))
        self.assertIn("禁止真人照片", str(captured["prompt"]))
        self.assertNotIn("二次元写实融合", str(captured["style"]))
        self.assertIn("二维手绘动画插画", str(captured["style"]))
        latest = self.service.require_project(project.id)
        saved_elder = next(character for character in latest.characters if character.id == elder.id)
        self.assertEqual(saved_elder.reference_asset_ids, [created[0].id])
        self.assertIsNotNone(self.store.get_asset(old_asset.id))

    def test_manual_storyboard_count_mismatch_is_not_retried_or_truncated(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="固定十镜", story="一百五十秒人物传记", target_duration_seconds=150)
        )
        calls: list[str] = []

        class FakeLLM:
            async def generate_storyboard(self, **kwargs: object) -> Storyboard:
                calls.append(str(kwargs.get("user_suggestions") or ""))
                count = 19
                return Storyboard(
                    topic="测试",
                    scenes=[
                        Scene(
                            id=index,
                            narrative=f"错误的十九镜段落 {index}",
                            story_beat="",
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
            with self.assertRaisesRegex(ValueError, "没有自动再次调用"):
                asyncio.run(self.service.generate_storyboard(project.id, 10, "manual"))

        self.assertEqual(len(calls), 1)
        self.assertIn("严格 10 个分镜", calls[0])
        self.assertEqual(self.store.list_shots(project.id), [])

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
            with self.assertRaisesRegex(ValueError, "没有自动再次调用"):
                asyncio.run(self.service.generate_storyboard(project.id, 5, "manual"))

        self.assertEqual(calls, 1)
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

    def test_shot_cast_sync_matches_bracket_aliases_and_all_character_references(self) -> None:
        project = self.service.create_project(ProjectBrief(title="群像匹配", story="多人电力试炼"))
        lin = CharacterProfile(name="林砚")
        glasses = CharacterProfile(name="戴眼镜男生（未具名）")
        monitor = CharacterProfile(name="女试炼者A（胡浩监护人）")
        unrelated = CharacterProfile(name="张峰")
        project.characters = [lin, glasses, monitor, unrelated]
        for character in project.characters:
            path = self.root / f"{character.id}.png"
            path.write_bytes(character.name.encode("utf-8"))
            asset = self.service.register_existing_asset(
                project.id,
                path,
                AssetRole.CHARACTER,
                f"{character.name}参考",
            )
            asset.character_id = character.id
            self.store.save_asset(asset)
            character.reference_asset_ids = [asset.id]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            title="林砚听戴眼镜男生解释",
            scene_description="林砚与戴眼镜的男生站在一起，女试炼者A在后方观察",
            character_ids=[lin.id],
            reference_asset_ids=[lin.reference_asset_ids[0]],
        )

        self.service.synchronize_shot_cast(project, shot)

        self.assertEqual(shot.character_ids, [lin.id, glasses.id, monitor.id])
        self.assertEqual(
            shot.reference_asset_ids,
            [lin.reference_asset_ids[0], glasses.reference_asset_ids[0], monitor.reference_asset_ids[0]],
        )
        keyframe_ids = self.service.keyframe_character_ids(project, shot, shot.scene_description)
        self.assertEqual(keyframe_ids, [lin.id, glasses.id, monitor.id])
        seedance_prompt = self.service.compile_seedance_prompt(
            project,
            shot,
            self.store.list_assets(project.id),
        )
        self.assertIn("定义为“戴眼镜男生（未具名）”", seedance_prompt)
        self.assertIn("定义为“女试炼者A（胡浩监护人）”", seedance_prompt)
        self.assertNotIn("定义为“张峰”", seedance_prompt)

    def test_storyboard_uses_explicit_scene_character_names_as_cast_source(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="显式角色数组", story="两个角色在车站交谈", target_duration_seconds=8)
        )
        first = CharacterProfile(name="阿青")
        second = CharacterProfile(name="小满（成年期）")
        project.characters = [first, second]
        self.store.save_project(project)

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                return Storyboard(
                    topic="车站",
                    scenes=[
                        Scene(
                            id=1,
                            narrative="两人站在月台",
                            visual_prompt="两位角色同框，但画面文字不重复姓名",
                            motion_prompt="两人轻轻点头",
                            character_names=["阿青", "小满"],
                            duration=8,
                        )
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 1, "manual"))

        self.assertEqual(shots[0].character_ids, [first.id, second.id])

    def test_storyboard_preserves_unregistered_character_names_as_recoverable_draft(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="角色校验", story="阿青在车站等待", target_duration_seconds=8)
        )
        project.characters = [CharacterProfile(name="阿青")]
        self.store.save_project(project)
        calls = 0

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                nonlocal calls
                calls += 1
                return Storyboard(
                    topic="车站",
                    scenes=[
                        Scene(
                            id=1,
                            narrative="短发男生在车站等待",
                            visual_prompt="短发男生近景",
                            motion_prompt="短发男生抬头",
                            character_names=["短发男生"],
                            duration=8,
                        )
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 1, "manual"))

        self.assertEqual(calls, 1)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].character_ids, [])
        saved_project = self.service.require_project(project.id)
        self.assertTrue(saved_project.storyboard_warnings)
        self.assertIn("短发男生", saved_project.storyboard_warnings[0])

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

    def test_asset_prompt_refresh_drops_deleted_style_reference(self) -> None:
        project = self.service.create_project(ProjectBrief(title="删除风格图", story="人物走入房间"))
        style_path = self.root / "style.png"
        style_path.write_bytes(b"style")
        style_asset = self.service.register_existing_asset(project.id, style_path, AssetRole.STYLE, "已删除风格图")
        project.style_profile = StyleProfile(
            name="旧风格",
            medium="电影写实",
            reference_asset_ids=[style_asset.id],
            approved=True,
        )
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="人物走入房间",
            scene_description="办公室",
            subject_motion="人物推门进入",
            seedance_prompt="旧缓存 @图片1（已删除风格图）",
            h3_prompt_skill_output="旧付费 H3",
            h3_director_version="h3-director-v3",
        )
        self.store.save_shot(shot)

        self.store.delete_asset(style_asset.id)
        project.style_profile.reference_asset_ids = []
        self.store.save_project(project)
        self.service.refresh_shot_prompt_caches(project.id)

        refreshed = self.service.require_shot(shot.id)
        self.assertNotIn("已删除风格图", refreshed.seedance_prompt)
        self.assertEqual(refreshed.h3_prompt_skill_output, "旧付费 H3")
        self.assertEqual(refreshed.h3_director_version, "h3-director-v3")
        self.assertLess(refreshed.h3_prompt_source_revision, refreshed.content_revision)
        self.assertEqual(refreshed.content_revision, 2)

    def test_storyboard_defaults_to_local_seedance_compile_without_h3_call(self) -> None:
        project = self.service.create_project(ProjectBrief(title="默认 Prompt", story="人物抬头", target_duration_seconds=6))

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                return Storyboard(scenes=[Scene(event="人物抬头看向光源", opening_state="人物站在窗边", duration=6)])

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with (
            patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator),
            patch.object(self.service, "generate_h3_prompts", side_effect=AssertionError("H3 should be on demand")),
        ):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 1, "manual"))

        self.assertEqual(self.service.require_project(project.id).preferred_prompt_targets, ["seedance"])
        self.assertEqual(shots[0].h3_prompt_skill_output, "")
        self.assertTrue(shots[0].seedance_prompt)

    def test_style_reference_analysis_can_atomically_fill_project_style(self) -> None:
        project = self.service.create_project(
            ProjectBrief(
                title="风格参考回填",
                story="电力检修员在高压线路旁作业",
                negative_prompt="不要字幕",
            )
        )
        reference = self.root / "style-reference.png"
        reference.write_bytes(b"image")
        asset = self.service.register_existing_asset(project.id, reference, AssetRole.STYLE, "客户风格图")

        class FakeLLM:
            async def generate_json(self, _system: str, _prompt: str, reference_images: list[str] | None = None):
                self.reference_images = reference_images
                return {
                    "name": "冷峻工业电影写实",
                    "medium": "高细节电影级 3D 写实",
                    "palette": "低饱和蓝灰主色，安全橙点缀",
                    "lighting": "阴天漫射冷光与轮廓侧光",
                    "camera_language": "中长焦、中近景、稳定缓推",
                    "composition": "人物位于三分线，电网结构形成纵深",
                    "texture": "防护服织物与金属设备纹理清晰",
                    "motion_language": "克制连续动作，少量稳定跟拍",
                    "analysis_summary": "全项目保持冷峻工业电影写实，不改变人物与设备材质规则。",
                    "negative_constraints": "避免卡通化、霓虹色、设备结构变形",
                    "confidence": 0.91,
                    "observations": ["画面整体为蓝灰冷色", "人物防护服和电网金属具有写实纹理"],
                }

        fake = FakeLLM()
        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=fake):
            draft = asyncio.run(self.service.analyze_style(project.id, [asset.id], apply=True))

        saved = self.service.require_project(project.id)
        self.assertTrue(draft.approved)
        self.assertIsNotNone(saved.style_profile)
        self.assertTrue(saved.style_profile.approved)
        self.assertEqual(saved.style_profile.reference_asset_ids, [asset.id])
        self.assertIn("冷峻工业电影写实", saved.brief.visual_style)
        self.assertIn("【镜头语言】中长焦、中近景、稳定缓推", saved.style_bible)
        self.assertIn("不要字幕", saved.brief.negative_prompt)
        self.assertIn("设备结构变形", saved.brief.negative_prompt)
        self.assertEqual(fake.reference_images, [str(reference.resolve())])

    def test_optional_scene_profile_and_seedance_tail_continuity(self) -> None:
        project = self.service.create_project(ProjectBrief(title="continuity", story="三分钟谈话"))
        scene_path = self.root / "interview-room.png"
        tail_path = self.root / "tail.png"
        keyframe_path = self.root / "old-keyframe.png"
        scene_path.write_bytes(b"scene")
        tail_path.write_bytes(b"tail")
        keyframe_path.write_bytes(b"old-keyframe")
        scene_asset = self.service.register_existing_asset(project.id, scene_path, AssetRole.SCENE, "谈话室母版")
        tail_asset = self.service.register_existing_asset(project.id, tail_path, AssetRole.LAST_FRAME, "上一镜尾帧")
        keyframe_asset = self.service.register_existing_asset(project.id, keyframe_path, AssetRole.KEYFRAME, "本镜旧首帧")
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
        self.assertIsNone(self.service.seedance_first_frame_asset_id(project, second, assets))

        second.use_scene_profile = True
        with_scene = self.service.seedance_reference_assets(second, assets, project)
        self.assertEqual([asset.id for asset in with_scene], [tail_asset.id, scene_asset.id])

        # Even when the user previously made a manual full-modal selection,
        # continuous mode replaces the stale current-shot keyframe with the
        # source shot's adopted tail while preserving the other selected refs.
        second.keyframe_asset_id = keyframe_asset.id
        second.video_reference_asset_ids = [keyframe_asset.id, scene_asset.id]
        manual_continuous = self.service.seedance_reference_assets(second, assets, project)
        self.assertEqual([asset.id for asset in manual_continuous], [tail_asset.id, scene_asset.id])
        prompt = self.service.compile_seedance_prompt(project, second, assets)
        self.assertIn("图片1是上一镜真实尾帧", prompt)
        self.assertIn("图片2", prompt)

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

    def test_ai_insert_adds_only_one_shot_and_atomically_renumbers_tail(self) -> None:
        project = self.service.create_project(ProjectBrief(title="补镜", story="前因后果"))
        scene_profile = SceneProfile(name="地府大厅", description="幽蓝地府办事大厅", continuity_notes="柜台和排队护栏固定")
        project.scene_profiles = [scene_profile]
        self.store.save_project(project)
        first = self.store.save_shot(Shot(project_id=project.id, ordinal=1, title="原镜一", narrative="起因"))
        second = self.store.save_shot(Shot(project_id=project.id, ordinal=2, title="原镜二", narrative="原结尾", video_path="kept.mp4"))

        class FakeLLM:
            async def generate_json(self, *_: object, **__: object) -> dict[str, object]:
                return {
                    "title": "地府排队反转",
                    "duration_seconds": 8,
                    "event": "镜头切回地府大厅，人挤人的队伍一直延伸到远处",
                    "opening_state": "地府大厅大全景，密集人群沿护栏排队",
                    "scene_profile_name": "地府大厅",
                    "character_names": [],
                    "shot_size": "大全景",
                    "camera_angle": "高机位",
                    "lens": "24mm广角",
                    "camera_motion": "缓慢后拉",
                    "subject_motion": "队伍不断向画面深处延伸",
                    "visual_beats": [
                        {"start_seconds": 0, "end_seconds": 4, "purpose": "揭示", "subject_action": "前景人群挪动", "camera_motion": "缓慢后拉"},
                        {"start_seconds": 4, "end_seconds": 8, "purpose": "反转", "subject_action": "露出看不到尽头的队伍", "camera_motion": "固定镜头"},
                    ],
                    "voice_events": [],
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            inserted = asyncio.run(
                self.service.insert_shot_with_ai(project.id, first.id, "插一个地府人挤人排队的结尾反转", [])
            )

        shots = self.store.list_shots(project.id)
        self.assertEqual([shot.id for shot in shots], [first.id, inserted.id, second.id])
        self.assertEqual([shot.ordinal for shot in shots], [1, 2, 3])
        self.assertEqual(shots[2].video_path, "kept.mp4")
        self.assertEqual(inserted.title, "地府排队反转")
        self.assertTrue(inserted.use_scene_profile)
        self.assertEqual(inserted.scene_profile_id, scene_profile.id)
        refreshed = self.service.require_project(project.id)
        self.assertIn(inserted.id, refreshed.scene_profiles[0].source_shot_ids)

    def test_changing_seedance_reference_mode_recompiles_material_numbers(self) -> None:
        project = self.service.create_project(ProjectBrief(title="参考策略", story="角色稍后入画"))
        character = CharacterProfile(name="张小差", wardrobe="炭黑工装", reference_asset_ids=["character-ref"])
        project.characters = [character]
        self.store.save_project(project)
        (self.root / "keyframe.png").write_bytes(b"keyframe")
        (self.root / "character.png").write_bytes(b"character")
        keyframe = self.service.register_existing_asset(
            project.id,
            self.root / "keyframe.png",
            AssetRole.KEYFRAME,
            name="首帧",
        )
        character_ref = self.service.register_existing_asset(
            project.id,
            self.root / "character.png",
            AssetRole.CHARACTER,
            name="张小差四视图",
        )
        character_ref.character_id = character.id
        self.store.save_asset(character_ref)
        project = self.service.require_project(project.id)
        project.characters[0].reference_asset_ids = [character_ref.id]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="张小差随后从墙里钻出",
            scene_description="空墙和客厅",
            keyframe_prompt="【首帧画面】空墙和客厅",
            character_ids=[character.id],
            keyframe_asset_id=keyframe.id,
            reference_asset_ids=[character_ref.id],
            seedance_reference_mode=SeedanceReferenceMode.STRICT_FIRST_FRAME,
        )
        shot.seedance_prompt = self.service.compile_seedance_prompt(project, shot, self.store.list_assets(project.id))
        shot.seedance_prompt_version = "seedance-2.0-director-v6"
        saved = self.store.save_shot(shot)
        original_revision = saved.content_revision

        incoming = saved.model_copy(deep=True)
        incoming.seedance_reference_mode = SeedanceReferenceMode.MULTIMODAL_REFERENCE
        updated = self.service.update_shot(project.id, incoming)

        self.assertEqual(updated.content_revision, original_revision)
        self.assertIn("张小差四视图", updated.seedance_prompt)
        self.assertIn("@图片2", updated.seedance_prompt)

    def test_seedance_multimodal_startup_migration_updates_legacy_shots_once(self) -> None:
        project = self.service.create_project(ProjectBrief(title="旧项目", story="旧分镜"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="旧镜头",
            seedance_reference_mode=SeedanceReferenceMode.STRICT_FIRST_FRAME,
            seedance_prompt="旧版提示词",
            seedance_prompt_version="seedance-2.0-director-v5",
        )
        self.store.save_shot(shot)

        first = self.service.migrate_all_seedance_shots_to_multimodal()
        migrated = self.service.require_shot(shot.id)
        second = self.service.migrate_all_seedance_shots_to_multimodal()

        self.assertEqual(first, {"project_count": 1, "shot_count": 1})
        self.assertEqual(second, {"project_count": 0, "shot_count": 0})
        self.assertEqual(migrated.seedance_reference_mode, SeedanceReferenceMode.MULTIMODAL_REFERENCE)
        self.assertEqual(migrated.seedance_prompt_version, "seedance-2.0-director-v6")

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
        previous_auto_segment = settings.H3_AUTO_SEGMENT_COMPLEX_SHOTS
        settings.H3_AUTO_SEGMENT_COMPLEX_SHOTS = True
        try:
            segments = self.service.h3_segment_plan(project, planned, [])
        finally:
            settings.H3_AUTO_SEGMENT_COMPLEX_SHOTS = previous_auto_segment
        self.assertEqual(len(segments), 3)
        self.assertEqual(sum(bool(segment["has_dialogue"]) for segment in segments), 1)
        self.assertAlmostEqual(sum(float(segment["duration"]) for segment in segments), 11.5, places=2)
        self.assertTrue(all("不得把对白或旁白自动转成字幕" in str(segment["prompt"]) for segment in segments))
        silent_segments = [segment for segment in segments if not segment["has_dialogue"]]
        self.assertTrue(silent_segments)
        self.assertTrue(all("全程嘴唇闭合" in str(segment["prompt"]) for segment in silent_segments))
        self.assertTrue(all("C同志要求秦绍辉" not in str(segment["prompt"]) for segment in silent_segments))
        speaking_segment = next(segment for segment in segments if segment["has_dialogue"])
        self.assertEqual(speaking_segment["dialogue"], shot.dialogue)
        self.assertEqual(speaking_segment["speaker_id"], speaker.id)
        self.assertIn("唯一发言者明确为C同志", str(speaking_segment["prompt"]))
        self.assertIn("秦绍辉全程闭嘴", str(speaking_segment["prompt"]))

    def test_spoken_dialogue_text_keeps_colon_ending_fragments(self) -> None:
        # Real dialogue fragments ending with a colon must survive; only short
        # speaker labels such as “秦绍辉同志：” may be stripped.
        self.assertEqual(ProjectService.spoken_dialogue_text("登入高危电力副本："), "登入高危电力副本：")
        self.assertEqual(ProjectService.spoken_dialogue_text("秦绍辉同志：请把手机放好。"), "请把手机放好。")
        self.assertEqual(ProjectService.spoken_dialogue_text("旁白：欢迎试炼者"), "欢迎试炼者")
        self.assertEqual(ProjectService.spoken_dialogue_text("【旁白】：欢迎试炼者"), "欢迎试炼者")

    def test_segment_plan_merges_punctuation_only_fragments(self) -> None:
        project = self.service.create_project(ProjectBrief(title="副本", story="系统播报任务"))
        speaker = CharacterProfile(name="系统", voice_description="电子合成音")
        project.characters = [speaker]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="系统播报副本任务",
            dialogue="欢迎试炼者073，登入高危电力副本：城郊老旧线路杆上抢修。",
            duration_seconds=15.0,
            subject_motion="起势：系统播报；发展：人物抬头；收束：人物出发",
            character_ids=[speaker.id],
        )
        self.store.save_shot(shot)
        segments = self.service.h3_segment_plan(project, shot, [])
        self.assertGreaterEqual(len(segments), 2)
        for segment in segments:
            text = str(segment["dialogue"]).strip("，。！？!?；;、 ")
            self.assertTrue(text, segment["dialogue"])
            self.assertTrue(ProjectService.spoken_dialogue_text(str(segment["dialogue"])))

    def test_hosted_segment_plan_keeps_15s_shot_single(self) -> None:
        # Hosted APIs render up to 15 continuous seconds per call, so the same
        # shot that ComfyUI splits into dialogue segments must stay one piece.
        project = self.service.create_project(ProjectBrief(title="副本", story="系统播报任务"))
        speaker = CharacterProfile(name="系统", voice_description="电子合成音")
        project.characters = [speaker]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="系统播报副本任务",
            dialogue="欢迎试炼者073，登入高危电力副本：城郊老旧线路杆上抢修。",
            duration_seconds=15.0,
            subject_motion="起势：系统播报；发展：人物抬头；收束：人物出发",
            character_ids=[speaker.id],
        )
        self.store.save_shot(shot)
        segments = self.service.h3_segment_plan(project, shot, [], max_segment_seconds=15.0)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(float(segments[0]["duration"]), 15.0, places=2)
        self.assertIn("欢迎试炼者073", str(segments[0]["dialogue"]))
        self.assertIn("城郊老旧线路杆上抢修", str(segments[0]["dialogue"]))

    def test_hosted_h3_clean_tts_reads_same_source_system_voice_events(self) -> None:
        project = self.service.create_project(ProjectBrief(title="系统播报", story="系统发布规则"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="系统发布精简规则",
            dialogue="",
            duration_seconds=15.0,
            voice_events=[
                VoiceEvent(
                    kind="system_vo",
                    speaker_name="系统",
                    text="进入高危电力副本。",
                    start_seconds=1.5,
                    end_seconds=4.0,
                ),
                VoiceEvent(
                    kind="system_vo",
                    speaker_name="系统",
                    text="双人登杆，一人作业，一人监护。",
                    start_seconds=4.2,
                    end_seconds=8.0,
                ),
            ],
        )

        segments = self.service.h3_segment_plan(
            project,
            shot,
            [],
            max_segment_seconds=15.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0]["has_dialogue"])
        self.assertIn("进入高危电力副本", str(segments[0]["dialogue"]))
        self.assertIn("一人监护", str(segments[0]["dialogue"]))
        self.assertEqual(segments[0]["speaker_id"], None)
        self.assertEqual(len(segments[0]["voice_events"]), 2)

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
        previous_audio = (settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE, settings.TTS_PROVIDER)
        settings.H3_POSTPROCESS_AUDIO = True
        settings.H3_AUDIO_MODE = "clean_tts"
        settings.TTS_PROVIDER = "edge"
        try:
            segments = self.service.h3_segment_plan(project, shot, [keyframe, qin, d_ref, c_ref])
        finally:
            settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE, settings.TTS_PROVIDER = previous_audio
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

    def test_character_voice_anchor_is_shared_by_seedance_h3_and_clean_tts(self) -> None:
        project = self.service.create_project(ProjectBrief(title="声音锚点", story="青年检修员抱怨"))
        speaker = CharacterProfile(
            name="李强",
            description="22岁左右东亚男性，体型中等",
            voice_description="年轻男性声线，语气毛躁缺乏耐心，说话节奏偏快，带着明显不耐烦",
        )
        project.characters = [speaker]
        self.store.save_project(project)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="李强敷衍检查电杆",
            dialogue="",
            duration_seconds=8,
            character_ids=[speaker.id],
            voice_events=[
                VoiceEvent(
                    kind="character",
                    speaker_id=speaker.id,
                    speaker_name=speaker.name,
                    text="真是麻烦。",
                    start_seconds=2,
                    end_seconds=4,
                    lip_sync=True,
                )
            ],
        )

        seedance = self.service.compile_seedance_prompt(project, shot, [])
        h3 = self.service.compile_h3_prompt(project, shot, [])
        profile = DialogueAudioService.profile_for(project, shot.model_copy(update={
            "dialogue": "真是麻烦。",
            "dialogue_speaker_id": speaker.id,
        }))

        self.assertIn(speaker.voice_description, seedance)
        self.assertIn("保持同一音色、年龄感、性别感、音高和口音", seedance)
        self.assertNotIn(speaker.voice_description, h3)
        self.assertIn("young adult male voice", h3)
        self.assertIn("a restless, impatient edge", h3)
        self.assertIn("Only the exact text inside existing <d>...</d> tags may become speech", h3)
        self.assertEqual(profile.voice, "zh-CN-YunjianNeural")
        self.assertEqual(profile.selection_reason, "young_forceful_male")
        self.assertGreater(profile.rate_percent, 0)
        self.assertLess(profile.pitch_hz, 0)
        self.assertEqual(profile.voice_anchor, speaker.voice_description)

    def test_voice_event_name_repairs_a_stale_speaker_id_before_prompt_and_tts(self) -> None:
        project = self.service.create_project(ProjectBrief(title="说话人修复", story="女监护人提醒胡浩"))
        male = CharacterProfile(name="胡浩", voice_description="松散随意的年轻男性声线")
        female = CharacterProfile(name="女试炼者A（胡浩监护人）", voice_description="清亮偏脆的年轻女声，语气急切")
        project.characters = [male, female]
        event = VoiceEvent(
            kind="character",
            speaker_id=male.id,
            speaker_name=female.name,
            text="你拉线还没做拉力测试！",
            start_seconds=1,
            end_seconds=4,
            lip_sync=True,
        )

        resolved = self.service._resolve_voice_events(project, [event], 8, 8)
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            duration_seconds=8,
            character_ids=[male.id, female.id],
            voice_events=[event],
        )
        seedance = self.service.compile_seedance_prompt(project, shot, [])
        h3 = self.service.compile_h3_prompt(project, shot, [])

        self.assertEqual(resolved[0].speaker_id, female.id)
        self.assertIn(female.voice_description, seedance)
        self.assertNotIn(female.voice_description, h3)
        self.assertIn("young adult female voice", h3)
        self.assertIn("urgent emotional intent", h3)
        self.assertNotIn(f"声音锚点：{male.voice_description}", seedance)

    def test_native_h3_audio_mode_leaves_generated_file_untouched(self) -> None:
        video = self.root / "native.mp4"
        video.write_bytes(b"native-h3-audio-fixture")
        previous_audio = (settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE)
        settings.H3_POSTPROCESS_AUDIO = True
        settings.H3_AUDIO_MODE = "native"
        try:
            delivery = RenderQueue._replace_generated_audio(video, None, 0.0)
        finally:
            settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE = previous_audio

        self.assertFalse(delivery["applied"])
        self.assertEqual(delivery["mode"], "native")
        self.assertEqual(video.read_bytes(), b"native-h3-audio-fixture")
        self.assertFalse((self.root / "native.native-audio.mp4").exists())

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
        native_backup = Path(str(delivery["native_h3_backup"]))
        self.assertTrue(native_backup.exists())
        self.assertEqual(native_backup.name, "native-noise.native-audio.mp4")
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

    def test_project_cover_uses_context_references_suggestion_and_ratio(self) -> None:
        project = self.service.create_project(ProjectBrief(
            title="排插安全科普",
            story="张小差发现老奶奶把取暖器、电视机和手机充电器插在同一个老化排插上。",
            aspect_ratio="16:9",
            visual_style="中式奇幻轻喜剧手绘插画",
        ))
        lead = CharacterProfile(
            name="张小差",
            description="年轻女性阴差，表情生动",
            wardrobe="黑色制服和 404 工牌",
        )
        character_path = self.root / "zhang.png"
        style_path = self.root / "style.png"
        prop_path = self.root / "power-strip.png"
        keyframe_path = self.root / "keyframe.png"
        old_cover_path = self.root / "old-cover.png"
        for path in (character_path, style_path, prop_path, keyframe_path, old_cover_path):
            path.write_bytes(path.stem.encode("utf-8"))
        character_asset = self.service.register_existing_asset(
            project.id, character_path, AssetRole.CHARACTER, "张小差四视图"
        )
        character_asset.character_id = lead.id
        self.store.save_asset(character_asset)
        style_asset = self.service.register_existing_asset(
            project.id, style_path, AssetRole.STYLE, "科普插画画风"
        )
        prop_asset = self.service.register_existing_asset(
            project.id, prop_path, AssetRole.PROP, "老化排插"
        )
        keyframe_asset = self.service.register_existing_asset(
            project.id, keyframe_path, AssetRole.KEYFRAME, "过载排插首帧"
        )
        old_cover = self.service.register_existing_asset(
            project.id, old_cover_path, AssetRole.COVER, "旧封面"
        )
        lead.reference_asset_ids = [character_asset.id]
        project.characters = [lead]
        project.style_profile = StyleProfile(
            name="项目画风",
            medium="数字手绘",
            reference_asset_ids=[style_asset.id],
            approved=True,
        )
        self.store.save_project(project)
        self.store.save_shot(Shot(
            project_id=project.id,
            ordinal=1,
            title="排插过载",
            narrative="张小差震惊地指向冒火花的排插。",
            character_ids=[lead.id],
            reference_asset_ids=[prop_asset.id],
            keyframe_asset_id=keyframe_asset.id,
        ))
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(
                self,
                scene: Scene,
                output_dir: str,
                reference_image_path: str | None = None,
                **kwargs: object,
            ) -> str:
                captured.update({"scene": scene, "references": reference_image_path, **kwargs})
                output = Path(output_dir) / "cover.png"
                output.write_bytes(b"generated-cover")
                return str(output)

        suggestion = "主标题改成“别让排插变火龙”，突出张小差的震惊表情"
        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            generated = asyncio.run(self.service.generate_project_cover(
                project.id,
                user_suggestions=suggestion,
                aspect_ratio="16:9",
            ))

        prompt = captured["scene"].visual_prompt  # type: ignore[union-attr]
        references = str(captured["references"]).split(",")
        self.assertEqual(generated.role, AssetRole.COVER)
        self.assertEqual(captured["aspect_ratio"], "16:9")
        self.assertIn("cover-ratio:16:9", generated.tags)
        self.assertIn(suggestion, prompt)
        self.assertIn("排插安全科普", prompt)
        self.assertIn("顶部大标题", prompt)
        self.assertEqual(
            references,
            [
                str(character_path.resolve()),
                str(style_path.resolve()),
                str(prop_path.resolve()),
                str(keyframe_path.resolve()),
            ],
        )
        self.assertNotIn(str(old_cover_path.resolve()), references)
        self.assertEqual(len(self.store.list_assets(project.id)), 6)
        self.assertTrue(resolve_media_path(generated.path).is_file())

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
            video_reference_asset_ids=[previous.id, character.id],
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
            self.assertNotIn(previous.id, iterated[0].video_reference_asset_ids or [])
            self.assertIn(iterated_asset.id, iterated[0].video_reference_asset_ids or [])  # type: ignore[union-attr]
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
        self.assertNotIn(iterated_asset.id, fresh[0].video_reference_asset_ids or [])  # type: ignore[union-attr]
        self.assertIn(fresh[0].keyframe_asset_id, fresh[0].video_reference_asset_ids or [])

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
        self.assertEqual(jobs[0].input_snapshot["duration_seconds"], shot.duration_seconds)
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
        previous_provider = settings.H3_PROVIDER
        previous_audio_mode = settings.H3_AUDIO_MODE
        manager = RuntimeSettingsManager()
        try:
            manager.update({"DEEPSEEK_API_KEY": "test-secret-1234", "DEEPSEEK_MODEL": "test-model"})
            settings.H3_PROVIDER = "comfyui_h3"
            settings.H3_AUDIO_MODE = "native"
            payload = manager.public_payload()
            fields = {field["key"]: field for group in payload["groups"] for field in group["fields"]}
            self.assertEqual(fields["DEEPSEEK_API_KEY"]["value"], "")
            self.assertEqual(fields["DEEPSEEK_API_KEY"]["masked"], "••••1234")
            self.assertEqual(settings.DEEPSEEK_MODEL, "test-model")
            self.assertEqual(manager.path.stat().st_mode & 0o777, 0o600)
            routes = {route["id"]: route for route in payload["routes"]}
            self.assertEqual(routes["storyboard"]["setting_key"], "LLM_PROVIDER")
            self.assertEqual(routes["h3_video"]["effective_label"], "MiniMax H3 / ComfyUI")
            self.assertEqual(routes["dialogue_audio"]["selected"], "native")
            self.assertEqual(routes["dialogue_audio"]["options"][0]["value"], "native")
            self.assertIn("默认直接交付", routes["dialogue_audio"]["description"])
            self.assertIsNone(routes["finalize"]["setting_key"])
        finally:
            settings.DEEPSEEK_API_KEY = previous_key
            settings.DEEPSEEK_MODEL = previous_model
            settings.H3_PROVIDER = previous_provider
            settings.H3_AUDIO_MODE = previous_audio_mode

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

    def test_brief_analysis_normalizes_common_claude_json_variations(self) -> None:
        payload = _normalize_project_analysis_payload(
            {
                "visual_style": ["电影写实", "冷蓝色调"],
                "recommended_shot_count": "建议 12 镜",
                "characters": {
                    "林砚": {
                        "appearance": ["青年", "短发"],
                        "costume": "蓝色工装",
                        "voice": "语速沉稳",
                    }
                },
                "analysis_notes": "确认安全带型号；\n确认现场天气",
            },
            8,
        )
        self.assertEqual(payload["recommended_shot_count"], 12)
        self.assertEqual(payload["visual_style"], "电影写实；冷蓝色调")
        self.assertEqual(payload["characters"][0]["name"], "林砚")
        self.assertEqual(payload["characters"][0]["description"], "青年；短发")
        self.assertEqual(payload["analysis_notes"], ["确认安全带型号", "确认现场天气"])

    def test_brief_analysis_turns_flagged_animal_into_editable_character_default(self) -> None:
        payload = _normalize_project_analysis_payload(
            {
                "characters": [],
                "analysis_notes": ["需确认剧情中出现的老黄狗是否需要单独设定形象"],
            },
            4,
        )
        self.assertEqual(payload["characters"][0]["name"], "老黄狗")
        self.assertIn("固定形象", payload["characters"][0]["description"])
        self.assertEqual(
            payload["analysis_notes"],
            ["已默认将老黄狗作为独立固定形象建档；如不需要，可在上方角色草稿中删除。"],
        )

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
