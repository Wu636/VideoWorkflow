from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from src.video_workflow.domain import (
    AssetRole,
    CharacterProfile,
    FirstFrameCompleteness,
    ProjectBrief,
    SceneProfile,
    SeedanceReferenceMode,
    Shot,
    StyleProfile,
)
from src.video_workflow.generators.image import GrsaiImageGenerator, ImageDownloadError, download_generated_image
from src.video_workflow.config import settings
from src.video_workflow.services.projects import ProjectService, SceneReferenceConflictError
from src.video_workflow.storage import ProjectStore
from src.video_workflow.types import Scene

PNG = b"\x89PNG\r\n\x1a\nfixture-image"


class SceneReferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root / "state.sqlite3")
        self.service = ProjectService(self.store)
        class EmptySetVision:
            async def generate_json(_, *_args, **_kwargs):
                return {"has_people": False}
        self.service._scene_reference_vision_generator = lambda: EmptySetVision()
        self.service.project_dir = lambda project_id: self.root / project_id
        self.project = self.service.create_project(ProjectBrief(title="场景测试", story="家中与地府"))
        self.home = SceneProfile(name="奶奶家", description="普通客厅，正常室内灯", continuity_notes="保留沙发位置")
        self.office = SceneProfile(name="地府", description="地府冷色环境光")
        self.project.scene_profiles = [self.home, self.office]
        self.project.style_profile = StyleProfile(name="插画", medium="手绘笔触", lighting="全片地府蓝光", composition="全片双人物", approved=True)
        self.store.save_project(self.project)
        self.shot = self.store.save_shot(Shot(project_id=self.project.id, ordinal=1,
            scene_profile_id=self.home.id, h3_prompt_skill_output="PAID H3", video_prompt="PAID H3"))

    def tearDown(self):
        self.temp.cleanup()

    def latest(self, scene_id=None):
        return next(item for item in ProjectStore(self.root / "state.sqlite3").get_project(self.project.id).scene_profiles if item.id == (scene_id or self.home.id))

    def test_edit_persists_and_rejects_stale_version_without_touching_shots(self):
        before = self.store.get_shot(self.shot.id).model_dump()
        saved = self.service.update_scene_profile(self.project.id, self.home.id,
            {"name": "新客厅", "description": "正常窗光", "continuity_notes": "桌子不动", "reference_prompt": "USER PROMPT"}, 1)
        self.assertEqual(self.latest().reference_prompt, "USER PROMPT")
        self.assertEqual(self.latest().version, 2)
        self.assertEqual(self.service.scene_reference_prompt(self.project.id, self.home.id)["prompt"], "USER PROMPT")
        self.assertEqual(self.store.get_shot(self.shot.id).model_dump(), before)
        with self.assertRaises(SceneReferenceConflictError):
            self.service.update_scene_profile(self.project.id, self.home.id, {"name": "旧表单"}, 1)
        # An unrelated stale whole-project save must preserve newer scene work.
        self.project.brief.title = "另一个表单保存"
        self.service.update_project(self.project)
        self.assertEqual(self.latest().model_dump(), saved.model_dump())
        self.project.scene_profiles = []  # form opened before scene creation
        self.service.update_project(self.project)
        self.assertEqual(self.latest().model_dump(), saved.model_dump())

    def test_default_prompt_uses_local_scene_without_global_lighting_or_cast(self):
        prompt = self.service.scene_reference_prompt(self.project.id, self.home.id)["prompt"]
        self.assertIn("正常室内灯", prompt)
        self.assertIn("手绘笔触", prompt)
        self.assertNotIn("全片地府蓝光", prompt)
        self.assertNotIn("全片双人物", prompt)
        self.assertEqual(self.latest().reference_prompt, "")  # preview is read-only

    def test_old_scene_profile_removes_cast_and_human_style_before_generation(self):
        project = self.service.require_project(self.project.id)
        project.characters = [CharacterProfile(name="陆峥"), CharacterProfile(name="林晓")]
        project.scene_profiles[0].description = (
            "中央为弧形主控屏，前方是操作台；陆峥、林晓围绕操作台活动。"
            "环境光以冷蓝为主，屏幕光映照人物与操作台。"
            "环境与道具保留写实质感，控制台具有金属质感。"
        )
        project.scene_profiles[0].continuity_notes = "保持三人相对站位；弧形大屏与操作台方位一致"
        project.style_profile.medium = "次世代 PBR；真人向写实动画质感"
        self.store.save_project(project)

        default_prompt = self.service.compile_scene_reference_prompt(project, project.scene_profiles[0])
        positive = default_prompt.split("【环境场景图硬约束")[0]
        self.assertIn("主控屏", positive)
        self.assertIn("操作台", positive)
        self.assertIn("PBR", positive)
        for person_detail in ("陆峥", "林晓", "人物", "站位", "真人"):
            self.assertNotIn(person_detail, positive)
        self.assertIn("绝对不出现任何人物", default_prompt)

    async def test_detected_people_trigger_one_correction_and_only_bind_clean_image(self):
        seen: list[tuple[str | None, str]] = []
        checked: list[str] = []

        class Vision:
            async def generate_json(_, _system, _prompt, reference_images):
                checked.extend(reference_images)
                return {"has_people": len(checked) == 1}

        class Generator:
            async def generate_image(_, scene, output_dir, reference_image_path, **_kwargs):
                seen.append((reference_image_path, scene.visual_prompt))
                output = Path(output_dir) / "result.png"
                output.write_bytes(PNG)
                return str(output)

        self.service._scene_reference_vision_generator = lambda: Vision()
        with patch("src.video_workflow.services.projects.create_image_generator", return_value=Generator()):
            clean_asset = await self.service.generate_scene_reference(self.project.id, self.home.id)

        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0][0])
        self.assertEqual(seen[1][0], checked[0])
        self.assertIn("彻底擦除所有人物", seen[1][1])
        self.assertEqual(self.latest().reference_asset_ids, [clean_asset.id])
        self.assertEqual(len(self.store.list_assets(self.project.id)), 2)
        self.assertTrue(any("含人物未绑定" in item.name for item in self.store.list_assets(self.project.id)))

    async def test_scene_master_passes_uploaded_style_image_on_both_attempts(self):
        style_path = self.root / "style.png"
        style_path.write_bytes(PNG)
        style_asset = self.service.register_existing_asset(self.project.id, style_path, AssetRole.STYLE, "用户画风")
        project = self.service.require_project(self.project.id)
        project.style_profile.reference_asset_ids = [style_asset.id]
        self.store.save_project(project)
        seen = []

        class Vision:
            def __init__(self):
                self.calls = 0

            async def generate_json(self, *_args, **_kwargs):
                self.calls += 1
                return {"has_people": self.calls == 1}

        class Generator:
            async def generate_image(_, scene, output_dir, reference_image_path, **kwargs):
                seen.append((reference_image_path, kwargs.get("style_reference_image_path"), scene.visual_prompt))
                output = Path(output_dir) / "result.png"
                output.write_bytes(PNG)
                return str(output)

        vision = Vision()
        self.service._scene_reference_vision_generator = lambda: vision
        with patch.object(settings, "IMAGE_STYLE_REFERENCE_MODE", "direct"), patch("src.video_workflow.services.projects.create_image_generator", return_value=Generator()):
            await self.service.generate_scene_reference(self.project.id, self.home.id)
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0][0])
        self.assertIsNotNone(seen[1][0])
        self.assertTrue(all(item[1] == str(style_path.resolve()) for item in seen))
        self.assertIn("只指导绘制媒介", seen[0][2])

    async def test_still_contains_people_after_correction_keeps_existing_reference(self):
        old_path = self.root / "previous.png"
        old_path.write_bytes(PNG)
        old_asset = self.service.register_existing_asset(self.project.id, old_path, AssetRole.SCENE, "旧母版")
        project = self.service.require_project(self.project.id)
        project.scene_profiles[0].reference_asset_ids = [old_asset.id]
        self.store.save_project(project)

        class Vision:
            async def generate_json(_, _system, _prompt, reference_images):
                return {"has_people": True}

        class Generator:
            async def generate_image(_, scene, output_dir, reference_image_path, **_kwargs):
                output = Path(output_dir) / "result.png"
                output.write_bytes(PNG)
                return str(output)

        self.service._scene_reference_vision_generator = lambda: Vision()
        with patch("src.video_workflow.services.projects.create_image_generator", return_value=Generator()):
            with self.assertRaisesRegex(Exception, "连续两次出图都检测到人物"):
                await self.service.generate_scene_reference(self.project.id, self.home.id)
        self.assertEqual(self.latest().reference_status, "failed")
        self.assertEqual(self.latest().reference_asset_ids, [old_asset.id])
        self.assertEqual(len(self.store.list_assets(self.project.id)), 3)

    def test_batch_apply_uses_scene_as_authority_and_preserves_paid_outputs(self):
        scene_path = self.root / "home.png"
        style_path = self.root / "style.png"
        keyframe_path = self.root / "keyframe.png"
        for path in (scene_path, style_path, keyframe_path):
            path.write_bytes(PNG)
        scene_asset = self.service.register_existing_asset(
            self.project.id, scene_path, AssetRole.SCENE, "奶奶家母版"
        )
        style_asset = self.service.register_existing_asset(
            self.project.id, style_path, AssetRole.STYLE, "混合场景参考"
        )
        keyframe_asset = self.service.register_existing_asset(
            self.project.id, keyframe_path, AssetRole.KEYFRAME, "旧首帧"
        )
        project = self.service.require_project(self.project.id)
        project.scene_profiles[0].source_shot_ids = [self.shot.id]
        project.scene_profiles[0].reference_asset_ids = [scene_asset.id]
        project.scene_profiles[0].approved = True
        project.style_profile.reference_asset_ids = [style_asset.id]
        self.store.save_project(project)
        shot = self.store.get_shot(self.shot.id)
        shot.keyframe_asset_id = keyframe_asset.id
        shot.scene_description = "旧描述要求全片地府蓝光"
        self.store.save_shot(shot)

        result = self.service.apply_scene_profiles_to_shots(self.project.id)
        saved = self.store.get_shot(self.shot.id)
        self.assertTrue(saved.use_scene_profile)
        self.assertEqual(saved.scene_profile_id, self.home.id)
        self.assertIn("【场景档案｜本镜最高优先级】奶奶家", saved.keyframe_prompt)
        self.assertIn("普通客厅，正常室内灯", saved.keyframe_prompt)
        self.assertIn("手绘笔触", saved.keyframe_prompt)
        self.assertNotIn("【风格锚点】插画；手绘笔触；全片地府蓝光", saved.keyframe_prompt)
        self.assertIn("【场景档案｜本镜最高优先级】", saved.seedance_prompt)
        self.assertEqual(saved.h3_prompt_skill_output, "PAID H3")
        self.assertEqual(saved.video_prompt, "PAID H3")
        self.assertEqual(saved.keyframe_asset_id, keyframe_asset.id)
        self.assertLess(saved.h3_prompt_source_revision, saved.content_revision)
        refs = self.service.keyframe_reference_assets(
            project, saved, self.store.list_assets(self.project.id), []
        )
        self.assertIn(scene_asset.id, [asset.id for asset in refs])
        self.assertNotIn(style_asset.id, [asset.id for asset in refs])
        self.assertEqual(result["applied_count"], 1)
        self.assertEqual(result["h3_preserved_count"], 1)
        self.assertEqual(result["keyframes_preserved_count"], 1)

    def test_cross_scene_shot_uses_start_for_keyframe_and_both_for_video(self):
        home_path = self.root / "home.png"
        office_path = self.root / "office.png"
        for path in (home_path, office_path):
            path.write_bytes(PNG)
        home_asset = self.service.register_existing_asset(
            self.project.id, home_path, AssetRole.SCENE, "奶奶家母版"
        )
        office_asset = self.service.register_existing_asset(
            self.project.id, office_path, AssetRole.SCENE, "地府办公室母版"
        )
        project = self.service.require_project(self.project.id)
        project.scene_profiles[0].reference_asset_ids = [home_asset.id]
        project.scene_profiles[0].approved = True
        project.scene_profiles[1].reference_asset_ids = [office_asset.id]
        project.scene_profiles[1].approved = True
        self.store.save_project(project)

        shot = self.store.get_shot(self.shot.id)
        shot.scene_profile_ids = [self.office.id, self.home.id]
        shot.scene_profile_id = self.office.id
        shot.use_scene_profile = True
        shot.seedance_reference_mode = SeedanceReferenceMode.STRICT_FIRST_FRAME
        shot.narrative = "张小差从地府办公室冲出，穿过通道后进入奶奶家客厅"
        self.store.save_shot(shot)
        project = self.service.require_project(self.project.id)
        assets = self.store.list_assets(self.project.id)

        keyframe_refs = self.service.keyframe_reference_assets(project, shot, assets, [])
        self.assertEqual([asset.id for asset in keyframe_refs], [office_asset.id])

        video_refs = self.service.seedance_reference_assets(shot, assets, project)
        self.assertEqual(
            [asset.id for asset in video_refs],
            [office_asset.id, home_asset.id],
        )
        self.assertEqual(
            self.service.resolve_seedance_reference_mode(project, shot, assets),
            SeedanceReferenceMode.MULTIMODAL_REFERENCE,
        )

        seedance_prompt = self.service.compile_seedance_prompt(project, shot, assets)
        self.assertIn("@图片1（地府办公室母版）", seedance_prompt)
        self.assertIn("@图片2（奶奶家母版）", seedance_prompt)
        self.assertIn("转场前的起始场景“地府”", seedance_prompt)
        self.assertIn("转场完成后的目标场景“奶奶家”", seedance_prompt)
        self.assertIn("禁止把两个房间拼贴在同一画面", seedance_prompt)

        # Existing paid H3 text is intentionally durable; clear it only in
        # this fixture to exercise the local H3 compiler contract.
        shot.h3_prompt_skill_output = ""
        h3_prompt = self.service.compile_h3_prompt(project, shot, assets)
        self.assertIn("【跨场景档案｜严格按顺序执行】", h3_prompt)
        self.assertIn("0.00 秒与转场前只使用起始场景“地府”", h3_prompt)
        self.assertIn("只使用目标场景“奶奶家”", h3_prompt)

    def test_seedance_auto_uses_strict_first_frame_only_after_completeness_confirmation(self):
        keyframe_path = self.root / "complete-keyframe.png"
        keyframe_path.write_bytes(PNG)
        keyframe = self.service.register_existing_asset(
            self.project.id, keyframe_path, AssetRole.KEYFRAME, "客厅双人完整首帧"
        )
        project = self.service.require_project(self.project.id)
        shot = self.store.get_shot(self.shot.id)
        shot.keyframe_asset_id = keyframe.id
        shot.scene_profile_ids = [self.home.id]
        shot.scene_profile_id = self.home.id
        shot.use_scene_profile = True
        shot.seedance_reference_mode = SeedanceReferenceMode.AUTO
        assets = self.store.list_assets(self.project.id)

        unknown = self.service.seedance_material_diagnostics(project, shot, assets)
        self.assertEqual(unknown["resolved_mode"], SeedanceReferenceMode.MULTIMODAL_REFERENCE.value)
        self.assertEqual(unknown["first_frame_completeness"], FirstFrameCompleteness.UNKNOWN.value)

        shot.first_frame_completeness = FirstFrameCompleteness.COMPLETE
        refs = self.service.seedance_reference_assets(shot, assets, project)
        complete = self.service.seedance_material_diagnostics(project, shot, assets)
        self.assertEqual(complete["resolved_mode"], SeedanceReferenceMode.STRICT_FIRST_FRAME.value)
        self.assertEqual([asset.id for asset in refs], [keyframe.id])
        self.assertIn("已确认包含", str(complete["reference_mode_reason"]))

        shot.first_frame_completeness = FirstFrameCompleteness.INCOMPLETE
        incomplete = self.service.seedance_material_diagnostics(project, shot, assets)
        self.assertEqual(incomplete["resolved_mode"], SeedanceReferenceMode.MULTIMODAL_REFERENCE.value)
        self.assertIn("信息不完整", str(incomplete["reference_mode_reason"]))

    def test_legacy_scene_profile_id_migrates_to_ordered_scene_list(self):
        shot = Shot.model_validate({
            "project_id": self.project.id,
            "ordinal": 2,
            "scene_profile_id": self.office.id,
        })
        self.assertEqual(shot.scene_profile_ids, [self.office.id])
        self.assertEqual(shot.scene_profile_id, self.office.id)

    def test_keyframe_keeps_explicit_cast_when_story_uses_a_role_alias(self):
        hero = CharacterProfile(name="张小差", description="主角详细外观" * 100)
        elder = CharacterProfile(name="王德发", description="花白短发，藏蓝中山装")
        dog = CharacterProfile(name="老黄狗", description="黄色短毛老犬")
        crowd = CharacterProfile(name="排队鬼魂群像")
        project = self.service.require_project(self.project.id)
        project.characters = [hero, elder, dog, crowd]
        self.store.save_project(project)
        shot = self.store.get_shot(self.shot.id)
        shot.character_ids = [hero.id, elder.id, dog.id]
        shot.scene_description = "张小差站在客厅，王大爷坐在沙发上，老黄狗趴着。"

        resolved = self.service.keyframe_character_ids(project, shot, shot.scene_description)
        prompt = self.service.compile_keyframe_prompt(
            project,
            shot.scene_description,
            resolved,
        )

        self.assertEqual(resolved, [hero.id, elder.id, dog.id])
        self.assertIn("王德发：", prompt)
        self.assertIn("老黄狗：", prompt)

    def test_manual_reference_allowlists_override_keyframe_and_video_defaults(self):
        home_path = self.root / "manual-home.png"
        office_path = self.root / "manual-office.png"
        for path in (home_path, office_path):
            path.write_bytes(PNG)
        home_asset = self.service.register_existing_asset(
            self.project.id, home_path, AssetRole.SCENE, "奶奶家母版"
        )
        office_asset = self.service.register_existing_asset(
            self.project.id, office_path, AssetRole.SCENE, "地府母版"
        )
        project = self.service.require_project(self.project.id)
        project.scene_profiles[0].reference_asset_ids = [home_asset.id]
        project.scene_profiles[0].approved = True
        self.store.save_project(project)
        shot = self.store.get_shot(self.shot.id)
        shot.use_scene_profile = True
        shot.keyframe_reference_asset_ids = [office_asset.id]
        shot.video_reference_asset_ids = [home_asset.id, office_asset.id]
        assets = self.store.list_assets(self.project.id)

        keyframe_refs = self.service.keyframe_reference_assets(project, shot, assets, [])
        video_refs = self.service.seedance_reference_assets(shot, assets, project)
        keyframe_diagnostics = self.service.keyframe_material_diagnostics(project, shot, assets)
        video_diagnostics = self.service.seedance_material_diagnostics(project, shot, assets)

        self.assertEqual([item.id for item in keyframe_refs], [office_asset.id])
        self.assertEqual([item.id for item in video_refs], [home_asset.id, office_asset.id])
        self.assertEqual(keyframe_diagnostics["selection_mode"], "manual")
        self.assertEqual(video_diagnostics["selection_mode"], "manual")
        self.assertEqual(
            [item["id"] for item in keyframe_diagnostics["materials"]],
            [office_asset.id],
        )

    def test_fixed_prop_reference_is_available_to_still_and_video_compilers(self):
        prop_path = self.root / "battery-sheet.png"
        prop_path.write_bytes(PNG)
        prop_asset = self.service.register_existing_asset(
            self.project.id,
            prop_path,
            AssetRole.PROP,
            "电动车铅酸电池组",
            description="四块绿色 12V20Ah 电池组成 2×2 重型电池组，约两只鞋盒并排，橙色连接线",
        )
        project = self.service.require_project(self.project.id)
        shot = self.store.get_shot(self.shot.id)
        shot.scene_description = "客厅地面摆着正在充电的电动车电池组"
        shot.reference_asset_ids = [prop_asset.id]
        assets = self.store.list_assets(self.project.id)

        keyframe_refs = self.service.keyframe_reference_assets(project, shot, assets, [])
        video_refs = self.service.seedance_reference_assets(shot, assets, project)
        anchored_prompt = self.service.apply_fixed_prop_anchor("【首帧画面】客厅地面", keyframe_refs)
        seedance_prompt = self.service.compile_seedance_prompt(project, shot, assets)
        diagnostics = self.service.keyframe_material_diagnostics(project, shot, assets)

        self.assertIn(prop_asset.id, [asset.id for asset in keyframe_refs])
        self.assertIn(prop_asset.id, [asset.id for asset in video_refs])
        self.assertIn("【固定物锚点】", anchored_prompt)
        self.assertIn("两只鞋盒并排", anchored_prompt)
        self.assertIn("固定物品定义", seedance_prompt)
        self.assertIn("禁止缩成掌心玩具", seedance_prompt)
        self.assertIn(prop_asset.id, [item["id"] for item in diagnostics["available_materials"]])

        shot.keyframe_reference_asset_ids = []
        shot.video_reference_asset_ids = []
        self.assertTrue(any("物品参考图未被勾选" in warning for warning in self.service.keyframe_material_diagnostics(project, shot, assets)["warnings"]))
        self.assertTrue(any("物品参考图未被选入视频参考" in warning for warning in self.service.seedance_material_diagnostics(project, shot, assets)["warnings"]))

    async def test_exact_prompt_and_parallel_scenes_preserve_each_other(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        class Generator:
            async def generate_image(_, scene, output_dir, *args, **kwargs):
                calls.append((scene.visual_prompt, kwargs))
                if scene.visual_prompt.startswith("CUSTOM HOME\n"):
                    started.set()
                    await release.wait()
                output = Path(output_dir) / "result.png"
                output.write_bytes(PNG)
                return str(output)

        with patch("src.video_workflow.services.projects.create_image_generator", return_value=Generator()):
            task = asyncio.create_task(self.service.generate_scene_reference(self.project.id, self.home.id, prompt="CUSTOM HOME", expected_version=1))
            await asyncio.wait_for(started.wait(), 2)
            with self.assertRaises(SceneReferenceConflictError):
                await self.service.generate_scene_reference(self.project.id, self.home.id, prompt="DUPLICATE")
            with self.assertRaises(SceneReferenceConflictError):
                self.service.update_scene_profile(self.project.id, self.home.id, {"name": "busy edit"}, 2)
            await self.service.generate_scene_reference(self.project.id, self.office.id, prompt="CUSTOM OFFICE")
            release.set()
            await task
        self.assertTrue(calls[0][0].startswith("CUSTOM HOME\n"))
        self.assertTrue(calls[1][0].startswith("CUSTOM OFFICE\n"))
        self.assertTrue(all("绝对不出现任何人物" in prompt for prompt, _ in calls))
        for _, kwargs in calls:
            self.assertEqual(kwargs["image_style"], "")
            self.assertEqual(kwargs["character_description"], "")
        self.assertEqual(self.latest().reference_status, "completed")
        self.assertEqual(self.latest(self.office.id).reference_status, "completed")
        self.assertEqual(len(self.store.list_assets(self.project.id)), 2)
        self.assertEqual(self.store.get_shot(self.shot.id).h3_prompt_skill_output, "PAID H3")

    async def test_failed_download_retains_old_asset_and_can_resume_without_generation(self):
        path = self.root / "old.png"
        path.write_bytes(PNG)
        asset = self.service.register_existing_asset(self.project.id, path, AssetRole.SCENE, "旧母版")
        self.project.scene_profiles[0].reference_asset_ids = [asset.id]
        self.store.save_project(self.project)

        class Generator:
            async def generate_image(_, scene, output_dir, *args, **kwargs):
                GrsaiImageGenerator._save_result_checkpoint(output_dir, {"result_url": "https://cdn.example/image.png"})
                raise ImageDownloadError("服务商已出图，下载失败")

        with patch("src.video_workflow.services.projects.create_image_generator", return_value=Generator()):
            with self.assertRaises(ImageDownloadError):
                await self.service.generate_scene_reference(self.project.id, self.home.id, prompt="PAID IMAGE PROMPT")
        failed = self.latest()
        self.assertEqual(failed.reference_status, "download_failed")
        self.assertEqual(failed.reference_asset_ids, [asset.id])
        self.assertEqual(failed.reference_prompt, "PAID IMAGE PROMPT")
        self.assertIn("绝对不出现任何人物", failed.reference_generation_prompt)

        async def resume(output_dir):
            result = Path(output_dir) / "result.png"
            result.write_bytes(PNG)
            return str(result)

        with patch("src.video_workflow.services.projects.create_image_generator", side_effect=AssertionError("paid submission forbidden")), patch.object(GrsaiImageGenerator, "resume_image", side_effect=resume):
            await self.service.retry_scene_reference_download(self.project.id, self.home.id)
        self.assertEqual(self.latest().reference_status, "completed")
        self.assertEqual(len(self.latest().reference_asset_ids), 1)
        self.assertNotEqual(self.latest().reference_asset_ids[0], asset.id)
        self.assertIsNotNone(self.store.get_asset(asset.id))
        self.assertTrue(self.latest().reference_generation_prompt.startswith("PAID IMAGE PROMPT\n"))

    def test_restart_and_download_stage_are_visible(self):
        profile = self.service._set_scene_reference_state(self.project.id, self.home.id,
            reference_run_id="a" * 32, reference_status="generating")
        GrsaiImageGenerator._save_result_checkpoint(str(self.service._scene_output_dir(self.project.id, profile)), {"result_url": "https://cdn.example/a.png"})
        self.service._active_scene_references.add(self.home.id)
        project = self.service.reconcile_scene_reference_status(self.service.require_project(self.project.id))
        self.assertEqual(project.scene_profiles[0].reference_status, "downloading")
        self.service._active_scene_references.clear()
        project = self.service.reconcile_scene_reference_status(self.service.require_project(self.project.id))
        self.assertEqual(project.scene_profiles[0].reference_status, "download_failed")


class GrsaiRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_failure_falls_back_to_verified_direct_and_follows_redirects(self):
        clients = []
        actual_client = httpx.AsyncClient

        def client(**kwargs):
            clients.append(kwargs)
            def handle(request):
                self.assertNotIn("authorization", request.headers)
                if kwargs["trust_env"]:
                    raise httpx.ConnectError("proxy TLS error", request=request)
                if request.url.path == "/image":
                    return httpx.Response(302, headers={"location": "/final.png"})
                return httpx.Response(200, content=PNG)
            return actual_client(transport=httpx.MockTransport(handle), **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch("src.video_workflow.generators.image.httpx.AsyncClient", side_effect=client), patch("src.video_workflow.generators.image.asyncio.sleep", new=AsyncMock()):
            output = await download_generated_image("https://cdn.example/image", folder)
            self.assertEqual(Path(output).read_bytes(), PNG)
            self.assertFalse(list(Path(folder).glob("*.part")))
        self.assertEqual([item["trust_env"] for item in clients], [True, False])
        self.assertTrue(all(item["follow_redirects"] for item in clients))
        self.assertTrue(all(item.get("verify", True) for item in clients))

    async def test_checkpoint_saved_before_download_and_resume_only_gets_image(self):
        requests = []
        fail_download = True
        actual_client = httpx.AsyncClient

        def handle(request):
            requests.append((request.method, request.url.path))
            if request.method == "POST":
                return httpx.Response(200, json={"status": "succeeded", "id": "paid-task", "url": "https://cdn.example/image.png"})
            if fail_download:
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(200, content=PNG)

        def client(**kwargs):
            return actual_client(transport=httpx.MockTransport(handle), **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch("src.video_workflow.generators.image.settings.GRSAI_API_KEY", "fixture-key"), patch("src.video_workflow.generators.image.httpx.AsyncClient", side_effect=client), patch("src.video_workflow.generators.image.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ImageDownloadError):
                await GrsaiImageGenerator("gpt-image-2").generate_image(Scene(id=1, duration=5, narrative="room", visual_prompt="EXACT", motion_prompt=""), folder, character_description="", image_style="")
            saved = json.loads((Path(folder) / "grsai_result.json").read_text())
            self.assertEqual(saved["result_url"], "https://cdn.example/image.png")
            fail_download = False
            output = await GrsaiImageGenerator.resume_image(folder)
            self.assertEqual(Path(output).read_bytes(), PNG)
        self.assertEqual(sum(method == "POST" for method, _ in requests), 1)

    async def test_html_error_page_is_not_saved_as_image(self):
        actual_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as folder, patch("src.video_workflow.generators.image.httpx.AsyncClient", side_effect=lambda **kwargs: actual_client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="<html>error</html>")), **kwargs)), patch("src.video_workflow.generators.image.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ImageDownloadError):
                await download_generated_image("https://cdn.example/a", folder)
            self.assertEqual(list(Path(folder).iterdir()), [])
