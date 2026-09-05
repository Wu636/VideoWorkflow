from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from src.video_workflow.domain import AssetRole, ProjectBrief, SceneProfile, Shot, StyleProfile
from src.video_workflow.generators.image import GrsaiImageGenerator, ImageDownloadError, download_generated_image
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

    async def test_exact_prompt_and_parallel_scenes_preserve_each_other(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        class Generator:
            async def generate_image(_, scene, output_dir, *args, **kwargs):
                calls.append((scene.visual_prompt, kwargs))
                if scene.visual_prompt == "CUSTOM HOME":
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
        self.assertEqual([item[0] for item in calls], ["CUSTOM HOME", "CUSTOM OFFICE"])
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

        async def resume(output_dir):
            result = Path(output_dir) / "result.png"
            result.write_bytes(PNG)
            return str(result)

        with patch("src.video_workflow.services.projects.create_image_generator", side_effect=AssertionError("paid submission forbidden")), patch.object(GrsaiImageGenerator, "resume_image", side_effect=resume):
            await self.service.retry_scene_reference_download(self.project.id, self.home.id)
        self.assertEqual(self.latest().reference_status, "completed")
        self.assertEqual(len(self.latest().reference_asset_ids), 2)
        self.assertEqual(self.latest().reference_generation_prompt, "PAID IMAGE PROMPT")

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
