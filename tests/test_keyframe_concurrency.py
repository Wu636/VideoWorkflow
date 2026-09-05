from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetRole, ProjectBrief, Shot
from src.video_workflow.services.projects import H3_DIRECTOR_VERSION, KeyframeBusyError, ProjectService
from src.video_workflow.storage import ProjectStore


H3_OUTPUT = (
    "For the target video, <Picture 1> is fully referenced.\n"
    "integrated_multimodal_description: A person opens the door.\n"
    "overall_soundscape: Door hinge and room tone.\n"
    "non_diegetic_music: None."
)


class KeyframeConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.projects_patch = patch.object(settings, "PROJECTS_DIR", self.root / "projects")
        self.projects_patch.start()
        self.store = ProjectStore(self.root / "state.sqlite3")
        self.service = ProjectService(self.store)
        self.project = self.service.create_project(ProjectBrief(title="并行生成", story="人物开门"))
        self.shot = self.store.save_shot(Shot(
            project_id=self.project.id, ordinal=1, narrative="人物开门", keyframe_prompt="门前的静态画面",
            seedance_prompt="KEEP_SEEDANCE",
        ))
        self.tasks: list[asyncio.Task] = []

    async def asyncTearDown(self) -> None:
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def tearDown(self) -> None:
        self.projects_patch.stop()
        self.temp.cleanup()

    def task(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def wait_started(self, event: asyncio.Event) -> None:
        await asyncio.wait_for(event.wait(), timeout=5)

    def assert_h3_saved(self, shot_id: str) -> Shot:
        # Reopen the database to verify persisted data, not response/local state.
        saved = ProjectStore(self.root / "state.sqlite3").get_shot(shot_id)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved.h3_prompt_skill_output, H3_OUTPUT)
        self.assertEqual(saved.video_prompt, H3_OUTPUT)
        self.assertEqual(saved.video_prompt_source, H3_OUTPUT)
        self.assertEqual(saved.h3_director_version, H3_DIRECTOR_VERSION)
        self.assertEqual(saved.h3_prompt_source_revision, saved.content_revision)
        return saved

    async def run_completion_order(self, first: str, fail_image: bool = False) -> None:
        image_started, llm_started = asyncio.Event(), asyncio.Event()
        release_image, release_llm = asyncio.Event(), asyncio.Event()
        previous_path = self.root / "previous.png"
        previous_path.write_bytes(b"old-image")
        previous_asset = self.service.register_existing_asset(self.project.id, previous_path, AssetRole.KEYFRAME, "旧首帧")
        self.shot.keyframe_asset_id = previous_asset.id
        self.shot.image_path = str(previous_path)
        self.store.save_shot(self.shot)

        class ImageGenerator:
            async def generate_image(self, scene, output_dir, *args, **kwargs):
                image_started.set()
                await release_image.wait()
                if fail_image:
                    raise RuntimeError("image provider failed")
                output = Path(output_dir) / "image.png"
                output.write_bytes(b"new-image")
                return str(output)

        class LLM:
            async def generate_json(self, *args, **kwargs):
                llm_started.set()
                await release_llm.wait()
                return {"video_prompt": H3_OUTPUT}

        with (
            patch("src.video_workflow.services.projects.create_image_generator", return_value=ImageGenerator()),
            patch("src.video_workflow.services.projects.create_llm_generator", return_value=LLM()),
        ):
            image_task = self.task(self.service.generate_keyframes(self.project.id, [self.shot.id], user_suggestions="增加逆光"))
            await self.wait_started(image_started)
            h3_task = self.task(self.service.generate_h3_prompts(self.project.id, [self.shot.id], "h3-prompt-writing"))
            await self.wait_started(llm_started)
            # A project edit during generation must survive too.
            latest_project = self.service.require_project(self.project.id)
            latest_project.brief.title = "并发保存的新项目标题"
            self.store.save_project(latest_project)
            if first == "image":
                release_image.set()
                await image_task
                release_llm.set()
                await h3_task
            else:
                release_llm.set()
                await h3_task
                self.assertEqual(self.assert_h3_saved(self.shot.id).image_status, "processing")
                release_image.set()
                if fail_image:
                    with self.assertRaisesRegex(RuntimeError, "image provider failed"):
                        await image_task
                else:
                    await image_task

        saved = self.assert_h3_saved(self.shot.id)
        self.assertEqual(saved.seedance_prompt, "KEEP_SEEDANCE")
        self.assertEqual(self.service.require_project(self.project.id).brief.title, "并发保存的新项目标题")
        self.service.ensure_keyframe_idle(self.shot.id)
        if fail_image:
            self.assertEqual(saved.image_status, "failed")
            self.assertEqual(saved.keyframe_asset_id, previous_asset.id)
            self.assertEqual(saved.image_path, str(previous_path))
            self.assertEqual(saved.keyframe_revision_suggestion_draft, "增加逆光")
        else:
            self.assertEqual(saved.image_status, "completed")
            self.assertNotEqual(saved.keyframe_asset_id, previous_asset.id)
            self.assertTrue(Path(saved.image_path).is_file())
            self.assertEqual(saved.keyframe_revision_suggestion_draft, "")
            self.assertEqual(saved.keyframe_revision_last_suggestion, "增加逆光")

    async def test_h3_finishes_before_image(self) -> None:
        await self.run_completion_order("h3")

    async def test_image_finishes_before_h3(self) -> None:
        await self.run_completion_order("image")

    async def test_image_failure_preserves_h3_previous_image_and_retry_draft(self) -> None:
        await self.run_completion_order("h3", fail_image=True)

    async def test_queued_image_preserves_h3_generated_while_waiting(self) -> None:
        second = self.store.save_shot(Shot(project_id=self.project.id, ordinal=2, narrative="人物关门"))
        first_started, release_first = asyncio.Event(), asyncio.Event()
        calls: list[int] = []

        class ImageGenerator:
            async def generate_image(self, scene, output_dir, *args, **kwargs):
                calls.append(scene.id)
                if scene.id == 1:
                    first_started.set()
                    await release_first.wait()
                output = Path(output_dir) / "image.png"
                output.write_bytes(b"image")
                return str(output)

        class LLM:
            async def generate_json(self, *args, **kwargs):
                return {"video_prompt": H3_OUTPUT}

        with (
            patch.object(settings, "WORKFLOW_CONCURRENCY", 1),
            patch("src.video_workflow.services.projects.create_image_generator", return_value=ImageGenerator()),
            patch("src.video_workflow.services.projects.create_llm_generator", return_value=LLM()),
        ):
            images = self.task(self.service.generate_keyframes(self.project.id, [self.shot.id, second.id]))
            await self.wait_started(first_started)
            await self.service.generate_h3_prompts(self.project.id, [second.id], "h3-prompt-writing")
            self.assertEqual(calls, [1])
            self.assertEqual(self.assert_h3_saved(second.id).image_status, "processing")
            with self.assertRaises(KeyframeBusyError):
                await self.service.generate_keyframes(self.project.id, [second.id])
            release_first.set()
            await images

        self.assertEqual(calls, [1, 2])
        self.assertEqual(self.assert_h3_saved(second.id).image_status, "completed")

    async def test_keyframe_prompt_patch_before_and_after_h3_keeps_both_results(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()

        class LLM:
            async def generate_json(self, *args, **kwargs):
                started.set()
                await release.wait()
                return {"video_prompt": H3_OUTPUT}

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=LLM()):
            h3 = self.task(self.service.generate_h3_prompts(self.project.id, [self.shot.id], "h3-prompt-writing"))
            await self.wait_started(started)
            self.service.update_keyframe_prompt(self.project.id, self.shot.id, "新版首帧", "增加逆光", "iterate")
            release.set()
            await h3
        saved = self.assert_h3_saved(self.shot.id)
        self.assertEqual(saved.keyframe_prompt, "新版首帧")
        self.service.update_keyframe_prompt(self.project.id, self.shot.id, "再次修改首帧")
        saved = self.assert_h3_saved(self.shot.id)
        self.assertEqual(saved.keyframe_prompt, "再次修改首帧")
        self.assertEqual(saved.keyframe_revision_suggestion_draft, "增加逆光")
        self.assertEqual(saved.keyframe_revision_mode, "iterate")
        self.assertEqual(saved.seedance_prompt, "KEEP_SEEDANCE")

    async def test_duplicate_image_and_same_shot_edits_block_but_other_shots_run(self) -> None:
        second = self.store.save_shot(Shot(project_id=self.project.id, ordinal=2))
        started, release = asyncio.Event(), asyncio.Event()
        calls: list[int] = []

        class ImageGenerator:
            async def generate_image(self, scene, output_dir, *args, **kwargs):
                calls.append(scene.id)
                if scene.id == 1:
                    started.set()
                    await release.wait()
                output = Path(output_dir) / "image.png"
                output.write_bytes(b"image")
                return str(output)

        with patch("src.video_workflow.services.projects.create_image_generator", return_value=ImageGenerator()):
            images = self.task(self.service.generate_keyframes(self.project.id, [self.shot.id]))
            await self.wait_started(started)
            with self.assertRaises(KeyframeBusyError):
                await self.service.generate_keyframes(self.project.id, [self.shot.id])
            with self.assertRaises(KeyframeBusyError):
                self.service.update_keyframe_prompt(self.project.id, self.shot.id, "另一个首帧请求")
            await self.service.generate_keyframes(self.project.id, [second.id])
            self.assertEqual(calls, [1, 2])
            release.set()
            await images
            # The guard is released after completion, allowing a deliberate redo.
            await self.service.generate_keyframes(self.project.id, [self.shot.id])
            self.assertEqual(calls, [1, 2, 1])

    async def test_cancellation_releases_keyframe_guard_and_keeps_prompt(self) -> None:
        started = asyncio.Event()

        class ImageGenerator:
            async def generate_image(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()

        with patch("src.video_workflow.services.projects.create_image_generator", return_value=ImageGenerator()):
            images = self.task(self.service.generate_keyframes(self.project.id, [self.shot.id]))
            await self.wait_started(started)
            latest = self.service.require_shot(self.shot.id)
            latest.h3_prompt_skill_output = H3_OUTPUT
            self.store.save_shot(latest)
            images.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await images
        saved = self.service.require_shot(self.shot.id)
        self.assertEqual(saved.image_status, "failed")
        self.assertEqual(saved.h3_prompt_skill_output, H3_OUTPUT)
        self.service.ensure_keyframe_idle(self.shot.id)

    async def test_prompt_preparation_failure_releases_keyframe_guard(self) -> None:
        with (
            patch("src.video_workflow.services.projects.create_image_generator"),
            patch.object(self.service, "compile_keyframe_prompt", side_effect=ValueError("invalid prompt")),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid prompt"):
                await self.service.generate_keyframes(self.project.id, [self.shot.id])
        self.assertEqual(self.service.require_shot(self.shot.id).image_status, "failed")
        self.service.ensure_keyframe_idle(self.shot.id)
