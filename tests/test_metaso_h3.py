from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetRole, ProjectBrief, Shot, ShotContinuityMode
from src.video_workflow.integrations.metaso_h3 import (
    METASO_H3_MODEL_ID,
    MetaSoH3Client,
    metaso_duration,
    metaso_request_ratio,
    metaso_resolution,
    metaso_ratio,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.storage import ProjectStore


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.content = b"video"

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("metaso error", request=None, response=self)  # type: ignore[arg-type]


class FakeAsyncClient:
    requests: list[dict] = []
    poll_responses: list[dict] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> FakeResponse:
        FakeAsyncClient.requests.append({"method": "POST", "url": url, "headers": headers or {}, "json": json or {}})
        return FakeResponse({"task_id": "task-1"})

    async def get(self, url: str, headers: dict | None = None) -> FakeResponse:
        FakeAsyncClient.requests.append({"method": "GET", "url": url, "headers": headers or {}})
        return FakeResponse(FakeAsyncClient.poll_responses.pop(0))


class MetaSoH3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = (
            settings.OUTPUT_DIR,
            settings.PROJECTS_DIR,
            settings.METASO_H3_API_KEY,
            settings.METASO_H3_BASE_URL,
            settings.METASO_H3_POLL_INTERVAL_SECONDS,
            settings.METASO_H3_RESOLUTION,
            settings.METASO_H3_RATIO,
            settings.METASO_H3_CONTEXT_IR_ENABLED,
            settings.H3_PROVIDER,
        )
        settings.OUTPUT_DIR = self.root / "legacy"
        settings.PROJECTS_DIR = self.root / "projects"
        settings.METASO_H3_API_KEY = "mk-test"
        settings.METASO_H3_BASE_URL = "https://metaso.example/api/minimax"
        settings.METASO_H3_POLL_INTERVAL_SECONDS = 0.01
        settings.METASO_H3_RESOLUTION = "768P"
        settings.METASO_H3_RATIO = "adaptive"
        settings.METASO_H3_CONTEXT_IR_ENABLED = False
        FakeAsyncClient.requests = []
        FakeAsyncClient.poll_responses = []

    def tearDown(self) -> None:
        (
            settings.OUTPUT_DIR,
            settings.PROJECTS_DIR,
            settings.METASO_H3_API_KEY,
            settings.METASO_H3_BASE_URL,
            settings.METASO_H3_POLL_INTERVAL_SECONDS,
            settings.METASO_H3_RESOLUTION,
            settings.METASO_H3_RATIO,
            settings.METASO_H3_CONTEXT_IR_ENABLED,
            settings.H3_PROVIDER,
        ) = self.previous
        self.temp.cleanup()

    def test_duration_is_clamped_to_h3_range(self) -> None:
        self.assertEqual(metaso_duration(2.0), 4)
        self.assertEqual(metaso_duration(7.6), 8)
        self.assertEqual(metaso_duration(20.0), 15)

    def test_project_ratio_overrides_global_adaptive_setting(self) -> None:
        self.assertEqual(metaso_ratio("9:16", "adaptive"), "9:16")
        self.assertEqual(metaso_ratio("16:9", "9:16"), "16:9")
        self.assertEqual(metaso_ratio("custom", "3:4"), "3:4")
        self.assertEqual(metaso_ratio("custom", "invalid"), "adaptive")

    def test_per_shot_api_settings_override_model_defaults(self) -> None:
        self.assertEqual(metaso_request_ratio("9:16", "3:4", "adaptive"), "3:4")
        self.assertEqual(metaso_request_ratio("9:16", "project", "adaptive"), "9:16")
        self.assertEqual(metaso_resolution("2K", "768P"), "2K")
        self.assertEqual(metaso_resolution("default", "2K"), "2K")

    def test_hosted_h3_keeps_fast_authored_commercial_read_in_one_call(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(ProjectBrief(title="快口播", story="商业口播测试"))
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            duration_seconds=11,
            dialogue="这是一段需要快速清晰表达的商业口播内容，主持人情绪急切而有感染力，整段不应因为默认慢速预算被拆开。",
            dialogue_rate_percent=40,
            h3_prompt_skill_output="VERTICAL 9:16 authored commercial prompt",
        )
        plan = service.h3_segment_plan(project, shot, [], max_segment_seconds=15)
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0]["prompt"].startswith(shot.h3_prompt_skill_output))
        self.assertIn("把全部49个有效中文口播字符完整放入11秒内", plan[0]["prompt"])
        self.assertIn("每个标签只说一次，不漏字、不加字、不重复", plan[0]["prompt"])

    def test_submit_uses_v2_multimodal_schema_and_context_ir_is_opt_in(self) -> None:
        with patch("src.video_workflow.integrations.metaso_h3.httpx.AsyncClient", FakeAsyncClient):
            task_id = asyncio.run(
                MetaSoH3Client().submit(
                    prompt="人物沿动作参考完成操作",
                    image_urls=["https://cdn.example/person.png"],
                    video_urls=["https://cdn.example/motion.mp4"],
                    audio_urls=["https://cdn.example/voice.wav"],
                    resolution="768P",
                    duration=5,
                    ratio="adaptive",
                )
            )
        self.assertEqual(task_id, "task-1")
        request = FakeAsyncClient.requests[0]
        self.assertEqual(request["url"], "https://metaso.example/api/minimax/v2/video_generation")
        self.assertEqual(request["headers"]["Authorization"], "Bearer mk-test")
        payload = request["json"]
        self.assertEqual(payload["model"], METASO_H3_MODEL_ID)
        self.assertFalse(payload["context_ir_enabled"])
        self.assertEqual(payload["resolution"], "768P")
        self.assertEqual([item["role"] for item in payload["content"][1:]], ["reference_image", "reference_video", "reference_audio"])

    def test_first_frame_mode_uses_first_and_last_frame_roles(self) -> None:
        with patch("src.video_workflow.integrations.metaso_h3.httpx.AsyncClient", FakeAsyncClient):
            asyncio.run(
                MetaSoH3Client().submit(
                    prompt="首尾帧测试",
                    first_frame_url="data:image/png;base64,AAAA",
                    last_frame_url="data:image/png;base64,BBBB",
                    resolution="2K",
                    duration=6,
                    ratio="16:9",
                )
            )
        content = FakeAsyncClient.requests[0]["json"]["content"]
        self.assertEqual([item.get("role") for item in content[1:]], ["first_frame", "last_frame"])
        with self.assertRaisesRegex(ValueError, "分开提交"):
            asyncio.run(
                MetaSoH3Client().submit(
                    prompt="错误混用",
                    first_frame_url="data:image/png;base64,AAAA",
                    image_urls=["https://cdn.example/ref.png"],
                    resolution="2K",
                    duration=6,
                    ratio="adaptive",
                )
            )

        with patch("src.video_workflow.integrations.metaso_h3.httpx.AsyncClient", FakeAsyncClient):
            asyncio.run(
                MetaSoH3Client().submit(
                    prompt="只给尾帧",
                    last_frame_url="data:image/png;base64,BBBB",
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                )
            )
            asyncio.run(
                MetaSoH3Client().submit(
                    prompt="只给声音参考",
                    audio_urls=["https://cdn.example/voice.wav"],
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                )
            )
        self.assertEqual(FakeAsyncClient.requests[-2]["json"]["content"][1]["role"], "last_frame")
        self.assertEqual(FakeAsyncClient.requests[-1]["json"]["content"][1]["role"], "reference_audio")

    def test_poll_extracts_official_task_content_url(self) -> None:
        FakeAsyncClient.poll_responses = [
            {"task": {"status": "processing"}},
            {"task": {"status": "success", "content": {"url": "https://cdn.example/result.mp4"}}},
        ]
        progress_calls: list[str] = []

        async def progress(value: float, message: str) -> None:
            progress_calls.append(message)

        with patch("src.video_workflow.integrations.metaso_h3.httpx.AsyncClient", FakeAsyncClient):
            output_url = asyncio.run(MetaSoH3Client().wait_for_result("task-1", progress))
        self.assertEqual(output_url, "https://cdn.example/result.mp4")
        self.assertTrue(all("/v2/query/video_generation/task-1" in item["url"] for item in FakeAsyncClient.requests))
        self.assertTrue(progress_calls)

    def test_enqueue_freezes_selected_metaso_provider(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(ProjectBrief(title="MetaSo", story="测试 MetaSo H3 入队"))
        image = self.root / "first.png"
        image.write_bytes(b"image")
        asset = service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="测试",
            duration_seconds=6,
            keyframe_asset_id=asset.id,
            image_path=str(image),
        )
        store.save_shot(shot)
        settings.H3_PROVIDER = "metaso_h3"
        jobs = RenderQueue(store, service).enqueue(project.id, [shot.id])
        self.assertEqual(jobs[0].provider, "metaso_h3")

    def test_queue_maps_i2v_keyframe_to_first_frame_and_saves_resume_snapshot(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(
            ProjectBrief(title="MetaSo I2V", story="首帧映射测试", aspect_ratio="9:16", width=768, height=1344)
        )
        image = self.root / "first.png"
        image.write_bytes(b"image")
        asset = service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        last_image = self.root / "last.png"
        last_image.write_bytes(b"last-image")
        last_asset = service.register_existing_asset(project.id, last_image, AssetRole.LAST_FRAME, "尾帧")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="人物转身",
            duration_seconds=6,
            keyframe_asset_id=asset.id,
            last_frame_asset_id=last_asset.id,
            image_path=str(image),
            video_prompt="人物自然转身，镜头稳定",
            h3_resolution="2K",
            h3_ratio="3:4",
            h3_context_ir_enabled=True,
        )
        store.save_shot(shot)
        settings.H3_PROVIDER = "metaso_h3"
        job = RenderQueue(store, service).enqueue(project.id, [shot.id])[0]
        settings.METASO_H3_RESOLUTION = "768P"
        settings.METASO_H3_RATIO = "16:9"
        settings.METASO_H3_CONTEXT_IR_ENABLED = False
        captured: dict[str, object] = {}

        class FakeMetaSoClient:
            async def submit(self, **kwargs: object) -> str:
                captured.update(kwargs)
                return "task-queue"

            async def wait_for_result(self, task_id: str, progress: object) -> str:
                raise RuntimeError("stop-after-submit")

        queue = RenderQueue(store, service)
        queue.metaso_client = FakeMetaSoClient()  # type: ignore[assignment]
        with self.assertRaisesRegex(RuntimeError, "stop-after-submit"):
            asyncio.run(queue._process_metaso_h3(store.get_job(job.id)))

        self.assertTrue(str(captured["first_frame_url"]).startswith("data:image/"))
        self.assertTrue(str(captured["last_frame_url"]).startswith("data:image/"))
        self.assertEqual(captured["image_urls"], [])
        self.assertEqual(captured["video_urls"], [])
        self.assertEqual(captured["resolution"], "2K")
        self.assertEqual(captured["ratio"], "3:4")
        self.assertTrue(captured["context_ir_enabled"])
        persisted = store.get_job(job.id)
        self.assertEqual(persisted.input_snapshot["metaso"]["model"], "MiniMax-H3")
        self.assertTrue(persisted.input_snapshot["metaso"]["frame_mode"])
        self.assertEqual(persisted.input_snapshot["metaso"]["input_mode"], "first_last_frame")
        self.assertEqual(persisted.input_snapshot["metaso"]["resolution"], "2K")
        self.assertEqual(persisted.input_snapshot["metaso"]["ratio"], "3:4")
        self.assertTrue(persisted.input_snapshot["metaso"]["context_ir_enabled"])
        self.assertEqual(persisted.input_snapshot["metaso"]["first_frame_asset_id"], asset.id)
        self.assertEqual(persisted.input_snapshot["metaso"]["last_frame_asset_id"], last_asset.id)
        self.assertEqual(persisted.input_snapshot["metaso_resume"]["parts"][0]["task_id"], "task-queue")

    def test_continuous_h3_uses_previous_shot_tail_as_first_frame(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(ProjectBrief(title="H3 连续续接", story="连续口播"))
        tail_path = self.root / "previous-tail.png"
        stale_keyframe_path = self.root / "stale-keyframe.png"
        tail_path.write_bytes(b"tail")
        stale_keyframe_path.write_bytes(b"stale")
        tail = service.register_existing_asset(project.id, tail_path, AssetRole.LAST_FRAME, "上一镜 H3 尾帧")
        stale_keyframe = service.register_existing_asset(project.id, stale_keyframe_path, AssetRole.KEYFRAME, "本镜旧首帧")
        first = store.save_shot(
            Shot(project_id=project.id, ordinal=1, title="第一镜", last_frame_asset_id=tail.id, video_status="completed")
        )
        second = store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=2,
                title="第二镜",
                duration_seconds=6,
                keyframe_asset_id=stale_keyframe.id,
                continuity_mode=ShotContinuityMode.CONTINUOUS,
            )
        )
        settings.H3_PROVIDER = "metaso_h3"
        job = RenderQueue(store, service).enqueue(project.id, [second.id])[0]
        self.assertEqual(job.input_snapshot["h3_parameters"]["continuity_mode"], "continuous")
        self.assertEqual(job.input_snapshot["h3_parameters"]["continuity_source_shot_id"], first.id)

        captured: dict[str, object] = {}

        class FakeMetaSoClient:
            async def submit(self, **kwargs: object) -> str:
                captured.update(kwargs)
                return "task-continuity"

            async def wait_for_result(self, task_id: str, progress: object) -> str:
                raise RuntimeError("stop-after-submit")

        queue = RenderQueue(store, service)
        queue.metaso_client = FakeMetaSoClient()  # type: ignore[assignment]
        with self.assertRaisesRegex(RuntimeError, "stop-after-submit"):
            asyncio.run(queue._process_metaso_h3(store.get_job(job.id)))

        self.assertTrue(str(captured["first_frame_url"]).startswith("data:image/"))
        self.assertIsNone(captured["last_frame_url"])
        self.assertEqual(captured["image_urls"], [])
        persisted = store.get_job(job.id)
        self.assertEqual(persisted.input_snapshot["metaso"]["first_frame_asset_id"], tail.id)
        self.assertEqual(persisted.input_snapshot["metaso"]["continuity_input_asset_id"], tail.id)


if __name__ == "__main__":
    unittest.main()
