from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetRole, ProjectBrief, Shot
from src.video_workflow.integrations.metaso_h3 import METASO_H3_MODEL_ID, MetaSoH3Client, metaso_duration
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
        project = service.create_project(ProjectBrief(title="MetaSo I2V", story="首帧映射测试"))
        image = self.root / "first.png"
        image.write_bytes(b"image")
        asset = service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            narrative="人物转身",
            duration_seconds=6,
            keyframe_asset_id=asset.id,
            image_path=str(image),
            video_prompt="人物自然转身，镜头稳定",
        )
        store.save_shot(shot)
        settings.H3_PROVIDER = "metaso_h3"
        job = RenderQueue(store, service).enqueue(project.id, [shot.id])[0]
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
        self.assertEqual(captured["image_urls"], [])
        self.assertEqual(captured["video_urls"], [])
        self.assertEqual(captured["resolution"], "768P")
        persisted = store.get_job(job.id)
        self.assertEqual(persisted.input_snapshot["metaso"]["model"], "MiniMax-H3")
        self.assertEqual(persisted.input_snapshot["metaso_resume"]["parts"][0]["task_id"], "task-queue")


if __name__ == "__main__":
    unittest.main()
