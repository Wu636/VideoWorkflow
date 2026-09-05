from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetRole, ProjectBrief, Shot
from src.video_workflow.integrations.atlas_h3 import (
    ATLAS_MODEL_ID,
    AtlasH3Client,
    atlas_duration,
    atlas_ratio,
    atlas_resolution,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.storage import ProjectStore


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("atlas error", request=None, response=self)  # type: ignore[arg-type]


class FakeAsyncClient:
    requests: list[dict] = []
    poll_responses: list[dict] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(
        self,
        url: str,
        headers: dict | None = None,
        json: dict | None = None,
        files: dict | None = None,
    ) -> FakeResponse:
        request = {"method": "POST", "url": url, "headers": headers or {}, "json": json or {}}
        if files:
            filename, _handle, mime_type = files["file"]
            request["file"] = {"filename": filename, "mime_type": mime_type}
        FakeAsyncClient.requests.append(request)
        if url.endswith("/uploadMedia"):
            return FakeResponse(
                {
                    "code": 200,
                    "message": "success",
                    "data": {
                        "type": "image",
                        "download_url": "https://media.atlas.invalid/uploaded-reference.png",
                        "filename": "uploaded-reference.png",
                        "size": 2045045,
                    },
                }
            )
        return FakeResponse({"data": {"id": "pred-1"}})

    async def get(self, url: str, headers: dict | None = None) -> FakeResponse:
        FakeAsyncClient.requests.append({"method": "GET", "url": url, "headers": headers or {}})
        return FakeResponse(FakeAsyncClient.poll_responses.pop(0))


class AtlasH3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = (
            settings.OUTPUT_DIR,
            settings.PROJECTS_DIR,
            settings.ATLASCLOUD_API_KEY,
            settings.ATLASCLOUD_POLL_INTERVAL_SECONDS,
            settings.H3_PROVIDER,
            settings.H3_ATLAS_RESOLUTION,
            settings.H3_ATLAS_RATIO,
        )
        settings.OUTPUT_DIR = self.root / "legacy"
        settings.PROJECTS_DIR = self.root / "projects"
        settings.ATLASCLOUD_API_KEY = "sk-atlas-test"
        settings.ATLASCLOUD_POLL_INTERVAL_SECONDS = 0.01
        FakeAsyncClient.requests = []
        FakeAsyncClient.poll_responses = []

    def tearDown(self) -> None:
        (
            settings.OUTPUT_DIR,
            settings.PROJECTS_DIR,
            settings.ATLASCLOUD_API_KEY,
            settings.ATLASCLOUD_POLL_INTERVAL_SECONDS,
            settings.H3_PROVIDER,
            settings.H3_ATLAS_RESOLUTION,
            settings.H3_ATLAS_RATIO,
        ) = self.previous
        self.temp.cleanup()

    def test_parameter_mapping(self) -> None:
        self.assertEqual(atlas_resolution(1344, 768), "768P")
        self.assertEqual(atlas_resolution(1920, 1080), "1080P")
        self.assertEqual(atlas_ratio(1344, 768), "16:9")
        self.assertEqual(atlas_ratio(768, 1344), "9:16")
        self.assertEqual(atlas_ratio(1024, 1024), "1:1")
        self.assertEqual(atlas_duration(3.0), 4)
        self.assertEqual(atlas_duration(7.4), 7)
        self.assertEqual(atlas_duration(15.6), 15)

    def test_upload_media_returns_temporary_public_url(self) -> None:
        path = self.root / "ref.png"
        path.write_bytes(b"image-bytes")
        with patch("src.video_workflow.integrations.atlas_h3.httpx.AsyncClient", FakeAsyncClient):
            url = asyncio.run(AtlasH3Client().upload_media(path))
        self.assertEqual(url, "https://media.atlas.invalid/uploaded-reference.png")
        request = FakeAsyncClient.requests[0]
        self.assertTrue(request["url"].endswith("/api/v1/model/uploadMedia"))
        self.assertEqual(request["file"]["filename"], "ref.png")
        self.assertEqual(request["file"]["mime_type"], "image/png")

    def test_submit_rejects_inline_reference_before_network_request(self) -> None:
        with patch("src.video_workflow.integrations.atlas_h3.httpx.AsyncClient", FakeAsyncClient):
            with self.assertRaisesRegex(ValueError, r"HTTP\(S\) URL"):
                asyncio.run(
                    AtlasH3Client().submit(
                        prompt="测试",
                        refers=[{"url": "data:image/png;base64,AAAA", "type": "image"}],
                        resolution="768P",
                        duration=6,
                        ratio="16:9",
                    )
                )
        self.assertEqual(FakeAsyncClient.requests, [])

    def test_submit_and_poll_flow(self) -> None:
        FakeAsyncClient.poll_responses = [
            {"data": {"status": "processing", "outputs": []}},
            {"data": {"status": "completed", "outputs": ["https://cdn.example/video.mp4"]}},
        ]
        progress_calls: list[str] = []

        async def progress(value: float, message: str) -> None:
            progress_calls.append(message)

        with patch("src.video_workflow.integrations.atlas_h3.httpx.AsyncClient", FakeAsyncClient):
            client = AtlasH3Client()
            prediction_id = asyncio.run(
                client.submit(
                    prompt="测试提示词",
                    refers=[{"url": "https://media.atlas.invalid/reference.png", "type": "image"}],
                    resolution="768P",
                    duration=6,
                    ratio="16:9",
                )
            )
            outputs = asyncio.run(client.wait_for_result(prediction_id, progress))

        self.assertEqual(prediction_id, "pred-1")
        self.assertEqual(outputs, ["https://cdn.example/video.mp4"])
        submit = FakeAsyncClient.requests[0]
        self.assertTrue(submit["url"].endswith("/api/v1/model/generateVideo"))
        self.assertEqual(submit["headers"]["Authorization"], "Bearer sk-atlas-test")
        payload = submit["json"]
        self.assertEqual(ATLAS_MODEL_ID, "minimax/h3-developer/reference-to-video")
        self.assertEqual(payload["model"], ATLAS_MODEL_ID)
        self.assertEqual(payload["resolution"], "768P")
        self.assertEqual(payload["duration"], 6)
        self.assertEqual(payload["ratio"], "16:9")
        self.assertFalse(payload["prompt_expansion"])
        self.assertEqual(payload["refers"][0]["type"], "image")
        self.assertTrue(all("prediction/pred-1" in item["url"] for item in FakeAsyncClient.requests[1:]))
        self.assertTrue(progress_calls)

    def test_enqueue_uses_selected_h3_provider(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(ProjectBrief(title="Atlas", story="测试 Atlas 通道入队"))
        image = self.root / "first.png"
        image.write_bytes(b"image")
        asset = service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        shot_one = Shot(project_id=project.id, ordinal=1, narrative="测试", duration_seconds=6, keyframe_asset_id=asset.id, image_path=str(image))
        shot_two = Shot(project_id=project.id, ordinal=2, narrative="测试二", duration_seconds=6, keyframe_asset_id=asset.id, image_path=str(image))
        store.save_shot(shot_one)
        store.save_shot(shot_two)

        settings.H3_PROVIDER = "atlas_h3"
        jobs = RenderQueue(store, service).enqueue(project.id, [shot_one.id])
        self.assertEqual(jobs[0].provider, "atlas_h3")

        settings.H3_PROVIDER = "comfyui_h3"
        jobs = RenderQueue(store, service).enqueue(project.id, [shot_two.id])
        self.assertEqual(jobs[0].provider, "comfyui_h3")

    def test_atlas_submit_uses_manual_resolution_and_ratio(self) -> None:
        store = ProjectStore(self.root / "state.sqlite3")
        service = ProjectService(store)
        project = service.create_project(ProjectBrief(title="Atlas 参数", story="手动参数测试"))
        image = self.root / "manual.png"
        image.write_bytes(b"image")
        asset = service.register_existing_asset(project.id, image, AssetRole.KEYFRAME, "首帧")
        shot = Shot(project_id=project.id, ordinal=1, narrative="测试", duration_seconds=6, keyframe_asset_id=asset.id, image_path=str(image))
        store.save_shot(shot)
        settings.H3_PROVIDER = "atlas_h3"
        settings.H3_ATLAS_RESOLUTION = "1080P"
        settings.H3_ATLAS_RATIO = "9:16"
        jobs = RenderQueue(store, service).enqueue(project.id, [shot.id])

        queue = RenderQueue(store, service)
        captured: dict[str, object] = {}

        class FakeAtlasClient:
            async def upload_media(self, path: Path) -> str:
                return f"https://media.atlas.invalid/{path.name}"

            async def submit(self, **kwargs: object) -> str:
                captured.update(kwargs)
                return "pred-manual"

            async def wait_for_result(self, prediction_id: str, progress: object) -> list[str]:
                raise RuntimeError("stop-test")

        queue.atlas_client = FakeAtlasClient()  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            asyncio.run(queue._process_atlas_h3(store.get_job(jobs[0].id)))

        self.assertEqual(captured["resolution"], "1080P")
        self.assertEqual(captured["ratio"], "9:16")
        self.assertEqual(captured["refers"], [{"url": "https://media.atlas.invalid/manual.png", "type": "image"}])
        snapshot = store.get_job(jobs[0].id).input_snapshot["atlas"]
        self.assertEqual(snapshot["model"], "minimax/h3-developer/reference-to-video")
        self.assertEqual(snapshot["resolution"], "1080P")
        self.assertEqual(snapshot["ratio"], "9:16")

    @unittest.skipUnless(shutil.which(settings.FFMPEG_BIN) and shutil.which(settings.FFPROBE_BIN), "ffmpeg required")
    def test_atlas_without_tts_preserves_native_audio_and_rejects_silent_track(self) -> None:
        native_video = self.root / "atlas-native.mp4"
        silent_video = self.root / "atlas-silent.mp4"
        subprocess.run(
            [
                settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=2",
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(native_video),
            ],
            check=True,
        )
        subprocess.run(
            [
                settings.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(silent_video),
            ],
            check=True,
        )
        previous_audio = (settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE)
        settings.H3_POSTPROCESS_AUDIO = True
        settings.H3_AUDIO_MODE = "clean_tts"
        try:
            delivery = RenderQueue._replace_generated_audio(
                native_video,
                [],
                0.0,
                preserve_native_without_speech=True,
            )
            audible_qc = RenderQueue._validate_rendered_media(native_video, 320, 180, 2.0, False, True)
            silent_qc = RenderQueue._validate_rendered_media(silent_video, 320, 180, 2.0, False, True)
        finally:
            settings.H3_POSTPROCESS_AUDIO, settings.H3_AUDIO_MODE = previous_audio

        self.assertFalse(delivery["applied"])
        self.assertEqual(delivery["delivered"], "native_h3_audio")
        self.assertEqual(delivery["native_h3_audio"], "preserved")
        self.assertTrue(audible_qc["passed"], audible_qc["issues"])
        self.assertGreater(float(audible_qc["audio_max_volume_db"]), -60.0)
        self.assertFalse(silent_qc["passed"])
        self.assertTrue(any("实际为静音" in issue for issue in silent_qc["issues"]))


if __name__ == "__main__":
    unittest.main()
