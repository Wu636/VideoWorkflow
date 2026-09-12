from __future__ import annotations

import socket
import unittest
import unittest.mock
import asyncio

from src.video_workflow.config import settings
from src.video_workflow.domain import (
    Asset,
    AssetRole,
    AssetType,
    CharacterProfile,
    DialogueTurn,
    Project,
    ProjectBrief,
    SeedanceReferenceMode,
    Shot,
    VisualBeat,
    VoiceEvent,
)
from src.video_workflow.integrations.seedance import (
    SEEDANCE_MODELS,
    SeedanceClient,
    _is_dns_failure,
    estimate_seedance_cost,
    seedance_catalog,
)
from src.video_workflow.services.projects import ProjectService


class SeedanceIntegrationTests(unittest.TestCase):
    def test_official_model_catalog_and_720p_estimates(self) -> None:
        self.assertEqual(
            set(SEEDANCE_MODELS),
            {
                "doubao-seedance-2-0-260128",
                "doubao-seedance-2-0-fast-260128",
                "doubao-seedance-2-0-mini-260615",
            },
        )
        catalog = seedance_catalog()
        self.assertEqual(len(catalog["models"]), 3)
        self.assertEqual(catalog["render_concurrency"], settings.SEEDANCE_RENDER_CONCURRENCY)
        self.assertEqual(
            {item["example_720p_yuan_per_second"] for item in catalog["models"]},
            {0.9936, 0.7992, 0.4968},
        )

        mini = estimate_seedance_cost(
            "doubao-seedance-2-0-mini-260615", "720p", "16:9", [5]
        )
        self.assertEqual(mini["estimated_tokens"], 108900)
        self.assertEqual(mini["estimated_yuan"], 2.5047)
        self.assertEqual(mini["estimated_yuan_per_second"], 0.4968)
        self.assertEqual(mini["estimated_yuan_per_task"], 0.0207)

        mini_with_video = estimate_seedance_cost(
            "doubao-seedance-2-0-mini-260615",
            "720p",
            "16:9",
            [5],
            has_video_input=True,
        )
        self.assertEqual(mini_with_video["unit_price_per_million_tokens"], 14.0)
        self.assertEqual(mini_with_video["estimated_yuan_per_second"], 0.3024)

    def test_content_preserves_modality_order_and_roles(self) -> None:
        assets = [
            Asset(
                id="image-a",
                project_id="project-a",
                type=AssetType.IMAGE,
                role=AssetRole.KEYFRAME,
                name="first.png",
                path="https://assets.example.invalid/first.png",
                mime_type="image/png",
            ),
            Asset(
                id="audio-a",
                project_id="project-a",
                type=AssetType.AUDIO,
                role=AssetRole.VOICE,
                name="voice.mp3",
                path="https://assets.example.invalid/voice.mp3",
                mime_type="audio/mpeg",
            ),
            Asset(
                id="image-b",
                project_id="project-a",
                type=AssetType.IMAGE,
                role=AssetRole.CHARACTER,
                name="character.png",
                path="https://assets.example.invalid/character.png",
                mime_type="image/png",
            ),
            Asset(
                id="video-a",
                project_id="project-a",
                type=AssetType.VIDEO,
                role=AssetRole.MOTION,
                name="motion.mp4",
                path="https://assets.example.invalid/motion.mp4",
                mime_type="video/mp4",
            ),
        ]
        content = SeedanceClient().build_content("测试提示词", assets)
        self.assertEqual([item["type"] for item in content], [
            "text", "image_url", "audio_url", "image_url", "video_url"
        ])
        self.assertEqual(
            [item.get("role") for item in content[1:]],
            ["reference_image", "reference_audio", "reference_image", "reference_video"],
        )

        first_frame_content = SeedanceClient().build_content(
            "连续镜头",
            assets,
            first_frame_asset_id="image-a",
        )
        # Ark rejects strict first_frame mixed with all-modal reference media.
        # The keyframe therefore remains 图片1/reference_image in this mode.
        self.assertEqual(first_frame_content[1]["role"], "reference_image")
        self.assertEqual(first_frame_content[3]["role"], "reference_image")
        self.assertNotIn("first_frame", [item.get("role") for item in first_frame_content])

        strict_first_frame_content = SeedanceClient().build_content(
            "只有首帧",
            [assets[0]],
            first_frame_asset_id="image-a",
        )
        self.assertEqual(strict_first_frame_content[1]["role"], "first_frame")

    def test_last_frame_url_accepts_official_response_shapes(self) -> None:
        client = SeedanceClient()
        self.assertEqual(
            client.last_frame_url({"content": {"last_frame_url": "https://cdn.example/last.png"}}),
            "https://cdn.example/last.png",
        )
        self.assertEqual(
            client.last_frame_url({"last_frame": {"url": "https://cdn.example/last-2.png"}}),
            "https://cdn.example/last-2.png",
        )
        self.assertIsNone(client.last_frame_url({"content": {"video_url": "https://cdn.example/video.mp4"}}))

    def test_submit_uses_official_first_frame_role_without_internal_asset_field(self) -> None:
        from src.video_workflow.integrations import seedance as seedance_module

        captured: dict[str, object] = {}

        class FakeResponse:
            is_error = False
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict[str, str]:
                return {"id": "task-1"}

        class FakeAsyncClient:
            def __init__(self, **kwargs: object) -> None:
                captured["client_kwargs"] = kwargs

            async def __aenter__(self) -> "FakeAsyncClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
                captured["path"] = path
                captured["payload"] = json
                return FakeResponse()

        first_frame = Asset(
            id="first-frame",
            project_id="project-a",
            type=AssetType.IMAGE,
            role=AssetRole.KEYFRAME,
            name="first.png",
            path="data:image/png;base64,AA==",
            mime_type="image/png",
        )
        client = SeedanceClient()
        client.api_key = "test-key"

        async def submit() -> None:
            with unittest.mock.patch.object(seedance_module.httpx, "AsyncClient", FakeAsyncClient):
                task_id, snapshot = await client.submit(
                    model_id="doubao-seedance-2-0-mini-260615",
                    prompt="连续镜头",
                    assets=[first_frame],
                    ratio="16:9",
                    duration=6.2,
                    resolution="480p",
                    seed=123,
                    generate_audio=True,
                    first_frame_asset_id=first_frame.id,
                )
            self.assertEqual(task_id, "task-1")
            self.assertEqual(snapshot["first_frame_asset_id"], first_frame.id)

        asyncio.run(submit())
        payload = captured["payload"]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertNotIn("first_frame_asset_id", payload)
        self.assertIs(payload["return_last_frame"], True)
        self.assertEqual(payload["content"][1]["role"], "first_frame")

    def test_prompt_uses_exact_asset_numbers_and_binds_dialogue_speaker(self) -> None:
        lead = CharacterProfile(
            id="lead",
            name="秦绍辉",
            description="戴黑框眼镜的年轻男性",
            wardrobe="深蓝西装",
            reference_asset_ids=["character-image"],
        )
        listener = CharacterProfile(id="listener", name="D同志", description="短发女性")
        project = Project(
            id="project-a",
            brief=ProjectBrief(title="谈话", story="秦绍辉介绍D同志", visual_style="皮克斯3D动画"),
            characters=[lead, listener],
        )
        shot = Shot(
            id="shot-a",
            project_id=project.id,
            ordinal=1,
            narrative="秦绍辉向C同志介绍D同志",
            scene_description="谈话室外间",
            character_ids=[lead.id, listener.id],
            dialogue="她是D同志。",
            dialogue_speaker_id=lead.id,
            dialogue_turns=[DialogueTurn(speaker_id=lead.id, text="她是D同志。")],
            keyframe_asset_id="keyframe",
            reference_asset_ids=["character-image", "voice"],
            seedance_reference_mode=SeedanceReferenceMode.STRICT_FIRST_FRAME,
        )
        assets = [
            Asset(id="keyframe", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.KEYFRAME, name="shot.png", path="data:image/png;base64,AA=="),
            Asset(id="character-image", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.CHARACTER, name="lead.png", path="data:image/png;base64,AA=="),
            Asset(id="voice", project_id=project.id, type=AssetType.AUDIO, role=AssetRole.VOICE, name="lead.mp3", path="data:audio/mpeg;base64,AA=="),
        ]

        # The compiler itself does not access storage, so a lightweight service is sufficient.
        service = ProjectService.__new__(ProjectService)
        prompt = service.compile_seedance_prompt(project, shot, assets)
        self.assertIn("@图片1（shot.png）", prompt)
        self.assertNotIn("lead.png", prompt)
        self.assertNotIn("lead.mp3", prompt)
        self.assertIn("@图片1是严格首帧", prompt)
        self.assertIn("“秦绍辉”说：“她是D同志。”", prompt)
        self.assertIn("其余人物闭嘴", prompt)
        self.assertIn("【逐段时间轴】", prompt)

    def test_auto_reference_mode_keeps_character_sheet_when_body_appears_after_frame_zero(self) -> None:
        lead = CharacterProfile(
            id="lead",
            name="张小差",
            description="半扎高丸子头的年轻阴差",
            wardrobe="炭黑色立领工装和双肩皮带",
            reference_asset_ids=["character-image"],
        )
        project = Project(
            id="project-reveal",
            brief=ProjectBrief(title="穿墙", story="张小差从墙里钻出来"),
            characters=[lead],
        )
        shot = Shot(
            id="shot-reveal",
            project_id=project.id,
            ordinal=3,
            narrative="墙壁先出现轮廓，张小差随后从墙里钻出",
            scene_description="老式客厅里墙面平整，李奶奶坐在沙发上",
            keyframe_prompt="【首帧画面】老式客厅里墙面平整，李奶奶坐在沙发上\n【角色锚点】张小差：炭黑色工装",
            subject_motion="张小差半个脑袋从墙内探出，随后全身钻出",
            character_ids=[lead.id],
            keyframe_asset_id="keyframe",
            reference_asset_ids=["character-image"],
        )
        assets = [
            Asset(id="keyframe", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.KEYFRAME, name="shot-3.png", path="data:image/png;base64,AA=="),
            Asset(id="character-image", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.CHARACTER, name="zhang-sheet.png", path="data:image/png;base64,AA==", character_id=lead.id),
        ]
        service = ProjectService.__new__(ProjectService)

        diagnostics = service.seedance_material_diagnostics(project, shot, assets)
        prompt = service.compile_seedance_prompt(project, shot, assets)

        self.assertEqual(diagnostics["resolved_mode"], "multimodal_reference")
        self.assertEqual([item["id"] for item in diagnostics["materials"]], ["keyframe", "character-image"])
        self.assertIsNone(diagnostics["first_frame_asset_id"])
        self.assertEqual(diagnostics["identity_risk_characters"], ["张小差"])
        self.assertIn("将@图片2中", prompt)
        self.assertIn("其脸型、发型、服装和配饰以对应人物参考图为准", prompt)

    def test_strict_first_frame_prompt_uses_timed_beats_and_external_system_voice(self) -> None:
        lead = CharacterProfile(id="lead", name="林砚", description="年轻工程师")
        project = Project(
            id="project-director",
            brief=ProjectBrief(title="闯关", story="系统发布挑战", visual_style="电影级 3D"),
            characters=[lead],
        )
        shot = Shot(
            id="shot-director",
            project_id=project.id,
            ordinal=1,
            duration_seconds=15,
            narrative="系统发布第一关，林砚启动设备并完成校验",
            character_ids=[lead.id],
            dialogue="第一关开始",
            keyframe_asset_id="keyframe",
            reference_asset_ids=["character", "voice"],
            seedance_reference_mode=SeedanceReferenceMode.STRICT_FIRST_FRAME,
            visual_beats=[
                VisualBeat(start_seconds=0, end_seconds=3, purpose="钩子", subject_action="设备突然亮起", camera_motion="快速推近"),
                VisualBeat(start_seconds=3, end_seconds=7, purpose="执行", subject_action="林砚抓住绝缘杆", camera_motion="侧向跟拍"),
                VisualBeat(start_seconds=7, end_seconds=11, purpose="反馈", subject_action="能量脉冲沿线路传播", environment_action="指示灯依次点亮", camera_motion="平稳摇摄"),
                VisualBeat(start_seconds=11, end_seconds=15, purpose="结果", subject_action="校验灯转为绿色", camera_motion="固定镜头"),
            ],
            voice_events=[
                VoiceEvent(
                    kind="system_vo",
                    speaker_name="系统播报",
                    text="第一关开始",
                    start_seconds=0.4,
                    end_seconds=2.2,
                    lip_sync=False,
                )
            ],
        )
        assets = [
            Asset(id="keyframe", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.KEYFRAME, name="shot-1.png", path="data:image/png;base64,AA==", description="KEYFRAME_FULL_PROMPT_SHOULD_NOT_LEAK"),
            Asset(id="character", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.CHARACTER, name="linyan.png", path="data:image/png;base64,AA=="),
            Asset(id="voice", project_id=project.id, type=AssetType.AUDIO, role=AssetRole.VOICE, name="system.mp3", path="data:audio/mpeg;base64,AA=="),
        ]

        service = ProjectService.__new__(ProjectService)
        refs = service.seedance_reference_assets(shot, assets, project)
        prompt = service.compile_seedance_prompt(project, shot, assets)

        self.assertEqual([asset.id for asset in refs], ["keyframe"])
        self.assertNotIn("KEYFRAME_FULL_PROMPT_SHOULD_NOT_LEAK", prompt)
        self.assertIn("0.00–3.00秒", prompt)
        self.assertIn("11.00–15.00秒", prompt)
        self.assertIn("系统播报说", prompt)
        self.assertIn("画面内所有人物不张嘴", prompt)
        self.assertNotIn("唯一发言者“林砚”", prompt)

    def test_legacy_system_broadcast_and_old_cached_prompt_are_upgraded(self) -> None:
        lead = CharacterProfile(id="lead", name="林砚")
        project = Project(
            id="project-legacy",
            brief=ProjectBrief(title="旧项目", story="系统发布任务"),
            characters=[lead],
        )
        shot = Shot(
            id="shot-legacy",
            project_id=project.id,
            ordinal=1,
            duration_seconds=15,
            narrative="系统发布任务，林砚抬头观察设备",
            dialogue="【系统播报】第一关开始。",
            dialogue_speaker_id=lead.id,
            audio_design="冰冷机械系统播报，无人物对白",
            subject_motion=(
                "总时长约15秒。林砚抬头并走向设备；设备亮起；林砚完成操作。"
                "本镜头只完成这一项核心动作；镜头固定或仅做一次缓慢单向移动。"
            ),
            keyframe_asset_id="keyframe",
            seedance_prompt="OLD_CACHED_PROMPT",
            seedance_prompt_version="seedance-2.0-official-2026-08",
        )
        assets = [
            Asset(id="keyframe", project_id=project.id, type=AssetType.IMAGE, role=AssetRole.KEYFRAME, name="shot.png", path="data:image/png;base64,AA=="),
        ]

        service = ProjectService.__new__(ProjectService)
        prompt = service.effective_seedance_prompt(project, shot, assets)

        self.assertNotEqual(prompt, "OLD_CACHED_PROMPT")
        self.assertIn("0.00–3.00秒", prompt)
        self.assertNotIn("只完成这一项核心动作", prompt)
        self.assertNotIn("缓慢单向", prompt)
        self.assertIn("系统播报说：“第一关开始。”", prompt)
        self.assertIn("画面内所有人物不张嘴", prompt)

    def test_dns_failure_detection_and_preflight_origin_probe(self) -> None:
        import httpx

        from src.video_workflow.integrations import seedance as seedance_module

        root = socket.gaierror(-2, "Name or service not known")
        wrapped = ValueError("boom")
        wrapped.__cause__ = root
        self.assertTrue(_is_dns_failure(wrapped))
        self.assertFalse(_is_dns_failure(ValueError("plain timeout")))

        connect_error = httpx.ConnectError("[Errno -2] Name or service not known")
        connect_error.__cause__ = socket.gaierror(-2, "Name or service not known")

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url):
                raise connect_error

        original_origin = settings.SEEDANCE_PUBLIC_ASSET_BASE_URL
        original_key = settings.ARK_API_KEY
        settings.SEEDANCE_PUBLIC_ASSET_BASE_URL = "https://seedance-probe.invalid"
        settings.ARK_API_KEY = "test-key"
        try:
            with unittest.mock.patch.object(seedance_module.httpx, "Client", _FakeClient):
                probe = SeedanceClient().probe_asset_origin()
                preflight = asyncio.run(SeedanceClient().preflight("doubao-seedance-2-0-mini-260615", "480p"))
        finally:
            settings.SEEDANCE_PUBLIC_ASSET_BASE_URL = original_origin
            settings.ARK_API_KEY = original_key
        self.assertFalse(probe["ok"])
        self.assertIn("隧道域名无法解析", str(probe["error"]))
        self.assertFalse(preflight["ok"])
        self.assertFalse(preflight["asset_origin_ok"])
        self.assertIn("素材公网通道不可用", preflight["message"])

    def test_local_images_inline_as_base64_and_transient_error_classification(self) -> None:
        import base64 as base64_module
        import tempfile
        from pathlib import Path

        from src.video_workflow.services.render_queue import _is_transient_seedance_error

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "frame.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            image = Asset(
                id="k1",
                project_id="p1",
                type=AssetType.IMAGE,
                role=AssetRole.KEYFRAME,
                name="frame.png",
                path=str(image_path),
            )
            data_url = SeedanceClient._asset_url(image)
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            self.assertEqual(base64_module.b64decode(data_url.split(",", 1)[1]), image_path.read_bytes())

            video_path = Path(temp) / "ref.mp4"
            video_path.write_bytes(b"fake-video")
            video = Asset(
                id="v1",
                project_id="p1",
                type=AssetType.VIDEO,
                role=AssetRole.OTHER,
                name="ref.mp4",
                path=str(video_path),
            )
            original = settings.SEEDANCE_PUBLIC_ASSET_BASE_URL
            settings.SEEDANCE_PUBLIC_ASSET_BASE_URL = ""
            try:
                with self.assertRaises(ValueError):
                    SeedanceClient._asset_url(video)
                settings.SEEDANCE_PUBLIC_ASSET_BASE_URL = "https://example-tunnel.invalid"
                url = SeedanceClient._asset_url(video)
                self.assertTrue(url.startswith("https://example-tunnel.invalid/api/projects/seedance-assets/v1?"))
            finally:
                settings.SEEDANCE_PUBLIC_ASSET_BASE_URL = original

        self.assertTrue(_is_transient_seedance_error(ValueError("content[1].image_url 的素材公网地址不可读取：")))
        self.assertTrue(_is_transient_seedance_error(RuntimeError("Seedance 提交失败 (400): timeout while fetching resource")))
        self.assertTrue(_is_transient_seedance_error(RuntimeError("Seedance 提交失败 (429): concurrent limit")))
        self.assertTrue(_is_transient_seedance_error(RuntimeError("50430 Request Has Reached API Concurrent Limit")))
        self.assertFalse(_is_transient_seedance_error(ValueError("Unsupported resolution: 360p")))


if __name__ == "__main__":
    unittest.main()
