from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.core.orchestrator import (
    create_llm_generator,
    resolve_reference_llm_provider,
)
from src.video_workflow.generators.llm import (
    OpenLuxGenerator,
    _normalize_storyboard_payload,
    _parse_json_object,
)
from src.video_workflow.runtime_settings import runtime_settings
from src.video_workflow.speech_budget import spoken_character_count


class _FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.completions = _FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


class OpenLuxGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings_patches = [
            patch.object(settings, "OPENLUX_API_KEY", "openlux-test-key"),
            patch.object(settings, "OPENLUX_BASE_URL", "https://api.openlux.ai/v1"),
            patch.object(settings, "OPENLUX_MODEL", "claude-sonnet-5"),
            patch.object(settings, "OPENLUX_VISION_MODEL", "gpt-5.6-sol"),
            patch.object(settings, "OPENLUX_STREAM", False),
            patch.object(settings, "OPENLUX_SAVE_RAW_RESPONSES", False),
        ]
        for item in self.settings_patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.settings_patches):
            item.stop()

    async def test_multimodal_json_uses_vision_model_and_embeds_local_image(self) -> None:
        generator = OpenLuxGenerator()
        generator.client = _FakeClient(["结果如下：\n```json\n{\"style\": \"皮克斯3D\"}\n```"])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "reference.png"
            image_path.write_bytes(b"fake-png")
            result = await generator.generate_json(
                "只输出 JSON",
                "分析参考图",
                reference_images=[str(image_path)],
            )

        self.assertEqual(result, {"style": "皮克斯3D"})
        call = generator.client.completions.calls[0]
        self.assertEqual(call["model"], "gpt-5.6-sol")
        user_content = call["messages"][1]["content"]
        self.assertEqual([item["type"] for item in user_content], ["image_url", "text"])
        self.assertTrue(user_content[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    async def test_storyboard_uses_text_model_and_keeps_requested_count_instruction(self) -> None:
        payload = {
            "topic": "测试",
            "scenes": [
                {
                    "id": 1,
                    "duration": 6,
                    "narrative": "人物抬头看向窗外。",
                    "visual_prompt": "室内中景，柔和侧光。",
                    "motion_prompt": "人物缓慢抬头并停住。",
                }
            ],
        }
        generator = OpenLuxGenerator()
        generator.client = _FakeClient([json.dumps(payload, ensure_ascii=False)])
        storyboard = await generator.generate_storyboard("测试剧情", count=1)

        self.assertEqual(len(storyboard.scenes), 1)
        call = generator.client.completions.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-5")
        prompt = call["messages"][1]["content"][-1]["text"]
        self.assertIn("恰好 1 个分镜", prompt)
        self.assertIn("不要事后截断", prompt)

    async def test_streaming_response_is_saved_before_json_parsing(self) -> None:
        class StreamingCompletions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)

                async def chunks():
                    for text in ('{"style":', ' "电影感"}'):
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                        )

                return chunks()

        completions = StreamingCompletions()
        generator = OpenLuxGenerator()
        generator.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(settings, "OPENLUX_STREAM", True),
                patch.object(settings, "OPENLUX_SAVE_RAW_RESPONSES", True),
                patch.object(settings, "OUTPUT_DIR", Path(directory)),
            ):
                result = await generator.generate_json("system", "user")
                saved_path = generator.last_response_path
                saved_content = json.loads(Path(saved_path).read_text(encoding="utf-8"))["content"]

        self.assertEqual(result, {"style": "电影感"})
        self.assertTrue(completions.calls[0]["stream"])
        self.assertIsNotNone(saved_path)
        self.assertEqual(saved_content, '{"style": "电影感"}')

    def test_client_disables_invisible_paid_retries(self) -> None:
        generator = OpenLuxGenerator()
        self.assertEqual(generator.client.max_retries, 0)

    def test_storyboard_normalization_keeps_fifteen_second_action_arc_and_system_vo(self) -> None:
        payload = _normalize_storyboard_payload(
            {
                "scenes": [
                    {
                        "id": 1,
                        "duration": 15,
                        "visual_prompt": "角色面前的闯关设备亮起",
                        "motion_prompt": "角色抬头；冲向设备；设备展开；能量脉冲穿过场景；角色完成挑战",
                        "dialogue": "第一关开始",
                        "dialogue_speaker": "系统播报（非角色台词）",
                    }
                ]
            },
            include_dialogue=True,
        )

        scene = payload["scenes"][0]
        self.assertEqual(scene["duration"], 15)
        self.assertGreaterEqual(len(scene["visual_beats"]), 4)
        self.assertEqual(scene["visual_beats"][0]["start_seconds"], 0)
        self.assertEqual(scene["visual_beats"][-1]["end_seconds"], 15)
        self.assertNotIn("总时长约15秒", scene["visual_beats"][0]["subject_action"])
        self.assertNotIn("极慢", scene["motion_prompt"])
        self.assertEqual(scene["voice_events"][0]["kind"], "system_vo")
        self.assertFalse(scene["voice_events"][0]["lip_sync"])
        self.assertEqual(scene["text_policy"], "post_overlay")

    def test_storyboard_normalization_compacts_dense_voice_events_to_normal_speed_budget(self) -> None:
        payload = _normalize_storyboard_payload(
            {
                "scenes": [
                    {
                        "duration": 15,
                        "event": "系统宣布高危副本规则",
                        "voice_events": [
                            {
                                "kind": "system_vo",
                                "speaker_name": "系统",
                                "text": "【副本任务：双人配合登杆，一人上杆并完成转位，一人专职地面监护，禁止单人登杆】",
                                "start_seconds": 2,
                                "end_seconds": 6,
                            },
                            {
                                "kind": "system_vo",
                                "speaker_name": "系统",
                                "text": "【核心铁律：登杆前必须全方位检查杆身、杆基、拉线，核验安全装备，从杆底起步攀爬每3米复检】",
                                "start_seconds": 6,
                                "end_seconds": 12,
                            },
                            {
                                "kind": "system_vo",
                                "speaker_name": "系统",
                                "text": "【失败惩罚：任何细节违规，即刻抹杀】",
                                "start_seconds": 12,
                                "end_seconds": 14.5,
                            },
                        ],
                    }
                ]
            },
            include_dialogue=True,
        )

        events = payload["scenes"][0]["voice_events"]
        self.assertLessEqual(sum(spoken_character_count(event["text"]) for event in events), 51)
        self.assertTrue(all(event["end_seconds"] > event["start_seconds"] for event in events))
        self.assertTrue(all(
            current["end_seconds"] < following["start_seconds"]
            for current, following in zip(events, events[1:], strict=False)
        ))
        self.assertTrue(all(not event["lip_sync"] for event in events))

    def test_factory_and_auto_vision_route_select_openlux(self) -> None:
        self.assertIsInstance(create_llm_generator("openlux"), OpenLuxGenerator)
        with (
            patch.object(settings, "REFERENCE_ANALYSIS_PROVIDER", "auto"),
            patch.object(settings, "LLM_PROVIDER", "openlux"),
            patch.object(settings, "ARK_API_KEY", "ark-also-configured"),
        ):
            self.assertEqual(resolve_reference_llm_provider(), "openlux")

    def test_runtime_settings_expose_openlux_fields_and_routes(self) -> None:
        payload = runtime_settings.public_payload()
        llm_group = next(group for group in payload["groups"] if group["id"] == "llm")
        keys = {field["key"] for field in llm_group["fields"]}
        self.assertTrue({
            "OPENLUX_API_KEY",
            "OPENLUX_BASE_URL",
            "OPENLUX_MODEL",
            "OPENLUX_VISION_MODEL",
            "OPENLUX_REQUEST_TIMEOUT_SECONDS",
            "OPENLUX_STREAM",
            "OPENLUX_SAVE_RAW_RESPONSES",
        }.issubset(keys))
        storyboard_route = next(route for route in payload["routes"] if route["id"] == "storyboard")
        self.assertIn("openlux", {option["value"] for option in storyboard_route["options"]})


class JsonRecoveryTests(unittest.TestCase):
    def test_recovers_json_after_provider_preamble(self) -> None:
        self.assertEqual(
            _parse_json_object("好的，结果如下：\n{\"ok\": true}", "OpenLux"),
            {"ok": True},
        )


if __name__ == "__main__":
    unittest.main()
