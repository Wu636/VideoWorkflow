from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.domain import Asset, AssetRole, AssetType, CharacterProfile, ProjectBrief, Shot
from src.video_workflow.generators.image import _build_image_prompt
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore
from src.video_workflow.types import Scene, Storyboard


class KeyframePromptSeparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_projects_dir = settings.PROJECTS_DIR
        settings.PROJECTS_DIR = self.root / "projects"
        self.store = ProjectStore(self.root / "state.sqlite3")
        self.service = ProjectService(self.store)

    def tearDown(self) -> None:
        settings.PROJECTS_DIR = self.previous_projects_dir
        self.temp.cleanup()

    def test_legacy_shot_uses_scene_description_as_keyframe_prompt(self) -> None:
        shot = Shot.model_validate(
            {
                "project_id": "project_legacy",
                "ordinal": 1,
                "scene_description": "干净的静态开场画面",
                "visual_prompt": "通用风格、全部角色和整镜描述",
            }
        )

        self.assertEqual(shot.keyframe_prompt, "干净的静态开场画面")
        self.assertEqual(shot.visual_prompt, "通用风格、全部角色和整镜描述")

    def test_storyboard_keeps_three_prompts_separate(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="三类 Prompt", story="人物走进办公室", visual_style="柔和 3D")
        )

        class FakeLLM:
            async def generate_storyboard(self, **_: object) -> Storyboard:
                return Storyboard(
                    topic="测试",
                    scenes=[
                        Scene(
                            id=1,
                            narrative="人物走进办公室",
                            visual_prompt="整镜描述：人物进门后走到桌边",
                            keyframe_prompt="开场静态画面：人物站在门外，右手刚触碰门把手",
                            motion_prompt="人物推门并走到桌边",
                            duration=5,
                        )
                    ],
                )

        class FakeOrchestrator:
            def __init__(self, *_: object, **__: object) -> None:
                self.llm = FakeLLM()

        with patch("src.video_workflow.services.projects.WorkflowOrchestrator", FakeOrchestrator):
            shots = asyncio.run(self.service.generate_storyboard(project.id, 1))

        self.assertIn("整镜描述：人物进门后走到桌边", shots[0].visual_prompt)
        self.assertIn("开场静态画面：人物站在门外", shots[0].keyframe_prompt)
        self.assertIn("人物推门并走到桌边", shots[0].video_prompt)
        self.assertNotEqual(shots[0].visual_prompt, shots[0].keyframe_prompt)

    def test_keyframe_generation_submits_only_keyframe_prompt(self) -> None:
        project = self.service.create_project(ProjectBrief(title="首帧生成", story="测试"))
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                visual_prompt="GENERIC_STORYBOARD_PROMPT",
                keyframe_prompt="STATIC_KEYFRAME_PROMPT",
                video_prompt="MINIMAX_H3_PROMPT",
            )
        )
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(self, scene: Scene, output_dir: str, *_: object, **__: object) -> str:
                captured["prompt"] = scene.visual_prompt
                output = Path(output_dir) / "keyframe.png"
                output.write_bytes(b"image")
                return str(output)

        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            generated = asyncio.run(self.service.generate_keyframes(project.id, [shot.id]))

        self.assertEqual(captured["prompt"], "STATIC_KEYFRAME_PROMPT")
        self.assertEqual(generated[0].visual_prompt, "GENERIC_STORYBOARD_PROMPT")
        self.assertEqual(generated[0].keyframe_prompt, "STATIC_KEYFRAME_PROMPT")
        self.assertEqual(generated[0].video_prompt, "MINIMAX_H3_PROMPT")

    def test_explicit_empty_character_description_suppresses_global_default(self) -> None:
        scene = Scene(id=1, narrative="", visual_prompt="纯黑标题卡", motion_prompt="", duration=5)
        previous = settings.CHARACTER_DESCRIPTION
        settings.CHARACTER_DESCRIPTION = "不应出现的全局角色"
        try:
            prompt = _build_image_prompt(scene, character_description="", image_style="")
        finally:
            settings.CHARACTER_DESCRIPTION = previous

        self.assertEqual(prompt, "纯黑标题卡")

    def test_title_card_drops_legacy_all_character_references_and_duplicate_suggestion(self) -> None:
        project = self.service.create_project(
            ProjectBrief(title="片尾标题", story="测试", visual_style="低饱和度 3D")
        )
        first_asset = self.store.save_asset(
            Asset(
                project_id=project.id,
                type=AssetType.IMAGE,
                role=AssetRole.CHARACTER,
                name="甲参考",
                path=str(self.root / "first.png"),
            )
        )
        second_asset = self.store.save_asset(
            Asset(
                project_id=project.id,
                type=AssetType.IMAGE,
                role=AssetRole.CHARACTER,
                name="乙参考",
                path=str(self.root / "second.png"),
            )
        )
        (self.root / "first.png").write_bytes(b"first")
        (self.root / "second.png").write_bytes(b"second")
        project.characters = [
            CharacterProfile(
                name="甲",
                description="甲的外观",
                voice_description="甲的声音",
                reference_asset_ids=[first_asset.id],
            ),
            CharacterProfile(
                name="乙",
                description="乙的外观",
                voice_description="乙的声音",
                reference_asset_ids=[second_asset.id],
            ),
        ]
        self.store.save_project(project)
        title_prompt = "画面渐暗至全黑，中央浮现白色书法字"
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                keyframe_prompt=title_prompt,
                character_ids=[character.id for character in project.characters],
                reference_asset_ids=[first_asset.id, second_asset.id],
            )
        )
        captured: dict[str, object] = {}

        class FakeImageGenerator:
            async def generate_image(
                self,
                scene: Scene,
                output_dir: str,
                reference_image_path: str | None = None,
                **kwargs: object,
            ) -> str:
                captured["prompt"] = scene.visual_prompt
                captured["references"] = reference_image_path
                captured.update(kwargs)
                output = Path(output_dir) / "keyframe.png"
                output.write_bytes(b"image")
                return str(output)

        with patch(
            "src.video_workflow.services.projects.create_image_generator",
            return_value=FakeImageGenerator(),
        ):
            asyncio.run(
                self.service.generate_keyframes(
                    project.id,
                    [shot.id],
                    user_suggestions=title_prompt,
                )
            )

        self.assertEqual(captured["prompt"], title_prompt)
        self.assertEqual(captured["character_description"], "")
        self.assertIsNone(captured["references"])
        self.assertEqual(captured["image_style"], "低饱和度 3D")

    def test_keyframe_uses_only_named_character_visual_details(self) -> None:
        project = self.service.create_project(ProjectBrief(title="单人镜头", story="测试"))
        project.characters = [
            CharacterProfile(name="甲", description="甲的外观", voice_description="甲的声音"),
            CharacterProfile(name="乙", description="乙的外观", voice_description="乙的声音"),
        ]
        shot = Shot(
            project_id=project.id,
            ordinal=1,
            keyframe_prompt="甲站在窗边",
            character_ids=[character.id for character in project.characters],
        )

        selected = self.service.keyframe_character_ids(project, shot, shot.keyframe_prompt)
        description = self.service.character_visual_bible(project, selected)

        self.assertEqual(selected, [project.characters[0].id])
        self.assertIn("甲的外观", description)
        self.assertNotIn("甲的声音", description)
        self.assertNotIn("乙", description)


if __name__ == "__main__":
    unittest.main()
