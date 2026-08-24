from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.domain import ProjectBrief, Shot
from src.video_workflow.h3_prompt_skills import (
    OFFICIAL_H3_SKILLS_COMMIT,
    get_h3_prompt_skill,
    list_h3_prompt_skills,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


class H3PromptSkillsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.previous_projects_dir = settings.PROJECTS_DIR
        settings.PROJECTS_DIR = root / "projects"
        self.store = ProjectStore(root / "state.sqlite3")
        self.service = ProjectService(self.store)

    def tearDown(self) -> None:
        settings.PROJECTS_DIR = self.previous_projects_dir
        self.temp.cleanup()

    def test_catalog_exposes_general_and_eight_official_styles(self) -> None:
        skills = list_h3_prompt_skills()
        self.assertEqual(len(skills), 9)
        self.assertEqual(skills[0]["id"], "h3-prompt-writing")
        self.assertEqual(skills[0]["source_commit"], OFFICIAL_H3_SKILLS_COMMIT)
        self.assertEqual(get_h3_prompt_skill("paper-collage-explainer-generator").version, "0.3.9")
        with self.assertRaisesRegex(ValueError, "未知的 H3 Prompt Skill"):
            get_h3_prompt_skill("missing-style")

    async def test_selected_skill_generates_and_persists_h3_prompt(self) -> None:
        project = self.service.create_project(
            ProjectBrief(
                title="纸艺测试",
                story="一张纸折成小船。",
                target_duration_seconds=5,
                visual_style="温暖手作风",
            )
        )
        shot = self.store.save_shot(
            Shot(
                project_id=project.id,
                ordinal=1,
                title="纸船成形",
                narrative="纸张逐步折成一艘小船",
                subject_motion="纸张沿折痕逐帧折叠",
                keyframe_prompt="桌面上的白纸",
            )
        )
        captured: dict[str, str] = {}

        class FakeLLM:
            async def generate_json(self, system_prompt: str, user_prompt: str, reference_images=None):
                captured["system"] = system_prompt
                captured["user"] = user_prompt
                return {
                    "video_prompt": (
                        "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                        "(from [Shot 1]) is fully referenced.\n\n"
                        "integrated_multimodal_description: [Shot 1] Layered cardstock folds frame by frame.\n"
                        "overall_soundscape: Soft paper creases and a light tabletop tap.\n"
                        "non_diegetic_music: None."
                    )
                }

        with patch("src.video_workflow.services.projects.create_llm_generator", return_value=FakeLLM()):
            generated = await self.service.generate_h3_prompts(
                project.id,
                [shot.id],
                "papercraft-stop-motion-explainer",
                "保持白纸颜色",
            )

        self.assertEqual(len(generated), 1)
        saved = self.service.require_shot(shot.id)
        self.assertEqual(saved.h3_prompt_skill_id, "papercraft-stop-motion-explainer")
        self.assertEqual(saved.h3_prompt_skill_version, "0.6.5")
        self.assertIn("integrated_multimodal_description:", saved.h3_prompt_skill_output)
        self.assertEqual(saved.video_prompt, saved.h3_prompt_skill_output)
        self.assertEqual(self.service.compile_h3_prompt(project, saved, []), saved.h3_prompt_skill_output)
        self.assertIn("handcrafted papercraft stop motion", captured["system"])
        self.assertIn("保持白纸颜色", captured["user"])


if __name__ == "__main__":
    unittest.main()
