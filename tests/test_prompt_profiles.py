from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.video_workflow.config import settings
from src.video_workflow.domain import ProjectBrief, Shot
from src.video_workflow.prompt_profiles import (
    PromptProfileContent,
    PromptTemplateVersion,
    render_prompt_template,
    system_default_profile,
    validate_template_variables,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


class PromptProfileTests(unittest.TestCase):
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

    def test_default_profile_is_renderable_and_rejects_unknown_variables(self) -> None:
        profile = system_default_profile()
        self.assertEqual(validate_template_variables(profile.content), [])
        rendered = render_prompt_template(
            profile.content.context_template,
            {"title": "测试剧", "story": "人物发现异常", "shot_count": 3},
        )
        self.assertIn("测试剧", rendered)
        self.assertIn("人物发现异常", rendered)
        invalid = PromptProfileContent(context_template="故事：{{not_allowed}}")
        self.assertEqual(validate_template_variables(invalid), ["not_allowed"])

    def test_template_versions_are_stored_and_latest_is_listed(self) -> None:
        first = PromptTemplateVersion(
            id="mystery",
            name="悬疑模板",
            source="user",
            content=PromptProfileContent(director_template="版本一"),
        )
        second = first.model_copy(
            deep=True,
            update={
                "version": 2,
                "content": PromptProfileContent(director_template="版本二"),
            },
        )
        self.store.save_prompt_template(first)
        self.store.save_prompt_template(second)
        self.assertEqual(self.store.get_prompt_template("mystery").version, 2)
        self.assertEqual(self.store.get_prompt_template("mystery", 1).content.director_template, "版本一")
        listed = self.store.list_prompt_templates()
        self.assertEqual([(item.id, item.version) for item in listed], [("mystery", 2)])

    def test_project_uses_applied_profile_for_downstream_prompts_and_can_reset(self) -> None:
        project = self.service.create_project(ProjectBrief(title="测试剧", story="人物推门"))
        custom = PromptTemplateVersion(
            id="custom-visual",
            name="低成本悬疑",
            source="user",
            content=PromptProfileContent(
                director_template="只写现实主义悬疑动作。",
                downstream_rules={"visual": "CUSTOM_VISUAL_RULE"},
            ),
        )
        applied = self.service.apply_prompt_profile(project.id, custom)
        self.assertEqual(applied.prompt_template_id, "custom-visual")
        self.assertEqual(applied.prompt_template_source, "user")
        self.assertIn("CUSTOM_VISUAL_RULE", self.service.compile_visual_prompt(applied, "人物推门"))

        reset = self.service.reset_prompt_profile(project.id)
        self.assertEqual(reset.prompt_template_id, "system-default")
        self.assertEqual(reset.prompt_template_source, "system")
        self.assertNotIn("CUSTOM_VISUAL_RULE", self.service.compile_visual_prompt(reset, "人物推门"))

    def test_custom_profile_is_inherited_by_series_episodes(self) -> None:
        project = self.service.create_project(ProjectBrief(title="系列首集", story="人物发现异常"))
        custom = PromptTemplateVersion(
            id="series-template",
            name="系列悬疑",
            source="user",
            content=PromptProfileContent(director_template="系列固定导演方法"),
        )
        self.service.apply_prompt_profile(project.id, custom)
        series, _ = self.service.save_project_as_series(project.id, "悬疑系列")
        episode = self.service.create_series_episode(
            series.id,
            ProjectBrief(title="系列第二集", story="人物追踪线索"),
            [],
        )
        self.assertEqual(episode.prompt_template_id, "series-template")
        self.assertEqual(episode.prompt_template_version, 1)
        self.assertEqual(episode.prompt_template_source, "series")
        self.assertEqual(
            self.service.prompt_profile_for_project(episode).content.director_template,
            "系列固定导演方法",
        )

    def test_seedance_generation_accepts_one_time_prompt_suggestions(self) -> None:
        project = self.service.create_project(ProjectBrief(title="测试剧", story="人物推门"))
        self.store.save_shot(Shot(project_id=project.id, ordinal=1, title="开场", narrative="人物推门"))
        saved = self.service.generate_seedance_prompts(
            project.id,
            input_mode="frame",
            user_suggestions="动作克制，结尾停在门缝出现异常光线",
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            saved[0].seedance_prompt_user_constraints,
            "动作克制，结尾停在门缝出现异常光线",
        )
        self.assertIn("动作克制", saved[0].seedance_prompt)


if __name__ == "__main__":
    unittest.main()
