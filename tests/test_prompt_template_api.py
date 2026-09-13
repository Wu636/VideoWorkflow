from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.video_workflow.config import settings
from src.video_workflow.domain import ProjectBrief
from src.video_workflow.prompt_profiles import PromptProfileContent
from src.video_workflow.server.app import app
from src.video_workflow.server.routers import prompt_templates as prompt_router, projects as project_router
from src.video_workflow.storage import ProjectStore
from src.video_workflow.services.projects import ProjectService


class _IdleRenderQueue:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class PromptTemplateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old = (prompt_router.store, prompt_router.project_service, project_router.render_queue)
        prompt_router.store = ProjectStore(root / "prompt.sqlite3")
        prompt_router.project_service = ProjectService(prompt_router.store)
        prompt_router.project_service.project_dir = lambda project_id: root / "projects" / project_id  # type: ignore[method-assign]
        project_router.render_queue = _IdleRenderQueue()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        prompt_router.store, prompt_router.project_service, project_router.render_queue = self.old
        self.temp.cleanup()

    def test_template_can_be_created_applied_and_reset(self) -> None:
        project = prompt_router.project_service.create_project(
            ProjectBrief(title="模板接口测试", story="人物发现异常")
        )
        listed = self.client.get("/api/prompt-templates")
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json()[0]["id"], "system-default")

        content = PromptProfileContent(
            director_template="CUSTOM_DIRECTOR",
            context_template="剧情：{{story}}；建议：{{run_notes}}",
        )
        created = self.client.post(
            "/api/prompt-templates",
            json={"name": "本剧悬疑", "description": "测试模板", "content": content.model_dump(mode="json")},
        )
        self.assertEqual(created.status_code, 200, created.text)
        template = created.json()
        applied = self.client.put(
            f"/api/prompt-templates/projects/{project.id}/prompt-profile",
            json={"template_id": template["id"], "version": 1},
        )
        self.assertEqual(applied.status_code, 200, applied.text)
        self.assertEqual(applied.json()["prompt_template_id"], template["id"])
        self.assertEqual(applied.json()["prompt_template_source"], "user")

        reset = self.client.post(f"/api/prompt-templates/projects/{project.id}/prompt-profile/reset")
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertEqual(reset.json()["prompt_template_id"], "system-default")

    def test_template_metadata_explains_variables(self) -> None:
        response = self.client.get("/api/prompt-templates/metadata")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        variables = {item["name"]: item for item in payload["variables"]}
        self.assertIn("run_notes", variables)
        self.assertEqual(variables["run_notes"]["label"], "本次生成建议")
        self.assertTrue(variables["story"]["description"])
        self.assertGreaterEqual(len(payload["examples"]), 5)
        self.assertIn("realistic-safety-training", {item["id"] for item in payload["examples"]})
        self.assertIn("series-fixed-rules", {item["id"] for item in payload["examples"]})
        self.assertTrue(payload["machine_contract_note"])

    def test_ai_draft_always_uses_openlux_opus5_setting(self) -> None:
        project = prompt_router.project_service.create_project(
            ProjectBrief(title="优化器测试", story="人物在雨夜发现线索")
        )
        captured: dict[str, object] = {}

        class FakeOpenLux:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def generate_json(self, _system: str, _user: str) -> dict[str, object]:
                return {
                    "name": "雨夜悬疑模板",
                    "description": "测试草稿",
                    "director_template": "加强雨夜悬疑节奏",
                    "context_template": "{{story}}\n{{run_notes}}",
                    "downstream_rules": {
                        "visual": "强化雨幕中的线索",
                        "keyframe": "锁定雨夜首帧",
                        "seedance": "动作克制",
                        "h3": "保留声音时序",
                    },
                    "change_summary": ["强化开场线索"],
                    "expected_effects": ["前3秒更快进入冲突"],
                    "warnings": [],
                    "variables_used": ["story", "run_notes"],
                }

        with patch.object(settings, "PROMPT_OPTIMIZER_MODEL", "claude-opus-5"), patch.object(
            prompt_router, "OpenLuxGenerator", FakeOpenLux
        ):
            response = self.client.post(
                "/api/prompt-templates/ai-draft",
                json={
                    "project_id": project.id,
                    "instruction": "开场更快出现异常线索",
                    "scopes": ["storyboard", "visual", "keyframe", "seedance", "h3"],
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["model"], "claude-opus-5")
        self.assertEqual(captured["vision_model"], "claude-opus-5")
        self.assertEqual(response.json()["draft"]["source"], "ai")


if __name__ == "__main__":
    unittest.main()
