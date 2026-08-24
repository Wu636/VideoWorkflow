from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.video_workflow.domain import Delivery, Shot
from src.video_workflow.server.app import app
from src.video_workflow.server.routers import projects as router
from src.video_workflow.services.finalize import Finalizer
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.storage import ProjectStore


class ProjectApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old = (router.store, router.project_service, router.render_queue, router.finalizer)
        router.store = ProjectStore(root / "api.sqlite3")
        router.project_service = ProjectService(router.store)
        router.render_queue = RenderQueue(router.store, router.project_service)
        router.finalizer = Finalizer(router.store, router.project_service)
        router.project_service.project_dir = lambda project_id: root / "projects" / project_id  # type: ignore[method-assign]
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        router.store, router.project_service, router.render_queue, router.finalizer = self.old
        self.temp.cleanup()

    def test_project_shot_asset_and_public_review_flow(self) -> None:
        response = self.client.post(
            "/api/projects",
            json={"title": "API 项目", "story": "测试完整项目接口", "target_duration_seconds": 10},
        )
        self.assertEqual(response.status_code, 200, response.text)
        project = response.json()
        project_id = project["id"]

        shot = Shot(project_id=project_id, ordinal=1, narrative="镜头叙事", duration_seconds=5)
        response = self.client.post(f"/api/projects/{project_id}/shots", json=shot.model_dump(mode="json"))
        self.assertEqual(response.status_code, 200, response.text)
        shot_payload = response.json()
        shot_payload["dialogue"] = "你好"
        shot_payload.update(
            h3_width=1344,
            h3_height=768,
            h3_steps=6,
            h3_scheduler="karras",
            h3_denoise=0.9,
            h3_seed=20260822,
        )
        response = self.client.put(f"/api/projects/{project_id}/shots/{shot.id}", json=shot_payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["dialogue"], "你好")
        self.assertEqual(response.json()["h3_width"], 1344)
        self.assertEqual(response.json()["h3_steps"], 6)
        self.assertEqual(response.json()["h3_scheduler"], "karras")
        self.assertEqual(response.json()["h3_seed"], 20260822)

        response = self.client.post(
            f"/api/projects/{project_id}/assets",
            data={"role": "character", "name": "人物参考"},
            files={"file": ("character.png", b"not-a-real-png", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        asset = response.json()
        self.assertEqual(asset["role"], "character")

        response = self.client.post(
            f"/api/projects/{project_id}/script/upload",
            files={"file": ("story.txt", "第一幕：角色走进雨中。".encode(), "text/plain")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("角色走进雨中", response.json()["project"]["brief"]["story"])

        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200, response.text)
        secret_fields = [field for group in response.json()["groups"] for field in group["fields"] if field["secret"]]
        self.assertTrue(secret_fields)
        self.assertTrue(all(field["value"] == "" for field in secret_fields))

        shot_payload = self.client.get(f"/api/projects/{project_id}").json()["shots"][0]
        shot_payload["reference_asset_ids"] = [asset["id"]]
        self.assertEqual(
            self.client.put(f"/api/projects/{project_id}/shots/{shot.id}", json=shot_payload).status_code,
            200,
        )

        response = self.client.get(f"/api/review/{project['review_token']}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["assets"][0]["path"], "")
        response = self.client.post(
            f"/api/review/{project['review_token']}",
            json={
                "target_type": "storyboard",
                "target_id": project_id,
                "decision": "approved",
                "comment": "确认",
                "reviewer": "客户",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

        approved_bundle = self.client.get(f"/api/projects/{project_id}").json()
        approved_version = approved_bundle["project"]["storyboard_version"]
        self.assertEqual(approved_bundle["project"]["status"], "storyboard_approved")
        self.assertEqual(approved_bundle["shots"][0]["approval_status"], "approved")

        edited = approved_bundle["shots"][0]
        edited["narrative"] = "客户确认后的新修改"
        response = self.client.put(f"/api/projects/{project_id}/shots/{shot.id}", json=edited)
        self.assertEqual(response.status_code, 200, response.text)
        invalidated = self.client.get(f"/api/projects/{project_id}").json()
        self.assertEqual(invalidated["project"]["status"], "storyboard_draft")
        self.assertGreater(invalidated["project"]["storyboard_version"], approved_version)
        self.assertEqual(invalidated["shots"][0]["approval_status"], "draft")

        response = self.client.delete(f"/api/projects/{project_id}/assets/{asset['id']}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get(f"/api/projects/{project_id}").json()["shots"][0]["reference_asset_ids"], [])

        blank = self.client.post(f"/api/projects/{project_id}/shots/blank", json={})
        self.assertEqual(blank.status_code, 200, blank.text)
        shots = self.client.get(f"/api/projects/{project_id}").json()["shots"]
        response = self.client.post(
            f"/api/projects/{project_id}/storyboard/reorder",
            json={"shot_ids": [shots[1]["id"], shots[0]["id"]]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()[0]["id"], shots[1]["id"])

        bundle = self.client.get(f"/api/projects/{project_id}").json()
        self.assertEqual(len(bundle["shots"]), 2)
        self.assertEqual(len(bundle["assets"]), 1)
        self.assertEqual(bundle["assets"][0]["type"], "document")
        self.assertEqual(len(bundle["reviews"]), 1)

        delivery = router.store.save_delivery(
            Delivery(project_id=project_id, output_path=str(Path(self.temp.name) / "final.mp4"))
        )
        response = self.client.post(
            f"/api/review/{project['review_token']}",
            json={
                "target_type": "delivery",
                "target_id": delivery.id,
                "decision": "approved",
                "comment": "成片确认",
                "reviewer": "客户",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        delivered = self.client.get(f"/api/projects/{project_id}").json()
        self.assertEqual(delivered["project"]["status"], "delivered")
        self.assertEqual(len(delivered["reviews"]), 2)


if __name__ == "__main__":
    unittest.main()
