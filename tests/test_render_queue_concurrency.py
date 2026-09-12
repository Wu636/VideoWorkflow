from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.video_workflow.config import settings
from src.video_workflow.domain import AssetRole, JobStatus, ProjectBrief, RenderJob, Shot, ShotContinuityMode
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.services.render_queue import RenderQueue
from src.video_workflow.storage import ProjectStore


class RenderQueueConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.previous = (
            settings.PROJECTS_DIR,
            settings.OUTPUT_DIR,
            settings.SEEDANCE_RENDER_CONCURRENCY,
        )
        settings.PROJECTS_DIR = root / "projects"
        settings.OUTPUT_DIR = root / "output"
        self.store = ProjectStore(root / "state.sqlite3")
        self.service = ProjectService(self.store)
        self.project = self.service.create_project(ProjectBrief(title="并发队列", story="测试"))

    async def asyncTearDown(self) -> None:
        (
            settings.PROJECTS_DIR,
            settings.OUTPUT_DIR,
            settings.SEEDANCE_RENDER_CONCURRENCY,
        ) = self.previous
        self.temp.cleanup()

    def _save_job(self, provider: str) -> RenderJob:
        return self.store.save_job(
            RenderJob(
                project_id=self.project.id,
                provider=provider,
                max_attempts=1,
            )
        )

    async def test_seedance_dispatcher_honors_configured_limit(self) -> None:
        jobs = [self._save_job("ark_seedance") for _ in range(5)]
        settings.SEEDANCE_RENDER_CONCURRENCY = 2
        queue = RenderQueue(self.store, self.service)
        active = 0
        max_active = 0
        seen: list[str] = []
        all_started = asyncio.Event()

        async def fake_process(job: RenderJob) -> None:
            nonlocal active, max_active
            seen.append(job.id)
            active += 1
            max_active = max(max_active, active)
            if len(seen) == len(jobs):
                all_started.set()
            await asyncio.sleep(0.04)
            active -= 1

        queue._process = fake_process  # type: ignore[method-assign]
        await queue.start()
        try:
            await asyncio.wait_for(all_started.wait(), timeout=3)
        finally:
            await queue.stop()

        self.assertEqual(set(seen), {job.id for job in jobs})
        self.assertEqual(max_active, 2)
        self.assertTrue(all(job.status == JobStatus.SUBMITTING for job in self.store.list_jobs(self.project.id)))

    async def test_h3_claim_skips_seedance_without_losing_queued_job(self) -> None:
        seedance = self._save_job("ark_seedance")
        h3 = self._save_job("metaso_h3")

        claimed_h3 = self.store.claim_next_job(exclude_provider="ark_seedance")
        self.assertIsNotNone(claimed_h3)
        self.assertEqual(claimed_h3.id, h3.id)  # type: ignore[union-attr]
        self.assertEqual(self.store.get_job(seedance.id).status, JobStatus.QUEUED)  # type: ignore[union-attr]
        claimed_seedance = self.store.claim_next_job(provider="ark_seedance")
        self.assertIsNotNone(claimed_seedance)
        self.assertEqual(claimed_seedance.id, seedance.id)  # type: ignore[union-attr]

    async def test_continuous_seedance_waits_for_previous_running_tail(self) -> None:
        source = self.store.save_shot(Shot(project_id=self.project.id, ordinal=1, title="上一镜"))
        following = self.store.save_shot(
            Shot(
                project_id=self.project.id,
                ordinal=2,
                title="连续镜头",
                continuity_mode=ShotContinuityMode.CONTINUOUS,
            )
        )
        self.store.save_job(
            RenderJob(
                project_id=self.project.id,
                shot_id=source.id,
                provider="ark_seedance",
                status=JobStatus.RUNNING,
            )
        )
        tail_path = Path(self.temp.name) / "tail.png"
        tail_path.write_bytes(b"tail")

        async def publish_tail() -> None:
            await asyncio.sleep(0.05)
            tail = self.service.register_existing_asset(self.project.id, tail_path, AssetRole.LAST_FRAME, "上一镜尾帧")
            refreshed = self.store.get_shot(source.id)
            assert refreshed is not None
            refreshed.last_frame_asset_id = tail.id
            self.store.save_shot(refreshed)

        publisher = asyncio.create_task(publish_tail())
        try:
            resolved = await RenderQueue(self.store, self.service)._resolve_seedance_continuity_asset(
                self.project, following
            )
        finally:
            await publisher
        self.assertEqual(resolved.name, "上一镜尾帧")


if __name__ == "__main__":
    unittest.main()
