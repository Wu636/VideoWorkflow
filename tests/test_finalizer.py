from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.video_workflow.config import settings
from src.video_workflow.domain import FinalizeRequest, ProjectBrief, Shot
from src.video_workflow.services.finalize import Finalizer
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class FinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = settings.PROJECTS_DIR
        settings.PROJECTS_DIR = self.root / "projects"
        self.store = ProjectStore(self.root / "state.sqlite3")
        self.projects = ProjectService(self.store)

    def tearDown(self) -> None:
        settings.PROJECTS_DIR = self.previous
        self.temp.cleanup()

    def make_clip(self, path: Path, color: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                settings.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=320x192:r=24:d=0.8",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-t",
                "0.8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    def test_two_shot_finalize_and_qc(self) -> None:
        project = self.projects.create_project(
            ProjectBrief(title="final", story="story", width=320, height=192, fps=24, target_duration_seconds=1.0)
        )
        shots = []
        for ordinal, color in enumerate(("red", "blue"), start=1):
            clip = self.root / f"clip-{ordinal}.mp4"
            self.make_clip(clip, color)
            shots.append(
                Shot(
                    project_id=project.id,
                    ordinal=ordinal,
                    duration_seconds=0.5,
                    video_path=str(clip),
                    video_status="completed",
                )
            )
        self.store.replace_shots(project.id, shots)
        delivery = Finalizer(self.store, self.projects).finalize(project.id, FinalizeRequest(preview=True))
        self.assertTrue(Path(delivery.output_path).exists())
        self.assertTrue(delivery.preview_path)
        self.assertTrue(Path(delivery.preview_path or "").exists())
        self.assertTrue(delivery.qc_report["passed"])
        self.assertAlmostEqual(delivery.duration_seconds, 1.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
