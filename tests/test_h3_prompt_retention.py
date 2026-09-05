from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.restore_h3_snapshot import extract_prompts
from src.video_workflow.config import settings
from src.video_workflow.domain import CharacterProfile, ProjectBrief, Shot
from src.video_workflow.services.projects import H3_DIRECTOR_VERSION, ProjectService
from src.video_workflow.storage import ProjectStore


class H3PromptRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.projects_patch = patch.object(settings, "PROJECTS_DIR", self.root / "projects")
        self.projects_patch.start()
        self.store = ProjectStore(self.root / "db.sqlite3")
        self.service = ProjectService(self.store)
        self.project = self.service.create_project(ProjectBrief(title="保留生成原文", story="人物开门"))
        self.shot = self.store.save_shot(Shot(
            project_id=self.project.id, ordinal=1, narrative="人物开门",
            video_prompt="Saved English H3", video_prompt_source="Saved English H3",
            h3_prompt_skill_output="Saved English H3", h3_director_version=H3_DIRECTOR_VERSION,
            h3_prompt_skill_version="saved-skill", h3_prompt_source_revision=1,
        ))

    def tearDown(self) -> None:
        self.projects_patch.stop()
        self.temp.cleanup()

    def assert_retained(self, shot: Shot) -> None:
        self.assertEqual(shot.h3_prompt_skill_output, "Saved English H3")
        self.assertEqual(shot.video_prompt, "Saved English H3")
        self.assertEqual(shot.video_prompt_source, "Saved English H3")
        self.assertEqual(shot.h3_prompt_skill_version, "saved-skill")
        self.assertEqual(shot.h3_prompt_source_revision, 1)
        self.assertGreater(shot.content_revision, shot.h3_prompt_source_revision)

    def test_project_style_change_marks_review_without_erasing_paid_text(self) -> None:
        self.project.brief.visual_style = "新的动画风格"
        self.service.update_project(self.project)
        self.assert_retained(self.service.require_shot(self.shot.id))

    def test_shot_content_change_keeps_paid_text_and_its_source_revision(self) -> None:
        incoming = self.service.require_shot(self.shot.id)
        incoming.narrative = "人物关门"
        self.service.update_shot(self.project.id, incoming)
        self.assert_retained(self.service.require_shot(self.shot.id))

    def test_legacy_cast_sync_during_bundle_read_keeps_paid_text(self) -> None:
        person = CharacterProfile(name="李强")
        self.project.characters = [person]
        self.store.save_project(self.project)
        self.shot.narrative = "李强走入房间"
        self.store.save_shot(self.shot)
        synced = self.service.synchronize_storyboard_casts(self.project.id)[0]
        self.assertIn(person.id, synced.character_ids)
        self.assert_retained(synced)
        revision = synced.content_revision
        again = self.service.synchronize_storyboard_casts(self.project.id)[0]
        self.assertEqual(again.content_revision, revision)
        self.assert_retained(again)

    def test_manual_h3_edit_survives_plan_and_preserves_previous_history(self) -> None:
        incoming = self.service.require_shot(self.shot.id)
        incoming.video_prompt = "User edited English H3"
        self.service.update_shot(self.project.id, incoming)
        planned = self.service.plan_shot(self.project.id, self.shot.id)
        self.assertEqual(planned.video_prompt, "User edited English H3")
        self.assertEqual(planned.h3_prompt_skill_output, "User edited English H3")
        history = self.store.list_h3_prompt_history(self.project.id, self.shot.id)
        self.assertEqual([entry["prompt"] for entry in history], ["User edited English H3", "Saved English H3"])

    def test_history_survives_accidental_clear_and_storyboard_replacement(self) -> None:
        self.shot.h3_prompt_skill_output = self.shot.video_prompt = ""
        self.store.save_shot(self.shot)
        self.store.replace_shots(self.project.id, [])
        reopened = ProjectStore(self.root / "db.sqlite3")
        history = reopened.list_h3_prompt_history(self.project.id, self.shot.id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["prompt"], "Saved English H3")

    def test_non_prompt_updates_do_not_duplicate_history(self) -> None:
        for _ in range(3):
            self.shot.version += 1
            self.store.save_shot(self.shot)
        self.assertEqual(len(self.store.list_h3_prompt_history(self.project.id, self.shot.id)), 1)

    def test_recovery_extracts_multiline_text_without_next_field(self) -> None:
        snapshot = (
            "\t1 text # 1 镜头 1 · 开门\n"
            "\t2 text h3-prompt-writing  · vsaved-skill\n"
            "\t3 text entry area (settable) MiniMax H3 Prompt, Value: subject_definitions:\n"
            "<Picture 1> is a character.\n\nnon_diegetic_music:\nNone.\n"
            "\t4 text 首帧与本镜 H3 Prompt\n"
        )
        recovered = extract_prompts(snapshot)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["title"], "镜头 1 · 开门")
        self.assertEqual(recovered[0]["skill_version"], "saved-skill")
        self.assertTrue(recovered[0]["prompt"].endswith("None."))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            extract_prompts(snapshot.replace("None.", "tokens truncated"))
