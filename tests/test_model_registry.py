from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_workflow.config import settings
from src.video_workflow.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_public_catalog_contains_grsai_and_masks_keys(self) -> None:
        with patch.object(settings, "GRSAI_API_KEY", "grsai-secret-value"):
            registry = ModelRegistry()
            payload = registry.public_payload()
        self.assertIn("grsai", {item["id"] for item in payload["connections"]})
        self.assertIn("gpt-5.6-terra", {item["model_id"] for item in payload["models"]})
        grsai = next(item for item in payload["connections"] if item["id"] == "grsai")
        self.assertTrue(grsai["configured"])
        self.assertNotIn("grsai-secret-value", json.dumps(payload, ensure_ascii=False))

    def test_custom_connection_and_model_are_file_backed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "OUTPUT_DIR", Path(directory)):
            registry = ModelRegistry()
            connection = registry.add_connection({
                "name": "测试中转",
                "base_url": "https://example.test/v1",
                "api_key": "custom-secret",
            })
            registry.add_model(connection.id, {"model_id": "claude-sonnet-5", "capabilities": ["text", "json", "vision"]})
            registry.set_route("storyboard", connection.id, "claude-sonnet-5")
            persisted = json.loads((Path(directory) / "model_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["connections"][0]["api_key"], "custom-secret")
            self.assertEqual(persisted["routes"]["storyboard"]["connection_id"], connection.id)
            self.assertEqual(registry.route_selection("storyboard"), (connection.id, "claude-sonnet-5"))

    def test_builtin_route_save_does_not_duplicate_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "OUTPUT_DIR", Path(directory)), patch.object(settings, "GRSAI_API_KEY", "builtin-secret"):
            registry = ModelRegistry()
            registry.set_route("storyboard", "grsai", "gpt-5.6-terra")
            persisted = json.loads((Path(directory) / "model_registry.json").read_text(encoding="utf-8"))
            self.assertNotIn("builtin-secret", json.dumps(persisted, ensure_ascii=False))
            self.assertEqual(persisted["routes"]["storyboard"]["model_id"], "gpt-5.6-terra")

    def test_special_routes_resolve_capability_specific_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "OUTPUT_DIR", Path(directory)):
            registry = ModelRegistry()
            image_connection, image_model = registry.route_selection("image_generation")
            expected_image_connection = "grsai" if settings.IMAGE_PROVIDER == "grsai" else "ark"
            expected_image_model = settings.GRSAI_IMAGE_MODEL if expected_image_connection == "grsai" else settings.ARK_IMAGE_MODEL
            self.assertEqual(image_connection, expected_image_connection)
            self.assertEqual(image_model, expected_image_model)
            image_route = next(item for item in registry.public_payload()["routes"] if item["id"] == "image_generation")
            self.assertIn(f"{image_connection}::{image_model}", {option["value"] for option in image_route["options"]})

            seedance_connection, seedance_model = registry.route_selection("seedance_video")
            self.assertEqual(seedance_connection, "ark")
            self.assertEqual(seedance_model, settings.SEEDANCE_DEFAULT_MODEL)

            payload = registry.public_payload()
            seedance_route = next(item for item in payload["routes"] if item["id"] == "seedance_video")
            self.assertTrue(all(option["connection_id"] == "ark" for option in seedance_route["options"]))
            h3_route = next(item for item in payload["routes"] if item["id"] == "h3_video")
            self.assertTrue(all(option["connection_id"] in {"metaso_h3", "atlas_h3", "comfyui_h3"} for option in h3_route["options"]))


class ModelRegistryAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_and_test_statuses_are_recorded_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "OUTPUT_DIR", Path(directory)):
            registry = ModelRegistry()
            result = await registry.discover("ark")
            self.assertFalse(result["supported"])
            connection = next(item for item in registry.public_payload()["connections"] if item["id"] == "ark")
            self.assertEqual(connection["last_discovery_status"], "manual")
            self.assertEqual(connection["last_test_status"], "unknown")
