"""Attach an already-generated provider image after a legacy download failure.

No generation API is called. Dry-run by default; --apply backs up SQLite first.
Only use a result URL verified in the user's provider history or open result tab.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video_workflow.config import settings
from src.video_workflow.generators.image import GrsaiImageGenerator, download_generated_image
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--scene-profile-id", required=True)
    parser.add_argument("--result-url", required=True)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = ProjectStore(settings.DATABASE_PATH)
    service = ProjectService(store)
    _, profile = service.require_scene_profile(args.project_id, args.scene_profile_id)
    if profile.reference_asset_ids or profile.reference_status in {"generating", "downloading"}:
        raise ValueError("目标场景已有图片或进行中的任务，请先核对，恢复脚本未修改数据")
    print(json.dumps({"scene": profile.name, "apply": args.apply, "generation_api_calls": 0}, ensure_ascii=False))
    if not args.apply:
        return
    recovery = Path("outputs/recovery") / ("scene-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    recovery.mkdir(parents=True)
    with sqlite3.connect(settings.DATABASE_PATH) as source, sqlite3.connect(recovery / "before.sqlite3") as target:
        source.backup(target)
    metadata = {"project_id": args.project_id, "scene_profile_id": args.scene_profile_id,
                "result_url": args.result_url, "task_id": args.task_id, "source": "verified provider result", "original_prompt_available": False}
    (recovery / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    run_id = uuid4().hex
    folder = service.project_dir(args.project_id) / "scene_assets" / profile.id / run_id
    GrsaiImageGenerator._save_result_checkpoint(str(folder), {"result_url": args.result_url, "scene_id": 1, "task_state": {"id": args.task_id, "status": "succeeded"}})
    output = await download_generated_image(args.result_url, str(folder))
    _, latest = service.require_scene_profile(args.project_id, profile.id)
    if latest.reference_asset_ids or latest.reference_status in {"generating", "downloading"}:
        raise ValueError(f"下载期间目标场景已更新，图片保存在 {output}，未覆盖绑定")
    service._set_scene_reference_state(args.project_id, profile.id, reference_run_id=run_id)
    asset = service._finish_scene_reference(args.project_id, profile.id, output)
    asset.description = "从已完成的 GRSAI 结果恢复，仅下载和绑定，未重新生图。旧版本未完整保存实际提交的 Prompt；编辑器提供当前档案生成的新 Prompt 草稿。"
    store.save_asset(asset)
    print(json.dumps({"asset_id": asset.id, "path": asset.path, "backup": str(recovery / "before.sqlite3")}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
