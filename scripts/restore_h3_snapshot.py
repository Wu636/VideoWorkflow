"""Restore H3 text from a complete recorded UI snapshot; dry-run by default.

Only prompt fields are restored. Current images, references and shot content
are preserved. A database backup and extracted text are saved before applying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.video_workflow.domain import Shot
from src.video_workflow.services.projects import H3_DIRECTOR_VERSION
from src.video_workflow.storage import ProjectStore


def extract_prompts(snapshot: str) -> list[dict]:
    headers = list(re.finditer(r"^\t+\d+ text # (\d+) (.+)$", snapshot, re.MULTILINE))
    fields = list(re.finditer(
        r"^\t+\d+ text entry area[^\n]*MiniMax H3 Prompt[^\n]*, Value: ([\s\S]*?)(?=\n\t+\d+ )",
        snapshot, re.MULTILINE,
    ))
    prompts = []
    for field in fields:
        header = next((header for header in reversed(headers) if header.start() < field.start()), None)
        if header is None:
            raise ValueError("Missing shot header")
        text = field.group(1).strip()
        if "tokens truncated" in text or "non_diegetic_music:" not in text:
            raise ValueError("Incomplete prompt snapshot")
        skill = re.search(r"text ([\w-]+)\s*· v([\w.-]+)", snapshot[header.end():field.start()])
        if skill is None:
            raise ValueError("Missing skill provenance")
        prompts.append({
            "ordinal": int(header.group(1)), "title": header.group(2), "prompt": text,
            "skill_id": skill.group(1), "skill_version": skill.group(2),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    if not prompts or len({p["ordinal"] for p in prompts}) != len(prompts):
        raise ValueError("Missing or duplicate prompt entries")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_log", type=Path)
    parser.add_argument("--event-line", type=int, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-revision", type=int, required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with args.snapshot_log.open() as handle:
        record = json.loads(next(line for index, line in enumerate(handle, 1) if index == args.event_line))
    blocks = record["payload"]["item"]["result"]["content"]
    snapshot = next(block["text"] for block in blocks if block.get("type") == "text" and "Browser tab:" in block["text"])
    if f"/projects/{args.project_id}" not in snapshot.splitlines()[0]:
        raise ValueError("Snapshot belongs to a different project")
    prompts = extract_prompts(snapshot)
    with sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True) as conn:
        current = [Shot.model_validate_json(row[0]) for row in conn.execute(
            "SELECT data FROM shots WHERE project_id=? ORDER BY ordinal", (args.project_id,),
        )]
    if len(prompts) != len(current):
        raise ValueError("Snapshot shot count differs from current storyboard")
    for prompt, shot in zip(prompts, current, strict=True):
        if prompt["ordinal"] != shot.ordinal or prompt["title"] != shot.title:
            raise ValueError(f"Shot identity/title mismatch: {shot.ordinal}")
        if shot.content_revision != args.expected_revision or shot.h3_prompt_skill_output:
            raise ValueError(f"Shot {shot.ordinal} has newer content or an existing H3 prompt; leave it untouched")
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "project_id": args.project_id,
                      "prompts": [{"ordinal": p["ordinal"], "chars": len(p["prompt"]), "sha256": p["sha256"]} for p in prompts]}, ensure_ascii=False))
    if not args.apply:
        return
    destination = args.database.parent / "recovery" / ("h3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    destination.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True) as source, sqlite3.connect(destination / "before.sqlite3") as backup:
        source.backup(backup)
    (destination / "recovered-prompts.json").write_text(json.dumps({
        "project_id": args.project_id, "source_revision": args.source_revision,
        "snapshot_timestamp": record["timestamp"], "prompts": prompts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    store = ProjectStore(args.database)
    # Recheck all rows under a write transaction before restoring any of them.
    with store._lock, store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for before, prompt in zip(current, prompts, strict=True):
            row = conn.execute("SELECT data FROM shots WHERE id=?", (before.id,)).fetchone()
            latest = Shot.model_validate_json(row["data"])
            if latest.model_dump() != before.model_dump():
                raise ValueError(f"Shot {before.ordinal} changed during recovery; aborted")
            latest.h3_prompt_skill_output = prompt["prompt"]
            latest.video_prompt = prompt["prompt"]
            latest.video_prompt_source = prompt["prompt"]
            latest.h3_prompt_skill_id = prompt["skill_id"]
            latest.h3_prompt_skill_version = prompt["skill_version"]
            latest.h3_director_version = H3_DIRECTOR_VERSION
            latest.h3_prompt_source_revision = args.source_revision
            latest.version += 1
            latest.updated_at = datetime.now(timezone.utc).isoformat()
            store._archive_h3_prompt(conn, latest)
            conn.execute("UPDATE shots SET updated_at=?,data=? WHERE id=?",
                         (latest.updated_at, store._dump(latest), latest.id))
    print(f"Restored {len(prompts)} prompts; backup and recovered text: {destination.resolve()}")


if __name__ == "__main__":
    main()
