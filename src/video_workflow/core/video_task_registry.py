import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _session_registry_path(session_dir: Path) -> Path:
    return session_dir / "video_tasks.json"


def _global_index_path(output_dir: Path) -> Path:
    return output_dir / "video_task_index.json"


def register_video_task(
    output_dir: Path,
    session_dir: Path,
    task_id: str,
    scene_id: int,
    provider: str = "grsai",
) -> None:
    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        return

    now = __import__("time").time()

    session_registry_path = _session_registry_path(session_dir)
    session_registry = _read_json(session_registry_path, {"tasks": {}})
    tasks = session_registry.get("tasks") if isinstance(session_registry.get("tasks"), dict) else {}

    existing = tasks.get(normalized_task_id, {})
    tasks[normalized_task_id] = {
        "task_id": normalized_task_id,
        "scene_id": scene_id,
        "provider": provider,
        "status": existing.get("status", "submitted"),
        "progress": existing.get("progress", 0),
        "url": existing.get("url", ""),
        "fail_reason": existing.get("fail_reason", ""),
        "updated_at": now,
    }
    session_registry["tasks"] = tasks
    _write_json(session_registry_path, session_registry)

    global_index_path = _global_index_path(output_dir)
    global_index = _read_json(global_index_path, {})
    global_index[normalized_task_id] = {
        "session_id": session_dir.name,
        "scene_id": scene_id,
        "provider": provider,
        "updated_at": now,
    }
    _write_json(global_index_path, global_index)


def update_video_task(
    session_dir: Path,
    task_id: str,
    status: str | None = None,
    progress: int | None = None,
    url: str | None = None,
    fail_reason: str | None = None,
) -> dict[str, Any] | None:
    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        return None

    session_registry_path = _session_registry_path(session_dir)
    session_registry = _read_json(session_registry_path, {"tasks": {}})
    tasks = session_registry.get("tasks") if isinstance(session_registry.get("tasks"), dict) else {}

    task_entry = tasks.get(normalized_task_id)
    if not isinstance(task_entry, dict):
        return None

    if status is not None:
        task_entry["status"] = status
    if progress is not None:
        task_entry["progress"] = progress
    if url is not None:
        task_entry["url"] = url
    if fail_reason is not None:
        task_entry["fail_reason"] = fail_reason

    task_entry["updated_at"] = __import__("time").time()
    tasks[normalized_task_id] = task_entry
    session_registry["tasks"] = tasks
    _write_json(session_registry_path, session_registry)
    return task_entry


def find_video_task(output_dir: Path, task_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        return None, None

    global_index = _read_json(_global_index_path(output_dir), {})
    index_entry = global_index.get(normalized_task_id) if isinstance(global_index, dict) else None
    if isinstance(index_entry, dict):
        session_id = index_entry.get("session_id")
        if isinstance(session_id, str) and session_id:
            session_dir = output_dir / session_id
            session_registry = _read_json(_session_registry_path(session_dir), {"tasks": {}})
            task_entry = (
                session_registry.get("tasks", {}).get(normalized_task_id)
                if isinstance(session_registry.get("tasks"), dict)
                else None
            )
            if isinstance(task_entry, dict):
                return session_dir, task_entry

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        session_registry_path = _session_registry_path(child)
        if not session_registry_path.exists():
            continue

        session_registry = _read_json(session_registry_path, {"tasks": {}})
        task_entry = (
            session_registry.get("tasks", {}).get(normalized_task_id)
            if isinstance(session_registry.get("tasks"), dict)
            else None
        )
        if isinstance(task_entry, dict):
            return child, task_entry

    return None, None
