from __future__ import annotations

from pathlib import Path

from src.video_workflow.config import settings


def resolve_media_path(value: str | Path) -> Path:
    """Resolve persisted media paths across host and Docker workspace roots."""
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()

    parts = path.parts
    candidates: list[Path] = []

    if "projects" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("projects")
        candidates.append(Path(settings.PROJECTS_DIR).resolve().joinpath(*parts[index + 1 :]))

    if "outputs" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("outputs")
        candidates.append(Path(settings.OUTPUT_DIR).resolve().joinpath(*parts[index + 1 :]))

    return next((candidate for candidate in candidates if candidate.exists()), path)
