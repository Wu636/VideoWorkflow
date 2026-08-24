from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.video_workflow.config import settings


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)[^\s,;]+"),
)


def redact(value: str) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(r"\1***", result)
    return result


class RuntimeLogHandler(logging.Handler):
    def __init__(self, capacity: int = 3000) -> None:
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.lock_records = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = redact(record.getMessage())
            exception = redact(logging.Formatter().formatException(record.exc_info)) if record.exc_info else ""
            item = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "exception": exception,
            }
            with self.lock_records:
                self.records.append(item)
        except Exception:
            self.handleError(record)

    def query(self, level: str | None = None, search: str = "", limit: int = 500) -> list[dict[str, Any]]:
        threshold = logging._nameToLevel.get((level or "DEBUG").upper(), logging.DEBUG)
        needle = search.casefold().strip()
        with self.lock_records:
            values = list(self.records)
        result = [
            item for item in values
            if logging._nameToLevel.get(item["level"], logging.INFO) >= threshold
            and (not needle or needle in f"{item['logger']} {item['message']} {item['exception']}".casefold())
        ]
        return result[-max(1, min(limit, 3000)):]

    def clear(self) -> None:
        with self.lock_records:
            self.records.clear()


runtime_log_handler = RuntimeLogHandler()
_configured = False


def log_file_path() -> Path:
    return Path(settings.OUTPUT_DIR) / "logs" / "video_workflow.log"


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    path = log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    class RedactingFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return redact(super().format(record))

    formatter = RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(runtime_log_handler)
    root.addHandler(file_handler)
    _configured = True


def clear_logs() -> None:
    runtime_log_handler.clear()
    path = log_file_path()
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path.resolve():
            handler.acquire()
            try:
                handler.flush()
                if handler.stream:
                    handler.stream.seek(0)
                    handler.stream.truncate(0)
            finally:
                handler.release()
