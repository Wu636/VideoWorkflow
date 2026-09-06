from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from pydantic import BaseModel

from src.video_workflow.domain import (
    Asset,
    Delivery,
    JobStatus,
    Project,
    ProductionSeries,
    RenderJob,
    Review,
    Shot,
    utc_now,
)

T = TypeVar("T", bound=BaseModel)


class ProjectStore:
    """Small durable repository built on SQLite JSON records.

    The JSON payload keeps the domain model easy to evolve while indexed columns
    provide the project/job queries used by the UI and render worker.
    """

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS production_series (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shots (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_shots_project ON shots(project_id, ordinal);
        CREATE TABLE IF NOT EXISTS h3_prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            shot_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL,
            UNIQUE(shot_id, fingerprint),
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_h3_history_shot ON h3_prompt_history(project_id, shot_id, id);
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            type TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id, created_at);
        CREATE TABLE IF NOT EXISTS render_jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            shot_id TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON render_jobs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_project ON render_jobs(project_id, created_at);
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS deliveries (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        """
        with self._lock, self._connect() as conn:
            conn.executescript(schema)

    @staticmethod
    def _dump(model: BaseModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)

    @staticmethod
    def _load(row: sqlite3.Row | None, model_type: type[T]) -> T | None:
        if row is None:
            return None
        return model_type.model_validate_json(row["data"])

    def save_project(self, project: Project) -> Project:
        project.updated_at = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO projects(id,status,created_at,updated_at,data)
                VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, updated_at=excluded.updated_at, data=excluded.data""",
                (project.id, project.status.value, project.created_at, project.updated_at, self._dump(project)),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as conn:
            return self._load(conn.execute("SELECT data FROM projects WHERE id=?", (project_id,)).fetchone(), Project)

    def list_projects(self) -> list[Project]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM projects ORDER BY updated_at DESC").fetchall()
        return [Project.model_validate_json(row["data"]) for row in rows]

    def delete_project(self, project_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            return cursor.rowcount > 0

    def save_series(self, series: ProductionSeries) -> ProductionSeries:
        series.updated_at = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO production_series(id,name,created_at,updated_at,data)
                VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,updated_at=excluded.updated_at,data=excluded.data""",
                (series.id, series.name, series.created_at, series.updated_at, self._dump(series)),
            )
        return series

    def get_series(self, series_id: str) -> ProductionSeries | None:
        with self._connect() as conn:
            return self._load(
                conn.execute("SELECT data FROM production_series WHERE id=?", (series_id,)).fetchone(),
                ProductionSeries,
            )

    def list_series(self) -> list[ProductionSeries]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM production_series ORDER BY updated_at DESC").fetchall()
        return [ProductionSeries.model_validate_json(row["data"]) for row in rows]

    def replace_shots(self, project_id: str, shots: Iterable[Shot]) -> list[Shot]:
        items = list(shots)
        with self._lock, self._connect() as conn:
            for row in conn.execute("SELECT data FROM shots WHERE project_id=?", (project_id,)).fetchall():
                self._archive_h3_prompt(conn, Shot.model_validate_json(row["data"]))
            for shot in items:
                self._archive_h3_prompt(conn, shot)
            conn.execute("DELETE FROM shots WHERE project_id=?", (project_id,))
            conn.executemany(
                "INSERT INTO shots(id,project_id,ordinal,updated_at,data) VALUES(?,?,?,?,?)",
                [(s.id, project_id, s.ordinal, s.updated_at, self._dump(s)) for s in items],
            )
        return items

    def save_shot(self, shot: Shot) -> Shot:
        shot.updated_at = utc_now()
        with self._lock, self._connect() as conn:
            existing = self._load(conn.execute("SELECT data FROM shots WHERE id=?", (shot.id,)).fetchone(), Shot)
            if existing is not None:
                self._archive_h3_prompt(conn, existing)
            self._archive_h3_prompt(conn, shot)
            conn.execute(
                """INSERT INTO shots(id,project_id,ordinal,updated_at,data) VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET ordinal=excluded.ordinal,
                updated_at=excluded.updated_at,data=excluded.data""",
                (shot.id, shot.project_id, shot.ordinal, shot.updated_at, self._dump(shot)),
            )
        return shot

    def insert_shot(self, shot: Shot, after_shot_id: str | None = None) -> Shot:
        """Insert one shot and renumber the tail in a single transaction."""
        shot.updated_at = utc_now()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM shots WHERE project_id=? ORDER BY ordinal",
                (shot.project_id,),
            ).fetchall()
            existing = [Shot.model_validate_json(row["data"]) for row in rows]
            if after_shot_id is None:
                ordinal = len(existing) + 1
            else:
                anchor = next((item for item in existing if item.id == after_shot_id), None)
                if anchor is None:
                    raise KeyError(f"Shot not found: {after_shot_id}")
                ordinal = anchor.ordinal + 1
            shot.ordinal = ordinal
            for current in reversed(existing):
                if current.ordinal < ordinal:
                    continue
                self._archive_h3_prompt(conn, current)
                current.ordinal += 1
                current.updated_at = utc_now()
                conn.execute(
                    "UPDATE shots SET ordinal=?,updated_at=?,data=? WHERE id=?",
                    (current.ordinal, current.updated_at, self._dump(current), current.id),
                )
            self._archive_h3_prompt(conn, shot)
            conn.execute(
                "INSERT INTO shots(id,project_id,ordinal,updated_at,data) VALUES(?,?,?,?,?)",
                (shot.id, shot.project_id, shot.ordinal, shot.updated_at, self._dump(shot)),
            )
        return shot

    def replace_shot_with_many(
        self,
        project_id: str,
        shot_id: str,
        replacements: Iterable[Shot],
        expected_version: int | None = None,
    ) -> list[Shot]:
        """Replace one shot with several fresh shots in a single transaction."""
        new_items = list(replacements)
        if len(new_items) < 2:
            raise ValueError("拆分结果至少需要两个镜头")
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM shots WHERE project_id=? ORDER BY ordinal",
                (project_id,),
            ).fetchall()
            existing = [Shot.model_validate_json(row["data"]) for row in rows]
            source_index = next((index for index, item in enumerate(existing) if item.id == shot_id), None)
            if source_index is None:
                raise KeyError(f"Shot not found: {shot_id}")
            source = existing[source_index]
            if expected_version is not None and source.version != expected_version:
                raise ValueError("本镜已有新的保存结果，请重新生成拆分预览")
            existing_ids = {item.id for item in existing if item.id != shot_id}
            replacement_ids = [item.id for item in new_items]
            if len(set(replacement_ids)) != len(replacement_ids) or existing_ids.intersection(replacement_ids):
                raise ValueError("拆分镜头标识发生冲突，请重新生成拆分预览")

            now = utc_now()
            for offset, item in enumerate(new_items):
                item.project_id = project_id
                item.ordinal = source.ordinal + offset
                item.updated_at = now
            self._archive_h3_prompt(conn, source)
            for item in new_items:
                self._archive_h3_prompt(conn, item)
            conn.execute("DELETE FROM shots WHERE id=? AND project_id=?", (shot_id, project_id))
            conn.executemany(
                "INSERT INTO shots(id,project_id,ordinal,updated_at,data) VALUES(?,?,?,?,?)",
                [(item.id, project_id, item.ordinal, item.updated_at, self._dump(item)) for item in new_items],
            )
            replacement_count_delta = len(new_items) - 1
            for item in existing[source_index + 1:]:
                self._archive_h3_prompt(conn, item)
                item.ordinal += replacement_count_delta
                item.updated_at = now
                conn.execute(
                    "UPDATE shots SET ordinal=?,updated_at=?,data=? WHERE id=? AND project_id=?",
                    (item.ordinal, item.updated_at, self._dump(item), item.id, project_id),
                )
        return new_items

    @staticmethod
    def _archive_h3_prompt(conn: sqlite3.Connection, shot: Shot) -> None:
        if not shot.h3_prompt_skill_output.strip():
            return
        payload = {
            "shot_id": shot.id,
            "ordinal": shot.ordinal,
            "prompt": shot.h3_prompt_skill_output,
            "video_prompt": shot.video_prompt,
            "source_revision": shot.h3_prompt_source_revision,
            "skill_id": shot.h3_prompt_skill_id,
            "skill_version": shot.h3_prompt_skill_version,
            "director_version": shot.h3_director_version,
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(data.encode("utf-8")).hexdigest()
        conn.execute(
            """INSERT OR IGNORE INTO h3_prompt_history(project_id,shot_id,fingerprint,created_at,data)
            VALUES(?,?,?,?,?)""",
            (shot.project_id, shot.id, fingerprint, utc_now(), data),
        )

    def archive_h3_prompt(self, shot: Shot) -> None:
        with self._lock, self._connect() as conn:
            self._archive_h3_prompt(conn, shot)

    def list_h3_prompt_history(self, project_id: str, shot_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,created_at,data FROM h3_prompt_history WHERE project_id=? AND shot_id=? ORDER BY id DESC",
                (project_id, shot_id),
            ).fetchall()
        return [{**json.loads(row["data"]), "id": row["id"], "created_at": row["created_at"]} for row in rows]

    def get_shot(self, shot_id: str) -> Shot | None:
        with self._connect() as conn:
            return self._load(conn.execute("SELECT data FROM shots WHERE id=?", (shot_id,)).fetchone(), Shot)

    def list_shots(self, project_id: str) -> list[Shot]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM shots WHERE project_id=? ORDER BY ordinal", (project_id,)).fetchall()
        return [Shot.model_validate_json(row["data"]) for row in rows]

    def save_asset(self, asset: Asset) -> Asset:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO assets(id,project_id,type,role,created_at,data) VALUES(?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET type=excluded.type,role=excluded.role,data=excluded.data""",
                (asset.id, asset.project_id, asset.type.value, asset.role.value, asset.created_at, self._dump(asset)),
            )
        return asset

    def get_asset(self, asset_id: str) -> Asset | None:
        with self._connect() as conn:
            return self._load(conn.execute("SELECT data FROM assets WHERE id=?", (asset_id,)).fetchone(), Asset)

    def list_assets(self, project_id: str) -> list[Asset]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM assets WHERE project_id=? ORDER BY created_at", (project_id,)).fetchall()
        return [Asset.model_validate_json(row["data"]) for row in rows]

    def delete_asset(self, asset_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM assets WHERE id=?", (asset_id,))
            return cursor.rowcount > 0

    def delete_job(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM render_jobs WHERE id=?", (job_id,))
            return cursor.rowcount > 0

    def save_job(self, job: RenderJob) -> RenderJob:
        job.updated_at = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO render_jobs(id,project_id,shot_id,status,created_at,updated_at,data)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,updated_at=excluded.updated_at,data=excluded.data""",
                (job.id, job.project_id, job.shot_id, job.status.value, job.created_at, job.updated_at, self._dump(job)),
            )
        return job

    def get_job(self, job_id: str) -> RenderJob | None:
        with self._connect() as conn:
            return self._load(conn.execute("SELECT data FROM render_jobs WHERE id=?", (job_id,)).fetchone(), RenderJob)

    def list_jobs(self, project_id: str | None = None) -> list[RenderJob]:
        query = "SELECT data FROM render_jobs"
        params: tuple[str, ...] = ()
        if project_id:
            query += " WHERE project_id=?"
            params = (project_id,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RenderJob.model_validate_json(row["data"]) for row in rows]

    def claim_next_job(self) -> RenderJob | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT data FROM render_jobs WHERE status=? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            job = RenderJob.model_validate_json(row["data"])
            job.status = JobStatus.SUBMITTING
            job.started_at = job.started_at or utc_now()
            job.attempt += 1
            job.updated_at = utc_now()
            conn.execute(
                "UPDATE render_jobs SET status=?,updated_at=?,data=? WHERE id=?",
                (job.status.value, job.updated_at, self._dump(job), job.id),
            )
            conn.commit()
            return job

    def recover_interrupted_jobs(self) -> int:
        recovered = 0
        for job in self.list_jobs():
            if job.status in {JobStatus.SUBMITTING, JobStatus.RUNNING}:
                job.status = JobStatus.QUEUED
                job.error = "Recovered after backend restart"
                self.save_job(job)
                recovered += 1
        return recovered

    def save_review(self, review: Review) -> Review:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO reviews(id,project_id,target_id,created_at,data) VALUES(?,?,?,?,?)",
                (review.id, review.project_id, review.target_id, review.created_at, self._dump(review)),
            )
        return review

    def list_reviews(self, project_id: str) -> list[Review]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM reviews WHERE project_id=? ORDER BY created_at", (project_id,)).fetchall()
        return [Review.model_validate_json(row["data"]) for row in rows]

    def save_delivery(self, delivery: Delivery) -> Delivery:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO deliveries(id,project_id,created_at,data) VALUES(?,?,?,?)",
                (delivery.id, delivery.project_id, delivery.created_at, self._dump(delivery)),
            )
        return delivery

    def list_deliveries(self, project_id: str) -> list[Delivery]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM deliveries WHERE project_id=? ORDER BY created_at DESC", (project_id,)).fetchall()
        return [Delivery.model_validate_json(row["data"]) for row in rows]
