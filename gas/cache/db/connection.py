"""Database connection management for the cache layer."""

import sqlite3
import time
import contextlib
from pathlib import Path
from typing import Iterator

import bittensor as bt


class ConnectionManager:
    """Manages SQLite connections with WAL mode, retry, and timeout configuration."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections with retry logic."""
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False,
                    isolation_level=None,
                )
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("PRAGMA foreign_keys=ON")

                yield conn
                conn.close()
                return
            except sqlite3.OperationalError as e:
                if "unable to open database file" in str(e) or "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        bt.logging.warning(
                            f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    bt.logging.error(f"Failed to open database after {max_retries} attempts: {e}")
                    raise
                raise
            except Exception as e:
                bt.logging.error(f"Database error: {e}")
                raise


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the base tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL CHECK (content_type = 'prompt'),
            modality TEXT CHECK (modality IN ('image', 'video', 'audio')),
            created_at REAL NOT NULL,
            used_count INTEGER DEFAULT 0,
            last_used REAL,
            source_media_id TEXT,
            UNIQUE(content, content_type, modality),
            FOREIGN KEY (source_media_id) REFERENCES media (id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media (
            id TEXT PRIMARY KEY,
            prompt_id TEXT,
            file_path TEXT NOT NULL UNIQUE,
            modality TEXT NOT NULL CHECK (modality IN ('image', 'video')),
            media_type TEXT NOT NULL CHECK (media_type IN ('real', 'synthetic', 'semisynthetic')),
            source_type TEXT DEFAULT 'generated' CHECK (source_type IN ('scraper', 'dataset', 'generated', 'miner')),
            download_url TEXT, scraper_name TEXT,
            dataset_name TEXT, dataset_source_file TEXT, dataset_index TEXT,
            model_name TEXT, generation_args TEXT,
            uid INTEGER, hotkey TEXT,
            verified BOOLEAN DEFAULT 0, failed_verification BOOLEAN DEFAULT 0, rewarded BOOLEAN DEFAULT 0,
            created_at REAL NOT NULL, mask_path TEXT, timestamp INTEGER,
            resolution TEXT, file_size INTEGER, format TEXT,
            FOREIGN KEY (prompt_id) REFERENCES prompts (id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generator_challenge_outcomes (
            task_id TEXT PRIMARY KEY,
            uid INTEGER NOT NULL,
            hotkey TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            modality TEXT NOT NULL CHECK (modality IN ('image', 'video', 'audio')),
            status TEXT NOT NULL CHECK (status IN ('pending', 'stored', 'verified', 'failed')),
            failure_reason TEXT,
            media_id TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES prompts (id),
            FOREIGN KEY (media_id) REFERENCES media (id)
        )
    """)
    for tbl, cols in [
        ("prompts", ["content_type", "used_count", "created_at", "source_media_id"]),
        ("media", ["prompt_id", "modality", "media_type", "file_path",
                   "model_name", "source_type", "created_at", "uid",
                   "hotkey", "verified", "failed_verification", "rewarded"]),
        ("generator_challenge_outcomes", ["hotkey", "uid", "status", "updated_at", "media_id"]),
    ]:
        for col in cols:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_{col} ON {tbl} ({col})")

    from gas.cache.db.migrations import run_migrations
    run_migrations(conn)
