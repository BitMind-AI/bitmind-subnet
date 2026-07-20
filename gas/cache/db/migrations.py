"""Schema migrations, tracked via a _schema_migrations table."""

import time
import sqlite3

import bittensor as bt


MIGRATIONS = [
    (
        "add_uploaded_column",
        [
            "ALTER TABLE media ADD COLUMN uploaded BOOLEAN DEFAULT 0",
            "CREATE INDEX IF NOT EXISTS idx_media_uploaded ON media (uploaded)",
            "UPDATE media SET uploaded = 0 WHERE uploaded IS NULL",
        ],
    ),
    (
        "add_perceptual_hash_column",
        [
            "ALTER TABLE media ADD COLUMN perceptual_hash TEXT",
            "CREATE INDEX IF NOT EXISTS idx_media_perceptual_hash ON media (perceptual_hash)",
        ],
    ),
    (
        "add_c2pa_verified_column",
        [
            "ALTER TABLE media ADD COLUMN c2pa_verified BOOLEAN DEFAULT 0",
            "CREATE INDEX IF NOT EXISTS idx_media_c2pa_verified ON media (c2pa_verified)",
        ],
    ),
    (
        "add_c2pa_issuer_column",
        [
            "ALTER TABLE media ADD COLUMN c2pa_issuer TEXT",
        ],
    ),
    (
        "add_gco_model_name_column",
        [
            "ALTER TABLE generator_challenge_outcomes ADD COLUMN model_name TEXT",
        ],
    ),
    (
        "add_task_id_column",
        [
            "ALTER TABLE media ADD COLUMN task_id TEXT",
            "CREATE INDEX IF NOT EXISTS idx_media_task_id ON media (task_id)",
        ],
    ),
    (
        "add_prompts_modality_column",
        [
            "ALTER TABLE prompts ADD COLUMN modality TEXT CHECK (modality IN ('image', 'video', 'audio'))",
        ],
    ),
    (
        "add_clip_embedding_column",
        [
            "ALTER TABLE media ADD COLUMN clip_embedding BLOB",
        ],
    ),
    (
        "add_resolution_tier_columns",
        [
            "ALTER TABLE generator_challenge_outcomes ADD COLUMN requested_resolution TEXT",
            "ALTER TABLE media ADD COLUMN has_audio BOOLEAN",
        ],
    ),
    (
        "add_prompt_spec_columns",
        [
            "ALTER TABLE prompts ADD COLUMN register TEXT",
            "ALTER TABLE prompts ADD COLUMN length_band TEXT",
            "ALTER TABLE prompts ADD COLUMN event_count INTEGER",
            "ALTER TABLE prompts ADD COLUMN scene_json TEXT",
            "ALTER TABLE prompts ADD COLUMN spec_json TEXT",
            "CREATE INDEX IF NOT EXISTS idx_prompts_register ON prompts (register)",
        ],
    ),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending schema migrations.

    Tracks which migrations have been applied in a ``_schema_migrations``
    table so each migration runs exactly once.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection (must support ``.execute`` / ``.commit``).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at REAL NOT NULL
        )
        """
    )

    applied = {row[0] for row in conn.execute("SELECT name FROM _schema_migrations").fetchall()}

    for name, statements in MIGRATIONS:
        if name in applied:
            continue
        try:
            for sql in statements:
                conn.execute(sql)
            conn.execute(
                "INSERT INTO _schema_migrations (name, applied_at) VALUES (?, ?)",
                (name, time.time()),
            )
            conn.commit()
            bt.logging.info(f"Applied migration: {name}")
        except Exception as e:
            err = str(e).lower()
            if "duplicate column name" in err or "already exists" in err:
                # Column already exists from a run before the migration system existed.
                # Mark as applied and move on.
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO _schema_migrations (name, applied_at) VALUES (?, ?)",
                        (name, time.time()),
                    )
                    conn.commit()
                except Exception:
                    pass
                bt.logging.debug(f"Migration {name} already applied (column exists)")
            else:
                bt.logging.error(f"Migration {name} failed: {e}")
