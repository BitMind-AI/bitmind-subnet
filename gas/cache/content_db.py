import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import contextlib
import random

import bittensor as bt

from gas.cache.types import PromptEntry, MediaEntry
from gas.types import Modality, MediaType, SOURCE_TYPE_TO_DB_NAME_FIELD, SourceType


class ContentDB:
    """
    SQLite-based database for managing prompts, search queries, and their associated media.
    Replaces the growing JSON files approach with a proper database structure.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the content database.

        Args:
            db_path: Path to SQLite database file (defaults to ~/.cache/sn34/prompts.db)
        """
        if db_path is None:
            db_path = Path("~/.cache/sn34").expanduser() / "prompts.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    @contextlib.contextmanager
    def _get_db_connection(self, max_retries=3, retry_delay=1.0):
        """Context manager for database connections with retry logic."""
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    check_same_thread=False,
                    isolation_level=None 
                )
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                # Set busy timeout
                conn.execute("PRAGMA busy_timeout=30000")
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys=ON")

                yield conn
                conn.close()
                return
            except sqlite3.OperationalError as e:
                if "unable to open database file" in str(e) or "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        bt.logging.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        bt.logging.error(f"Failed to open database after {max_retries} attempts: {e}")
                        raise
                else:
                    raise
            except Exception as e:
                bt.logging.error(f"Database error: {e}")
                raise

    def get_connection(self):
        """Public context manager for DB connections (read-only or read-write)."""
        return self._get_db_connection()

    def _init_database(self):
        """Initialize the database schema."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL CHECK (content_type IN ('prompt', 'search_query')),
                    created_at REAL NOT NULL,
                    used_count INTEGER DEFAULT 0,
                    last_used REAL,
                    source_media_id TEXT,
                    UNIQUE(content, content_type),
                    FOREIGN KEY (source_media_id) REFERENCES media (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT,
                    file_path TEXT NOT NULL UNIQUE,
                    modality TEXT NOT NULL CHECK (modality IN ('image', 'video')),
                    media_type TEXT NOT NULL CHECK (media_type IN ('real', 'synthetic', 'semisynthetic')),
                    
                    -- Source type to distinguish data origins
                    source_type TEXT DEFAULT 'generated' CHECK (source_type IN ('scraper', 'dataset', 'generated')),
                    
                    -- For scraped media
                    download_url TEXT,
                    scraper_name TEXT,
                    
                    -- For dataset media
                    dataset_name TEXT,
                    dataset_source_file TEXT,
                    dataset_index TEXT,
                    
                    -- For generated media
                    model_name TEXT,
                    generation_args TEXT,  -- JSON string for generation parameters when source_type='generated'

                    -- Common fields
                    created_at REAL NOT NULL,
                    mask_path TEXT,
                    timestamp INTEGER,
                    resolution TEXT,  -- JSON string: "[width, height]"
                    file_size INTEGER,  -- in bytes
                    format TEXT,  -- File format (e.g., "PNG", "JPEG", "MP4")
                    
                    FOREIGN KEY (prompt_id) REFERENCES prompts (id)
                )
            """
            )

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_content_type ON prompts (content_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_used_count ON prompts (used_count)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_created_at ON prompts (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_source_media_id ON prompts (source_media_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_prompt_id ON media (prompt_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_modality ON media (modality)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_media_type ON media (media_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_file_path ON media (file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_model_name ON media (model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_source_type ON media (source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_created_at ON media (created_at)")

            # Temporary data hygiene: prune prompts whose source_media_id no longer exists in media table
            # Note: This only checks DB-level association, not filesystem existence.
            try:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM prompts 
                    WHERE source_media_id IS NOT NULL 
                      AND source_media_id NOT IN (SELECT id FROM media)
                    """
                )
                orphan_count = cursor.fetchone()[0] or 0
                if orphan_count:
                    bt.logging.warning(
                        f"Pruning {orphan_count} orphan prompts with missing media references"
                    )
                    conn.execute(
                        """
                        DELETE FROM prompts 
                        WHERE source_media_id IS NOT NULL 
                          AND source_media_id NOT IN (SELECT id FROM media)
                        """
                    )
                    conn.commit()
            except Exception as e:
                bt.logging.error(f"Failed pruning orphan prompts: {e}")

    def add_prompt_entry(
        self,
        content: str,
        content_type: str = "prompt",
        source_media_id: Optional[str] = None,
    ) -> str:
        prompt_id = str(uuid.uuid4())
        created_at = time.time()

        with self._get_db_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO prompts (id, content, content_type, created_at, source_media_id)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        prompt_id,
                        content,
                        content_type,
                        created_at,
                        source_media_id,
                    ),
                )
                conn.commit()
                return prompt_id
            except sqlite3.IntegrityError:
                cursor = conn.execute(
                    """
                    SELECT id FROM prompts WHERE content = ? AND content_type = ?
                """,
                    (content, content_type),
                )
                return cursor.fetchone()[0]

    def _row_to_media_entry(self, row) -> MediaEntry:
        """
        Helper method to convert a database row to a MediaEntry object.
        """
        # Parse resolution from JSON string if it exists
        resolution = None
        if row["resolution"]:
            try:
                resolution_data = json.loads(row["resolution"])
                resolution = tuple(resolution_data)
            except (json.JSONDecodeError, TypeError):
                resolution = None
        
        return MediaEntry(
            id=row["id"],
            prompt_id=row["prompt_id"],
            file_path=row["file_path"],
            modality=Modality(row["modality"]),
            media_type=MediaType(row["media_type"]),
            source_type=SourceType(row["source_type"]),
            model_name=row["model_name"],
            download_url=row["download_url"],
            scraper_name=row["scraper_name"],
            dataset_name=row["dataset_name"],
            dataset_source_file=row["dataset_source_file"],
            dataset_index=row["dataset_index"],
            created_at=row["created_at"],
            generation_args=json.loads(row["generation_args"]) if row["generation_args"] else None,
            resolution=resolution,
            file_size=row["file_size"],
            format=row["format"],
        )

    def add_media_entry(
        self,
        prompt_id: Optional[str],
        file_path: str,
        modality: Modality,
        media_type: MediaType,
        source_type: SourceType = SourceType.GENERATED,
        model_name: Optional[str] = None,
        download_url: Optional[str] = None,
        scraper_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_source_file: Optional[str] = None,
        dataset_index: Optional[str] = None,
        generation_args: Optional[Dict] = None,
        mask_path: Optional[str] = None,
        timestamp: Optional[int] = None,
        resolution: Optional[tuple[int, int]] = None,
        file_size: Optional[int] = None,
        format: Optional[str] = None,
    ) -> str:
        media_id = str(uuid.uuid4())
        created_at = time.time()

        if timestamp is None:
            timestamp = int(created_at)

        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO media (id, prompt_id, file_path, modality, media_type, source_type,
                                 model_name, download_url, scraper_name, dataset_name, 
                                 dataset_source_file, dataset_index, created_at, generation_args, 
                                 mask_path, timestamp, resolution, file_size, format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    media_id,
                    prompt_id,
                    str(file_path),
                    modality.value,
                    media_type.value,
                    source_type.value,
                    model_name,
                    download_url,
                    scraper_name,
                    dataset_name,
                    dataset_source_file,
                    dataset_index,
                    created_at,
                    json.dumps(generation_args) if generation_args else None,
                    mask_path,
                    timestamp,
                    json.dumps(list(resolution)) if resolution else None,
                    file_size,
                    format,
                ),
            )
            conn.commit()

        return media_id

    def get_media_entries(
        self, 
        prompt_id: Optional[str] = None, 
        media_id: Optional[str] = None
    ) -> List[MediaEntry]:
        """
        Get all media entries associated with a prompt.

        Args:
            prompt_id: ID of the prompt

        Returns:
            List of MediaEntry objects
        """
        if prompt_id is not None:
            col = 'prompt_id'
            val = prompt_id
        elif media_id is not None:
            col = 'id'
            val = media_id
        else:
            raise ValueError("Must provide either prompt_id or media_id")

        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
            f"""
                SELECT * FROM media WHERE {col} = ?
            """,
                (val,),
            )

            media_entries = []
            for row in cursor.fetchall():
                media_entries.append(self._row_to_media_entry(row))

            return media_entries

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with various statistics
        """
        with self._get_db_connection() as conn:
            # Count prompts by type
            cursor = conn.execute(
                """
                SELECT content_type, COUNT(*) as count 
                FROM prompts 
                GROUP BY content_type
            """
            )
            prompt_counts = dict(cursor.fetchall())

            # Count media by modality
            cursor = conn.execute(
                """
                SELECT modality, COUNT(*) as count 
                FROM media 
                GROUP BY modality
            """
            )
            media_counts = dict(cursor.fetchall())

            # Total counts
            total_prompts = sum(prompt_counts.values())
            total_media = sum(media_counts.values())

            # Average usage
            cursor = conn.execute(
                """
                SELECT AVG(used_count) as avg_usage 
                FROM prompts
            """
            )
            avg_usage = cursor.fetchone()[0] or 0

            return {
                "total_prompts": total_prompts,
                "total_media": total_media,
                "prompt_counts": prompt_counts,
                "media_counts": media_counts,
                "average_prompt_usage": avg_usage,
                "database_size_mb": self.db_path.stat().st_size / (1024 * 1024),
            }

    def cleanup_old_entries(self, days_old: int = 30, min_usage: int = 0) -> int:
        """
        Clean up old and unused entries.

        Args:
            days_old: Remove entries older than this many days
            min_usage: Remove entries used less than this many times

        Returns:
            Number of entries removed
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        with self._get_db_connection() as conn:
            # First, get IDs of prompts to remove
            cursor = conn.execute(
                """
                SELECT id FROM prompts 
                WHERE created_at < ? AND used_count <= ?
            """,
                (cutoff_time, min_usage),
            )

            prompt_ids_to_remove = [row[0] for row in cursor.fetchall()]

            if prompt_ids_to_remove:
                # Remove associated media first
                placeholders = ",".join("?" * len(prompt_ids_to_remove))
                conn.execute(
                    f"""
                    DELETE FROM media WHERE prompt_id IN ({placeholders})
                """,
                    prompt_ids_to_remove,
                )

                # Remove prompts
                conn.execute(
                    f"""
                    DELETE FROM prompts WHERE id IN ({placeholders})
                """,
                    prompt_ids_to_remove,
                )

                conn.commit()

            return len(prompt_ids_to_remove)

    def get_media_entry_by_file_path(self, file_path: str) -> Optional[MediaEntry]:
        """
        Get a media entry by its file path.

        Args:
            file_path: Path to the media file

        Returns:
            MediaEntry object or None if not found
        """
        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM media WHERE file_path = ?
            """,
                (file_path,),
            )

            row = cursor.fetchone()
            if row:
                return self._row_to_media_entry(row)
            return None

    def delete_media_entry_by_file_path(self, file_path: str) -> bool:
        """
        Delete a media entry by its file path.

        Args:
            file_path: Path to the media file

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM media WHERE file_path = ?
                """,
                    (file_path,),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error deleting media entry for {file_path}: {e}")
            return False

    def get_media_by_model(
        self, model_name: str, modality: Optional[Modality] = None
    ) -> List[MediaEntry]:
        """
        Get all media entries generated by a specific model.

        Args:
            model_name: Name of the model
            modality: Optional modality filter (Modality.IMAGE or Modality.VIDEO)

        Returns:
            List of MediaEntry objects
        """
        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row

            if modality:
                cursor = conn.execute(
                    """
                    SELECT * FROM media 
                    WHERE model_name = ? AND modality = ?
                    ORDER BY created_at DESC
                """,
                    (model_name, modality.value),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM media 
                    WHERE model_name = ?
                    ORDER BY created_at DESC
                """,
                    (model_name,),
                )

            media_entries = []
            for row in cursor.fetchall():
                media_entries.append(self._row_to_media_entry(row))

            return media_entries

    def get_dataset_media_counts(self) -> Dict[str, int]:
        """
        Get counts of dataset media entries (NULL prompt_id) by modality and media type.

        Returns:
            Dictionary with counts for each modality/media_type combination
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT modality, media_type, COUNT(*) as count
                FROM media 
                WHERE prompt_id IS NULL
                GROUP BY modality, media_type
                """
            )

            counts = {}
            for row in cursor.fetchall():
                modality, media_type, count = row
                key = f"{modality}_{media_type}"
                counts[key] = count

            return counts

    def get_source_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get counts of media grouped by source type and source name.

        Returns:
            { 'dataset': {dataset_name: count, ...}, 'scraper': {...}, 'generated': {...} }
        """
        results: Dict[str, Dict[str, int]] = { 'dataset': {}, 'scraper': {}, 'generated': {} }
        with self._get_db_connection() as conn:
            # Dataset counts
            cursor = conn.execute(
                """
                SELECT dataset_name, COUNT(*) AS cnt
                FROM media
                WHERE source_type = 'dataset' AND dataset_name IS NOT NULL
                GROUP BY dataset_name
                ORDER BY cnt DESC
                """
            )
            results['dataset'] = { name: cnt for name, cnt in cursor.fetchall() }

            # Scraper counts
            cursor = conn.execute(
                """
                SELECT scraper_name, COUNT(*) AS cnt
                FROM media
                WHERE source_type = 'scraper' AND scraper_name IS NOT NULL
                GROUP BY scraper_name
                ORDER BY cnt DESC
                """
            )
            results['scraper'] = { name: cnt for name, cnt in cursor.fetchall() }

            # Generated/model counts
            cursor = conn.execute(
                """
                SELECT model_name, COUNT(*) AS cnt
                FROM media
                WHERE source_type = 'generated' AND model_name IS NOT NULL
                GROUP BY model_name
                ORDER BY cnt DESC
                """
            )
            results['generated'] = { name: cnt for name, cnt in cursor.fetchall() }

        return results

    def get_source_count(self, source_type: SourceType, source_name: str) -> int:
        """
        Get count of media items for a particular source.
        """
        col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
        if not col:
            return 0

        with self._get_db_connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM media WHERE source_type = ? AND {col} = ?",
                (source_type.value, source_name),
            )
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def prune_source_media(self, source_type: SourceType, source_name: str, max_count: int, strategy: str = 'oldest') -> int:
        """
        Prune items for a source so that the total count is <= max_count.
        Deletes database rows for the oldest (or random) items.
        Also deletes any prompts whose source_media_id references the pruned media.

        Returns:
            Number of items pruned.
        """
        col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
        if not col:
            return 0

        current = self.get_source_count(source_type, source_name)
        if current <= max_count:
            return 0

        to_remove = current - max_count
        order_clause = 'created_at ASC' if strategy in ('oldest', 'least_used') else 'RANDOM()'

        with self._get_db_connection() as conn:
            # Select ids and file_paths to delete so we can clear FK references from prompts
            cursor = conn.execute(
                f"""
                SELECT id, file_path
                FROM media
                WHERE source_type = ? AND {col} = ?
                ORDER BY {order_clause}
                LIMIT ?
                """,
                (source_type.value, source_name, to_remove),
            )
            rows = cursor.fetchall()
            if not rows:
                return 0

            media_ids = [row[0] for row in rows]
            file_paths = [row[1] for row in rows]

            # Delete prompts that reference these media ids
            placeholders_ids = ",".join(["?"] * len(media_ids))
            conn.execute(
                f"DELETE FROM prompts WHERE source_media_id IN ({placeholders_ids})",
                media_ids,
            )

            # Delete the media rows
            placeholders_files = ",".join(["?"] * len(file_paths))
            conn.execute(
                f"DELETE FROM media WHERE file_path IN ({placeholders_files})",
                file_paths,
            )

            conn.commit()
            return len(rows)

    def sample_prompt_entries(
        self,
        k: int,
        content_type: str = "prompt",
        strategy: str = "random",
        remove: bool = False,
    ) -> List[PromptEntry]:
        """
        Sample prompts with different strategies. If remove=True, increments used_count and last_used.
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row

            if strategy == "random":
                query = (
                    "SELECT * FROM prompts WHERE content_type = ? ORDER BY RANDOM() LIMIT ?"
                )
            elif strategy == "least_used":
                query = (
                    "SELECT * FROM prompts WHERE content_type = ? ORDER BY used_count ASC, RANDOM() LIMIT ?"
                )
            elif strategy == "oldest":
                query = (
                    "SELECT * FROM prompts WHERE content_type = ? ORDER BY created_at ASC, RANDOM() LIMIT ?"
                )
            elif strategy == "newest":
                query = (
                    "SELECT * FROM prompts WHERE content_type = ? ORDER BY created_at DESC, RANDOM() LIMIT ?"
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

            rows = conn.execute(query, (content_type, k)).fetchall()
            entries: List[PromptEntry] = []
            now = time.time()
            for row in rows:
                entries.append(
                    PromptEntry(
                        id=row["id"],
                        content=row["content"],
                        content_type=row["content_type"],
                        created_at=row["created_at"],
                        used_count=row["used_count"],
                        last_used=row["last_used"],
                        source_media_id=row["source_media_id"],
                    )
                )
                if remove:
                    conn.execute(
                        "UPDATE prompts SET used_count = used_count + 1, last_used = ? WHERE id = ?",
                        (now, row["id"]),
                    )
            conn.commit()
            return entries

    def sample_media_entries(
        self,
        k: int,
        modality: Modality,
        media_type: MediaType,
        strategy: str = "random",
        remove: bool = False,
    ) -> List[MediaEntry]:
        """
        Sample media with strategies. If remove=True and media has a prompt_id, increments the prompt used_count.
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row

            def rows_to_entries(rows: List[sqlite3.Row]) -> List[MediaEntry]:
                items: List[MediaEntry] = []
                now = time.time()
                for row in rows:
                    items.append(self._row_to_media_entry(row))
                    if remove and row["prompt_id"]:
                        conn.execute(
                            "UPDATE prompts SET used_count = used_count + 1, last_used = ? WHERE id = ?",
                            (now, row["prompt_id"]),
                        )
                conn.commit()
                return items

            if strategy == "random":
                query = (
                    """
                    SELECT m.*
                    FROM media m
                    LEFT JOIN prompts p ON m.prompt_id = p.id
                    WHERE m.modality = ? AND m.media_type = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                    """
                )
                rows = conn.execute(query, (modality.value, media_type.value, k)).fetchall()
                return rows_to_entries(rows)

            if strategy == "least_used":
                query = (
                    """
                    SELECT m.*
                    FROM media m
                    LEFT JOIN prompts p ON m.prompt_id = p.id
                    WHERE m.modality = ? AND m.media_type = ?
                    ORDER BY COALESCE(p.used_count, 0) ASC, RANDOM()
                    LIMIT ?
                    """
                )
                rows = conn.execute(query, (modality.value, media_type.value, k)).fetchall()
                return rows_to_entries(rows)

            if strategy == "oldest":
                query = (
                    """
                    SELECT m.*
                    FROM media m
                    LEFT JOIN prompts p ON m.prompt_id = p.id
                    WHERE m.modality = ? AND m.media_type = ?
                    ORDER BY m.created_at ASC, RANDOM()
                    LIMIT ?
                    """
                )
                rows = conn.execute(query, (modality.value, media_type.value, k)).fetchall()
                return rows_to_entries(rows)

            if strategy == "newest":
                query = (
                    """
                    SELECT m.*
                    FROM media m
                    LEFT JOIN prompts p ON m.prompt_id = p.id
                    WHERE m.modality = ? AND m.media_type = ?
                    ORDER BY m.created_at DESC, RANDOM()
                    LIMIT ?
                    """
                )
                rows = conn.execute(query, (modality.value, media_type.value, k)).fetchall()
                return rows_to_entries(rows)

            if strategy == "random_source":
                items: List[MediaEntry] = []
                for _ in range(k):
                    sources = conn.execute(
                        """
                        SELECT DISTINCT 
                            source_type,
                            CASE 
                                WHEN source_type = 'generated' THEN model_name
                                WHEN source_type = 'dataset' THEN dataset_name
                                WHEN source_type = 'scraper' THEN scraper_name
                                ELSE NULL
                            END as source_name
                        FROM media 
                        WHERE modality = ? AND media_type = ? 
                            AND (
                                (source_type = 'generated' AND model_name IS NOT NULL) OR
                                (source_type = 'dataset' AND dataset_name IS NOT NULL) OR
                                (source_type = 'scraper' AND scraper_name IS NOT NULL)
                            )
                        """,
                        (modality.value, media_type.value),
                    ).fetchall()

                    if not sources:
                        row = conn.execute(
                            """
                            SELECT m.* 
                            FROM media m
                            LEFT JOIN prompts p ON m.prompt_id = p.id
                            WHERE m.modality = ? AND m.media_type = ?
                            ORDER BY RANDOM() 
                            LIMIT 1
                            """,
                            (modality.value, media_type.value),
                        ).fetchone()
                        if row:
                            items.append(self._row_to_media_entry(row))
                            if remove and row["prompt_id"]:
                                conn.execute(
                                    "UPDATE prompts SET used_count = used_count + 1, last_used = ? WHERE id = ?",
                                    (time.time(), row["prompt_id"]),
                                )
                        continue

                    source_type, source_name = random.choice(sources)
                    source_column = SOURCE_TYPE_TO_DB_NAME_FIELD.get(SourceType(source_type))
                    if not source_column or not source_name:
                        continue

                    row = conn.execute(
                        f"""
                        SELECT m.* 
                        FROM media m
                        LEFT JOIN prompts p ON m.prompt_id = p.id
                        WHERE m.modality = ? AND m.media_type = ? AND m.source_type = ? AND m.{source_column} = ?
                        ORDER BY RANDOM() 
                        LIMIT 1
                        """,
                        (modality.value, media_type.value, source_type, source_name),
                    ).fetchone()

                    if row:
                        items.append(self._row_to_media_entry(row))
                        if remove and row["prompt_id"]:
                            conn.execute(
                                "UPDATE prompts SET used_count = used_count + 1, last_used = ? WHERE id = ?",
                                (time.time(), row["prompt_id"]),
                            )
                conn.commit()
                return items

            raise ValueError(f"Unknown sampling strategy: {strategy}")

    
