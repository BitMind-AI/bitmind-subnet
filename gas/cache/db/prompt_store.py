"""PromptStore — CRUD and sampling for the prompts table."""

import time
import uuid
from typing import List, Dict, Optional, Tuple, Any

import bittensor as bt

from gas.cache.types import PromptEntry
from gas.cache.db.connection import ConnectionManager


class PromptStore:
    """Data access for the ``prompts`` table."""

    def __init__(self, db: ConnectionManager):
        self.db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_prompt_entry(
        self,
        content: str,
        content_type: str = "prompt",
        source_media_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> str:
        """Insert a prompt, returning its id (or existing id if duplicate)."""
        import sqlite3

        prompt_id = str(uuid.uuid4())
        created_at = time.time()

        with self.db.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO prompts (id, content, content_type, modality, created_at, source_media_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (prompt_id, content, content_type, modality, created_at, source_media_id),
                )
                conn.commit()
                return prompt_id
            except sqlite3.IntegrityError:
                cursor = conn.execute(
                    "SELECT id FROM prompts WHERE content = ? AND content_type = ?",
                    (content, content_type),
                )
                return cursor.fetchone()[0]

    def get_prompt_by_id(self, prompt_id: str) -> Optional[str]:
        """Return prompt content for a given id, or None."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(
                    "SELECT content FROM prompts WHERE id = ?", (prompt_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            bt.logging.error(f"Error retrieving prompt {prompt_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_prompt_entries(
        self,
        k: int,
        content_type: str = "prompt",
        strategy: str = "random",
        remove: bool = False,
        modality: Optional[str] = None,
        min_prompts_threshold: int = 100,
    ) -> List[PromptEntry]:
        """Sample prompts with different strategies.

        Parameters
        ----------
        k : int
            Number of prompts to sample.
        strategy : str
            One of ``'random'``, ``'least_used'``, ``'oldest'``, ``'newest'``.
        remove : bool
            If True, delete sampled prompts (only if enough remain).
        modality : str or None
            Optional modality filter (``'image'``, ``'video'``, ``'audio'``).
        min_prompts_threshold : int
            Minimum prompts to keep when *remove* is True.
        """
        with self.db.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row

            if modality:
                where_clause = "content_type = ? AND modality = ?"
                count_params = (content_type, modality)
                params = (content_type, modality, k)
            else:
                where_clause = "content_type = ?"
                count_params = (content_type,)
                params = (content_type, k)

            strategy_sql = {
                "random": "ORDER BY RANDOM()",
                "least_used": "ORDER BY used_count ASC, RANDOM()",
                "oldest": "ORDER BY created_at ASC, RANDOM()",
                "newest": "ORDER BY created_at DESC, RANDOM()",
            }
            order = strategy_sql.get(strategy)
            if order is None:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

            rows = conn.execute(
                f"SELECT * FROM prompts WHERE {where_clause} {order} LIMIT ?",
                params,
            ).fetchall()

            entries: List[PromptEntry] = []
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
                        modality=row["modality"] if "modality" in row.keys() else None,
                    )
                )

            # Bump usage counters
            if entries:
                now = time.time()
                ids = [e.id for e in entries]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE prompts SET used_count = used_count + 1, last_used = ? WHERE id IN ({placeholders})",
                    [now] + ids,
                )

            # Optionally delete sampled prompts
            if remove and entries:
                current_count = conn.execute(
                    f"SELECT COUNT(*) FROM prompts WHERE {where_clause}", count_params
                ).fetchone()[0]
                remaining = current_count - len(entries)
                if remaining >= min_prompts_threshold:
                    ids_to_delete = [e.id for e in entries]
                    placeholders = ",".join("?" * len(ids_to_delete))
                    conn.execute(
                        f"DELETE FROM prompts WHERE id IN ({placeholders})", ids_to_delete
                    )
                    bt.logging.debug(
                        f"Deleted {len(ids_to_delete)} sampled prompts ({remaining} remaining "
                        f"for {content_type}/{modality or 'all'})"
                    )

            conn.commit()

        return entries
