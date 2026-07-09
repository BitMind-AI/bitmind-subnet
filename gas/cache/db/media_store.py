"""MediaStore — CRUD, verification queries, and sampling for the media table."""

import json
import time
import uuid
import random
from typing import List, Dict, Optional, Tuple, Any

import bittensor as bt

from gas.cache.types import MediaEntry
from gas.types import Modality, MediaType, SourceType, SOURCE_TYPE_TO_DB_NAME_FIELD
from gas.cache.db.connection import ConnectionManager
from gas.cache.db.prompt_store import PromptStore
from gas.cache.db.challenge_store import ChallengeStore


class MediaStore:
    """Data access for the ``media`` table."""

    def __init__(self, db: ConnectionManager, prompts: PromptStore, challenges: ChallengeStore):
        self.db = db
        self.prompts = prompts
        self.challenges = challenges

    # ------------------------------------------------------------------
    # Row mapper
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_media_entry(row) -> MediaEntry:
        """Convert a database row (sqlite3.Row or dict-like) to a MediaEntry."""
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
            uid=row["uid"] if "uid" in row.keys() and row["uid"] is not None else None,
            hotkey=row["hotkey"],
            verified=bool(row["verified"]) if "verified" in row.keys() and row["verified"] is not None else False,
            failed_verification=bool(row["failed_verification"]) if "failed_verification" in row.keys() and row["failed_verification"] is not None else False,
            uploaded=bool(row["uploaded"]) if "uploaded" in row.keys() and row["uploaded"] is not None else False,
            rewarded=bool(row["rewarded"]) if "rewarded" in row.keys() and row["rewarded"] is not None else False,
            created_at=row["created_at"],
            task_id=row["task_id"] if "task_id" in row.keys() else None,
            resolution=resolution,
            file_size=row["file_size"],
            format=row["format"],
            mask_path=row["mask_path"] if "mask_path" in row.keys() else None,
            perceptual_hash=row["perceptual_hash"] if "perceptual_hash" in row.keys() else None,
            c2pa_verified=bool(row["c2pa_verified"]) if "c2pa_verified" in row.keys() and row["c2pa_verified"] is not None else False,
            c2pa_issuer=row["c2pa_issuer"] if "c2pa_issuer" in row.keys() else None,
        )

    # ------------------------------------------------------------------
    # Verification-status helper
    # ------------------------------------------------------------------

    @staticmethod
    def _verification_where(status: Optional[str]) -> str:
        """Return a safe WHERE clause fragment for miner-media verification status.

        Only accepts known values; raises ValueError otherwise.
        """
        _CLAUSES = {
            "pending": "verified = 0 AND failed_verification = 0",
            "verified": "verified = 1 AND failed_verification = 0",
            "failed": "verified = 0 AND failed_verification = 1",
        }
        if status is None:
            return "1=1"
        if status not in _CLAUSES:
            raise ValueError(
                f"Invalid verification_status: {status}. "
                "Must be 'pending', 'verified', 'failed', or None"
            )
        return _CLAUSES[status]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_media_entry(
        self,
        prompt_id: str,
        file_path: str,
        modality: Modality,
        media_type: MediaType,
        source_type: SourceType = SourceType.GENERATED,
        download_url: Optional[str] = None,
        scraper_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_source_file: Optional[str] = None,
        dataset_index: Optional[str] = None,
        model_name: Optional[str] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        uid: Optional[int] = None,
        hotkey: Optional[str] = None,
        verified: bool = False,
        failed_verification: bool = False,
        rewarded: bool = False,
        timestamp: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
        file_size: Optional[int] = None,
        format: Optional[str] = None,
        mask_path: Optional[str] = None,
        perceptual_hash: Optional[str] = None,
        c2pa_verified: Optional[bool] = None,
        c2pa_issuer: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Add a media entry to the database. Returns the media id."""
        media_id = str(uuid.uuid4())
        created_at = time.time()

        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO media (
                    id, prompt_id, file_path, modality, media_type, source_type,
                    download_url, scraper_name, dataset_name, dataset_source_file, dataset_index,
                    model_name, generation_args, uid, hotkey,
                    verified, failed_verification, rewarded,
                    created_at, mask_path, timestamp, resolution, file_size, format,
                    perceptual_hash, c2pa_verified, c2pa_issuer, task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id, prompt_id, file_path, modality.value, media_type.value, source_type.value,
                    download_url, scraper_name, dataset_name, dataset_source_file, dataset_index,
                    model_name, json.dumps(generation_args) if generation_args else None, uid, hotkey,
                    1 if verified else 0, 1 if failed_verification else 0, 1 if rewarded else 0,
                    created_at, mask_path, timestamp,
                    json.dumps(list(resolution)) if resolution else None,
                    file_size, format,
                    perceptual_hash,
                    1 if c2pa_verified else 0 if c2pa_verified is not None else None,
                    c2pa_issuer,
                    task_id,
                ),
            )
            conn.commit()

        return media_id

    def get_media_entries(
        self, prompt_id: Optional[str] = None, media_id: Optional[str] = None
    ) -> List[MediaEntry]:
        """Get media entries by prompt_id or media_id."""
        if prompt_id is not None:
            col = "prompt_id"
            val = prompt_id
        elif media_id is not None:
            col = "id"
            val = media_id
        else:
            raise ValueError("Must provide either prompt_id or media_id")

        with self.db.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row
            cursor = conn.execute(f"SELECT * FROM media WHERE {col} = ?", (val,))
            return [self._row_to_media_entry(row) for row in cursor.fetchall()]

    def delete_media_entry_by_file_path(self, file_path: str) -> bool:
        """Delete a media entry by its file path."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute("DELETE FROM media WHERE file_path = ?", (file_path,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error deleting media entry for {file_path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Miner-media queries
    # ------------------------------------------------------------------

    def get_miner_media(self, verification_status: Optional[str] = None) -> List[MediaEntry]:
        """Get miner media by verification status."""
        where_clause = self._verification_where(verification_status)
        with self.db.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row
            cursor = conn.execute(
                f"SELECT * FROM media WHERE source_type = 'miner' AND {where_clause} ORDER BY created_at ASC"
            )
            return [self._row_to_media_entry(row) for row in cursor.fetchall()]

    def count_miner_media(self, verification_status: Optional[str] = None) -> int:
        """Count miner media by verification status."""
        where_clause = self._verification_where(verification_status)
        with self.db.connect() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM media WHERE source_type = 'miner' AND {where_clause}"
            )
            return int(cursor.fetchone()[0])

    def get_unrewarded_verified_miner_media(self, limit: int = 100) -> List[MediaEntry]:
        """Get verified miner media that hasn't been rewarded yet."""
        try:
            if limit is None:
                limit = 100
            with self.db.connect() as conn:
                conn.row_factory = __import__("sqlite3").Row
                cursor = conn.execute(
                    """
                    SELECT * FROM media 
                    WHERE source_type = 'miner' AND verified = 1 AND (rewarded = 0 OR rewarded IS NULL)
                    ORDER BY created_at ASC LIMIT ?
                    """,
                    (int(limit),),
                )
                return [self._row_to_media_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            bt.logging.error(f"Error getting unrewarded verified miner media: {e}")
            return []

    def get_recent_verified_miner_media(self, lookback_hours: float = 2.0, limit: int = 1000) -> List[MediaEntry]:
        """Get verified miner media from the last N hours."""
        try:
            cutoff = time.time() - (lookback_hours * 3600)
            with self.db.connect() as conn:
                conn.row_factory = __import__("sqlite3").Row
                cursor = conn.execute(
                    """
                    SELECT * FROM media 
                    WHERE source_type = 'miner' AND verified = 1 AND created_at >= ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (cutoff, int(limit or 1000)),
                )
                entries = [self._row_to_media_entry(row) for row in cursor.fetchall()]
                bt.logging.debug(f"Found {len(entries)} verified miner media in last {lookback_hours}h")
                return entries
        except Exception as e:
            bt.logging.error(f"Error getting recent verified miner media: {e}")
            return []

    def get_recent_failed_miner_media(self, lookback_hours: float = 2.0, limit: int = 1000) -> List[MediaEntry]:
        """Get miner media that failed verification in the last N hours."""
        try:
            cutoff = time.time() - (lookback_hours * 3600)
            with self.db.connect() as conn:
                conn.row_factory = __import__("sqlite3").Row
                cursor = conn.execute(
                    """
                    SELECT * FROM media
                    WHERE source_type = 'miner' AND verified = 0 AND failed_verification = 1 AND created_at >= ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (cutoff, int(limit or 1000)),
                )
                entries = [self._row_to_media_entry(row) for row in cursor.fetchall()]
                bt.logging.debug(f"Found {len(entries)} failed miner media in last {lookback_hours}h")
                return entries
        except Exception as e:
            bt.logging.error(f"Error getting recent failed miner media: {e}")
            return []

    def mark_miner_media_verified(self, media_id: str) -> bool:
        """Mark a miner media entry as verified, also updating its challenge outcome."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(
                    "UPDATE media SET verified = 1, failed_verification = 0 WHERE id = ? AND source_type = 'miner'",
                    (media_id,),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            if updated:
                self.challenges.mark_outcome_for_media(media_id, "verified")
            return updated
        except Exception as e:
            bt.logging.error(f"Error marking miner media as verified: {e}")
            return False

    def mark_miner_media_failed_verification(self, media_id: str) -> bool:
        """Mark a miner media entry as failed verification, also updating its challenge outcome."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(
                    "UPDATE media SET verified = 0, failed_verification = 1 WHERE id = ? AND source_type = 'miner'",
                    (media_id,),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            if updated:
                self.challenges.mark_outcome_for_media(media_id, "failed", "clip_verification_failed")
            return updated
        except Exception as e:
            bt.logging.error(f"Error marking miner media as failed verification: {e}")
            return False

    def update_media_embedding(self, media_id: str, embedding_blob: bytes) -> bool:
        """Store a CLIP embedding for a media entry."""
        try:
            with self.db.connect() as conn:
                conn.execute("UPDATE media SET clip_embedding = ? WHERE id = ?", (embedding_blob, media_id))
                conn.commit()
                return True
        except Exception as e:
            bt.logging.error(f"Error updating media embedding: {e}")
            return False

    def get_stored_embeddings(
        self, exclude_ids: Optional[List[str]] = None, limit: int = 5000
    ) -> List[tuple]:
        """Get stored CLIP embeddings for duplicate detection."""
        try:
            with self.db.connect() as conn:
                if exclude_ids:
                    placeholders = ",".join("?" * len(exclude_ids))
                    cursor = conn.execute(
                        f"""SELECT id, clip_embedding FROM media
                            WHERE clip_embedding IS NOT NULL AND source_type = 'miner'
                              AND id NOT IN ({placeholders})
                            ORDER BY created_at DESC LIMIT ?""",
                        (*exclude_ids, limit),
                    )
                else:
                    cursor = conn.execute(
                        """SELECT id, clip_embedding FROM media
                            WHERE clip_embedding IS NOT NULL AND source_type = 'miner'
                            ORDER BY created_at DESC LIMIT ?""",
                        (limit,),
                    )
                return cursor.fetchall()
        except Exception as e:
            bt.logging.error(f"Error getting stored embeddings: {e}")
            return []

    # ------------------------------------------------------------------
    # Upload / reward state
    # ------------------------------------------------------------------

    def get_unuploaded_media(
        self, limit: int = 100, modality: str = None, source_type: str = None
    ) -> List[MediaEntry]:
        """Get media entries that haven't been uploaded to HuggingFace yet."""
        try:
            limit = int(limit) if limit is not None else None
            with self.db.connect() as conn:
                conn.row_factory = __import__("sqlite3").Row
                base_where = "(uploaded = 0 OR uploaded IS NULL)"

                params: list = []
                if source_type == 'miner':
                    source_filter = "(source_type = 'miner' AND verified = 1)"
                elif source_type == 'generated':
                    source_filter = "source_type = 'generated'"
                elif source_type in ('scraper', 'dataset'):
                    source_filter = "source_type = ?"
                    params.append(source_type)
                else:
                    source_filter = "(source_type IN ('scraper', 'dataset', 'generated') OR (source_type = 'miner' AND verified = 1))"

                query = f"SELECT * FROM media WHERE {base_where} AND {source_filter}"
                if modality:
                    query += " AND modality = ?"
                    params.append(modality)
                query += " ORDER BY (source_type = 'miner' AND verified = 1) DESC, created_at ASC"
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                entries = []
                for row in cursor.fetchall():
                    entry = self._row_to_media_entry(row)
                    entry.prompt_content = ""
                    if entry.prompt_id:
                        try:
                            content = self.prompts.get_prompt_by_id(entry.prompt_id)
                            entry.prompt_content = content or ""
                        except Exception as e:
                            bt.logging.warning(f"Failed to fetch prompt content for {entry.prompt_id}: {e}")
                    entries.append(entry)
                return entries
        except Exception as e:
            bt.logging.error(f"Error getting unuploaded media: {e}")
            return []

    def count_unuploaded_media(self, modality: Optional[str] = None, source_type: Optional[str] = None) -> int:
        """Count unuploaded media."""
        try:
            base_where = "(uploaded = 0 OR uploaded IS NULL)"
            if source_type == 'miner':
                source_filter = "(source_type = 'miner' AND verified = 1)"
            elif source_type == 'generated':
                source_filter = "source_type = 'generated'"
            elif source_type in ('scraper', 'dataset'):
                source_filter = f"source_type = '{source_type}'"
            else:
                source_filter = "(source_type IN ('scraper', 'dataset', 'generated') OR (source_type = 'miner' AND verified = 1))"
            query = f"SELECT COUNT(*) FROM media WHERE {base_where} AND {source_filter}"
            params: tuple = ()
            if modality:
                query += " AND modality = ?"
                params = (modality,)
            with self.db.connect() as conn:
                return int(conn.execute(query, params).fetchone()[0])
        except Exception as e:
            bt.logging.error(f"Error counting unuploaded media: {e}")
            return 0

    def mark_media_uploaded(self, media_ids: List[str]) -> bool:
        """Mark media entries as uploaded to HuggingFace."""
        if not media_ids:
            return True
        try:
            with self.db.connect() as conn:
                placeholders = ",".join("?" * len(media_ids))
                cursor = conn.execute(
                    f"UPDATE media SET uploaded = 1 WHERE id IN ({placeholders})", media_ids
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error marking media as uploaded: {e}")
            return False

    def mark_media_rewarded(self, media_ids: List[str]) -> bool:
        """Mark media entries as rewarded."""
        if not media_ids:
            return True
        try:
            with self.db.connect() as conn:
                placeholders = ",".join("?" * len(media_ids))
                cursor = conn.execute(
                    f"UPDATE media SET rewarded = 1 WHERE id IN ({placeholders})", media_ids
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error marking media as rewarded: {e}")
            return False

    # ------------------------------------------------------------------
    # Stats & counts
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.db.connect() as conn:
            prompt_counts = dict(conn.execute(
                "SELECT content_type, COUNT(*) FROM prompts GROUP BY content_type"
            ).fetchall())
            media_counts = dict(conn.execute(
                "SELECT modality, COUNT(*) FROM media GROUP BY modality"
            ).fetchall())
            avg_usage = conn.execute("SELECT AVG(used_count) FROM prompts").fetchone()[0] or 0
            return {
                "total_prompts": sum(prompt_counts.values()),
                "total_media": sum(media_counts.values()),
                "prompt_counts": prompt_counts,
                "media_counts": media_counts,
                "average_prompt_usage": avg_usage,
                "database_size_mb": self.db.db_path.stat().st_size / (1024 * 1024),
            }

    def get_dataset_media_counts(self) -> Dict[str, int]:
        """Get counts of dataset media entries by modality/media_type."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT modality, media_type, COUNT(*) FROM media WHERE prompt_id IS NULL GROUP BY modality, media_type"
            )
            return {f"{m}_{t}": c for m, t, c in cursor.fetchall()}

    def get_source_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts of media grouped by source type and source name."""
        results: Dict[str, Dict[str, int]] = {"dataset": {}, "scraper": {}, "generated": {}}
        with self.db.connect() as conn:
            for name, cnt in conn.execute(
                "SELECT dataset_name, COUNT(*) FROM media WHERE source_type = 'dataset' AND dataset_name IS NOT NULL GROUP BY dataset_name ORDER BY COUNT(*) DESC"
            ).fetchall():
                results["dataset"][name] = cnt
            for name, cnt in conn.execute(
                "SELECT scraper_name, COUNT(*) FROM media WHERE source_type = 'scraper' AND scraper_name IS NOT NULL GROUP BY scraper_name ORDER BY COUNT(*) DESC"
            ).fetchall():
                results["scraper"][name] = cnt
            for name, cnt in conn.execute(
                "SELECT model_name, COUNT(*) FROM media WHERE source_type = 'generated' AND model_name IS NOT NULL GROUP BY model_name ORDER BY COUNT(*) DESC"
            ).fetchall():
                results["generated"][name] = cnt
        return results

    def get_source_count(self, source_type: SourceType, source_name: str) -> int:
        """Get count of media items for a particular source."""
        col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
        if not col:
            return 0
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM media WHERE source_type = ? AND {col} = ?",
                (source_type.value, source_name),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_media_entries(
        self,
        k: int,
        modality: Modality,
        media_type: MediaType,
        strategy: str = "random",
        remove: bool = False,
    ) -> List[MediaEntry]:
        """Sample media entries with various strategies."""
        with self.db.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row

            def rows_to_entries(rows: List) -> List[MediaEntry]:
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

            from sqlite3 import Row as SQLiteRow
            _ = SQLiteRow  # type hint only

            if strategy in ("random", "least_used", "oldest", "newest"):
                order_map = {
                    "random": "RANDOM()",
                    "least_used": "COALESCE(p.used_count, 0) ASC, RANDOM()",
                    "oldest": "m.created_at ASC, RANDOM()",
                    "newest": "m.created_at DESC, RANDOM()",
                }
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM media m
                    LEFT JOIN prompts p ON m.prompt_id = p.id
                    WHERE m.modality = ? AND m.media_type = ?
                    ORDER BY {order_map[strategy]}
                    LIMIT ?
                    """,
                    (modality.value, media_type.value, k),
                ).fetchall()
                return rows_to_entries(rows)

            if strategy == "random_source":
                items: List[MediaEntry] = []
                for _ in range(k):
                    sources = conn.execute(
                        """
                        SELECT DISTINCT source_type,
                            CASE
                                WHEN source_type = 'generated' THEN model_name
                                WHEN source_type = 'dataset' THEN dataset_name
                                WHEN source_type = 'scraper' THEN scraper_name
                                ELSE NULL
                            END as source_name
                        FROM media
                        WHERE modality = ? AND media_type = ?
                            AND ((source_type = 'generated' AND model_name IS NOT NULL)
                              OR (source_type = 'dataset' AND dataset_name IS NOT NULL)
                              OR (source_type = 'scraper' AND scraper_name IS NOT NULL))
                        """,
                        (modality.value, media_type.value),
                    ).fetchall()
                    if not sources:
                        row = conn.execute(
                            """
                            SELECT m.* FROM media m LEFT JOIN prompts p ON m.prompt_id = p.id
                            WHERE m.modality = ? AND m.media_type = ? ORDER BY RANDOM() LIMIT 1
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

                    source_type_val, source_name = random.choice(sources)
                    source_column = SOURCE_TYPE_TO_DB_NAME_FIELD.get(SourceType(source_type_val))
                    if not source_column or not source_name:
                        continue

                    row = conn.execute(
                        f"""
                        SELECT m.* FROM media m LEFT JOIN prompts p ON m.prompt_id = p.id
                        WHERE m.modality = ? AND m.media_type = ? AND m.source_type = ? AND m.{source_column} = ?
                        ORDER BY RANDOM() LIMIT 1
                        """,
                        (modality.value, media_type.value, source_type_val, source_name),
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

    def get_media_by_model(self, model_name: str, modality: Optional[Modality] = None) -> List[MediaEntry]:
        """Get all media entries generated by a specific model."""
        with self.db.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row
            if modality:
                cursor = conn.execute(
                    "SELECT * FROM media WHERE model_name = ? AND modality = ? ORDER BY created_at DESC",
                    (model_name, modality.value),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM media WHERE model_name = ? ORDER BY created_at DESC",
                    (model_name,),
                )
            return [self._row_to_media_entry(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_old_entries(self, days_old: int = 30, min_usage: int = 0) -> int:
        """Clean up old and unused prompt entries."""
        cutoff = time.time() - (days_old * 24 * 3600)
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT id FROM prompts WHERE created_at < ? AND used_count <= ?",
                (cutoff, min_usage),
            )
            ids = [row[0] for row in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM media WHERE prompt_id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM prompts WHERE id IN ({placeholders})", ids)
                conn.commit()
            return len(ids)

    def prune_source_media(
        self, source_type: SourceType, source_name: str, max_count: int, strategy: str = "oldest"
    ) -> int:
        """Prune items for a source so that total count <= max_count."""
        col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
        if not col:
            return 0

        current = self.get_source_count(source_type, source_name)
        if current <= max_count:
            return 0

        to_remove = current - max_count
        order = "created_at ASC" if strategy in ("oldest", "least_used") else "RANDOM()"

        with self.db.connect() as conn:
            cursor = conn.execute(
                f"SELECT id, file_path FROM media WHERE source_type = ? AND {col} = ? ORDER BY {order} LIMIT ?",
                (source_type.value, source_name, to_remove),
            )
            rows = cursor.fetchall()
            if not rows:
                return 0

            media_ids = [r[0] for r in rows]
            file_paths = [r[1] for r in rows]
            ids_ph = ",".join("?" * len(media_ids))
            files_ph = ",".join("?" * len(file_paths))

            conn.execute(f"DELETE FROM prompts WHERE source_media_id IN ({ids_ph})", media_ids)
            conn.execute(f"DELETE FROM media WHERE file_path IN ({files_ph})", file_paths)
            conn.commit()
            return len(rows)

    def cleanup_uploaded_media(
        self, min_age_hours: float = 24.0, require_rewarded: bool = True, batch_size: int = 1000,
    ) -> Tuple[int, int, List[str]]:
        """Delete media entries that have been uploaded (and optionally rewarded)."""
        try:
            cutoff = time.time() - (min_age_hours * 3600)
            with self.db.connect() as conn:
                conn.row_factory = __import__("sqlite3").Row
                if require_rewarded:
                    where = "uploaded = 1 AND rewarded = 1 AND created_at < ?"
                else:
                    where = "uploaded = 1 AND created_at < ?"

                rows = conn.execute(
                    f"SELECT id, file_path, prompt_id FROM media WHERE {where} LIMIT ?",
                    (cutoff, batch_size),
                ).fetchall()
                if not rows:
                    return 0, 0, []

                media_ids = [r["id"] for r in rows]
                file_paths = [r["file_path"] for r in rows]
                prompt_ids = [r["prompt_id"] for r in rows if r["prompt_id"]]

                mid_ph = ",".join("?" * len(media_ids))
                conn.execute(f"DELETE FROM generator_challenge_outcomes WHERE media_id IN ({mid_ph})", media_ids)
                conn.execute(f"DELETE FROM media WHERE id IN ({mid_ph})", media_ids)
                conn.commit()

                if prompt_ids:
                    pid_ph = ",".join("?" * len(prompt_ids))
                    conn.execute(f"DELETE FROM prompts WHERE id IN ({pid_ph})", prompt_ids)
                    conn.commit()

                return len(media_ids), len(prompt_ids), file_paths
        except Exception as e:
            bt.logging.error(f"Error cleaning up uploaded media: {e}")
            return 0, 0, []
