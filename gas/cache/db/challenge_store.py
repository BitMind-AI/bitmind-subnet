"""ChallengeStore — CRUD and reward stats for the generator_challenge_outcomes table."""

import json
import sqlite3
import time
from typing import List, Dict, Optional, Any, Tuple

import bittensor as bt

from gas.cache.types import ChallengeOutcome
from gas.cache.db.connection import ConnectionManager


def _parse_resolution(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse the media table's JSON-encoded (width, height) resolution."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return (int(data[0]), int(data[1]))
    except (ValueError, TypeError, IndexError, json.JSONDecodeError):
        return None


class ChallengeStore:
    """Data access for the ``generator_challenge_outcomes`` table."""

    def __init__(self, db: ConnectionManager):
        self.db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        task_id: str,
        uid: int,
        hotkey: str,
        prompt_id: str,
        modality: str,
        status: str = "pending",
        failure_reason: Optional[str] = None,
        media_id: Optional[str] = None,
        created_at: Optional[float] = None,
        requested_resolution: Optional[str] = None,
    ) -> bool:
        """Insert or update a generation challenge outcome."""
        try:
            now = time.time()
            created_at = created_at or now
            with self.db.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO generator_challenge_outcomes (
                        task_id, uid, hotkey, prompt_id, modality, status,
                        failure_reason, media_id, requested_resolution, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        uid = excluded.uid,
                        hotkey = excluded.hotkey,
                        prompt_id = excluded.prompt_id,
                        modality = excluded.modality,
                        status = excluded.status,
                        failure_reason = excluded.failure_reason,
                        media_id = COALESCE(excluded.media_id, generator_challenge_outcomes.media_id),
                        requested_resolution = COALESCE(excluded.requested_resolution, generator_challenge_outcomes.requested_resolution),
                        updated_at = excluded.updated_at
                    """,
                    (task_id, uid, hotkey, prompt_id, modality, status, failure_reason, media_id, requested_resolution, created_at, now),
                )
                conn.commit()
                return True
        except Exception as e:
            bt.logging.error(f"Error recording challenge outcome for task {task_id}: {e}")
            return False

    def update_outcome(
        self,
        task_id: str,
        status: str,
        failure_reason: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> bool:
        """Update an existing challenge outcome status."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE generator_challenge_outcomes
                    SET status = ?, failure_reason = ?, media_id = COALESCE(?, media_id), updated_at = ?
                    WHERE task_id = ?
                    """,
                    (status, failure_reason, media_id, time.time(), task_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error updating challenge outcome for task {task_id}: {e}")
            return False

    def mark_outcome_for_media(
        self,
        media_id: str,
        status: str,
        failure_reason: Optional[str] = None,
    ) -> bool:
        """Update a challenge outcome using its stored media row.

        When *status* is ``'verified'``, also copies ``model_name`` from
        the ``media`` table into the outcome row.
        """
        try:
            with self.db.connect() as conn:
                if status == "verified":
                    cursor = conn.execute(
                        """
                        UPDATE generator_challenge_outcomes
                        SET status = ?, failure_reason = ?,
                            media_id = COALESCE(media_id, ?),
                            model_name = COALESCE(
                                (SELECT model_name FROM media WHERE id = ?), model_name
                            ),
                            updated_at = ?
                        WHERE media_id = ? OR task_id = (SELECT task_id FROM media WHERE id = ?)
                        """,
                        (status, failure_reason, media_id, media_id, time.time(), media_id, media_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE generator_challenge_outcomes
                        SET status = ?, failure_reason = ?, media_id = COALESCE(media_id, ?), updated_at = ?
                        WHERE media_id = ? OR task_id = (SELECT task_id FROM media WHERE id = ?)
                        """,
                        (status, failure_reason, media_id, time.time(), media_id, media_id),
                    )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            bt.logging.error(f"Error updating challenge outcome for media {media_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_outcomes_last_n_hours(
        self, lookback_hours: float = 2.0, limit: int = 1000
    ) -> List[ChallengeOutcome]:
        """Get terminal generation challenge outcomes from the last N hours."""
        try:
            cutoff = time.time() - (lookback_hours * 3600)
            if limit is None:
                limit = 1000
            with self.db.connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT o.*, m.resolution AS media_resolution, m.has_audio AS media_has_audio
                    FROM generator_challenge_outcomes o
                    LEFT JOIN media m ON o.media_id = m.id
                    WHERE o.status IN ('verified', 'failed') AND o.updated_at >= ?
                    ORDER BY o.updated_at DESC LIMIT ?
                    """,
                    (cutoff, int(limit)),
                )
                return [
                    ChallengeOutcome(
                        task_id=row["task_id"],
                        uid=row["uid"],
                        hotkey=row["hotkey"],
                        prompt_id=row["prompt_id"],
                        modality=row["modality"],
                        status=row["status"],
                        failure_reason=row["failure_reason"],
                        media_id=row["media_id"],
                        model_name=row["model_name"] if "model_name" in row.keys() else None,
                        requested_resolution=(
                            row["requested_resolution"] if "requested_resolution" in row.keys() else None
                        ),
                        observed_resolution=_parse_resolution(row["media_resolution"]),
                        has_audio=(
                            bool(row["media_has_audio"]) if row["media_has_audio"] is not None else None
                        ),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            bt.logging.error(f"Error getting recent challenge outcomes: {e}")
            return []

    def get_outcome_stats_last_n_hours(
        self, lookback_hours: float = 2.0, limit: int = 1000
    ) -> Dict[str, Dict[str, Any]]:
        """Build reward stats from terminal challenge outcomes, split by modality."""
        outcomes = self.get_outcomes_last_n_hours(lookback_hours, limit)
        miner_stats: Dict[str, Dict[str, Any]] = {}
        for outcome in outcomes:
            hotkey = outcome.hotkey
            if hotkey not in miner_stats:
                miner_stats[hotkey] = {
                    "uid": outcome.uid,
                    "verified_media_ids": [],
                    "total_verified": 0, "total_failed": 0,
                    "image_verified": 0, "video_verified": 0,
                    "image_failed": 0, "video_failed": 0,
                    "image_model_names": [], "video_model_names": [],
                    "image_generations": [], "video_generations": [],
                    "last_timestamp": outcome.updated_at,
                }
            modality = (outcome.modality or "").lower()
            if outcome.status == "verified":
                miner_stats[hotkey]["total_verified"] += 1
                if modality == "image":
                    miner_stats[hotkey]["image_verified"] += 1
                    if outcome.model_name:
                        miner_stats[hotkey]["image_model_names"].append(outcome.model_name)
                    miner_stats[hotkey]["image_generations"].append({
                        "model_name": outcome.model_name,
                        "requested_resolution": outcome.requested_resolution,
                        "observed_resolution": (
                            list(outcome.observed_resolution)
                            if outcome.observed_resolution else None
                        ),
                    })
                elif modality == "video":
                    miner_stats[hotkey]["video_verified"] += 1
                    if outcome.model_name:
                        miner_stats[hotkey]["video_model_names"].append(outcome.model_name)
                    miner_stats[hotkey]["video_generations"].append({
                        "model_name": outcome.model_name,
                        "requested_resolution": outcome.requested_resolution,
                        "observed_resolution": (
                            list(outcome.observed_resolution)
                            if outcome.observed_resolution else None
                        ),
                        "has_audio": outcome.has_audio,
                    })
                if outcome.media_id:
                    miner_stats[hotkey]["verified_media_ids"].append(outcome.media_id)
            elif outcome.status == "failed":
                miner_stats[hotkey]["total_failed"] += 1
                if modality == "image":
                    miner_stats[hotkey]["image_failed"] += 1
                elif modality == "video":
                    miner_stats[hotkey]["video_failed"] += 1
            if outcome.updated_at > miner_stats[hotkey]["last_timestamp"]:
                miner_stats[hotkey]["last_timestamp"] = outcome.updated_at

        result = {}
        for hotkey, stats in miner_stats.items():
            verified = stats["total_verified"]
            failed = stats["total_failed"]
            total = verified + failed
            img_v, vid_v = stats["image_verified"], stats["video_verified"]
            img_f, vid_f = stats["image_failed"], stats["video_failed"]
            img_t, vid_t = img_v + img_f, vid_v + vid_f
            result[hotkey] = {
                "uid": stats["uid"],
                "total_verified": verified,
                "total_submissions": total,
                "total_failed": failed,
                "total_evaluated": total,
                "pass_rate": (verified / total) if total > 0 else 0.0,
                "image_verified": img_v,
                "image_failed": img_f,
                "image_pass_rate": (img_v / img_t) if img_t > 0 else 0.0,
                "image_model_names": stats.get("image_model_names", []),
                "image_generations": stats.get("image_generations", []),
                "video_verified": vid_v,
                "video_failed": vid_f,
                "video_pass_rate": (vid_v / vid_t) if vid_t > 0 else 0.0,
                "video_model_names": stats.get("video_model_names", []),
                "video_generations": stats.get("video_generations", []),
                "media_ids": stats["verified_media_ids"],
                "last_timestamp": stats["last_timestamp"],
            }
        return result
