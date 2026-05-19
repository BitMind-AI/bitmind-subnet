import tempfile
import unittest
from pathlib import Path

from gas.cache.db import ConnectionManager, PromptStore, MediaStore, ChallengeStore
from gas.cache.db.connection import create_schema
from gas.types import MediaType, Modality, SourceType


class ChallengeOutcomeStatsTest(unittest.TestCase):
    def test_challenge_outcomes_count_pre_storage_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "prompts.db"
            conn = ConnectionManager(db_path)
            with conn.connect() as c:
                create_schema(c)

            prompts = PromptStore(conn)
            challenges = ChallengeStore(conn)
            media = MediaStore(conn, prompts, challenges)

            prompt_id = prompts.add_prompt_entry("a test prompt", modality="image")
            media_id = media.add_media_entry(
                prompt_id=prompt_id,
                file_path=str(Path(tmpdir) / "media.png"),
                modality=Modality.IMAGE,
                media_type=MediaType.SYNTHETIC,
                source_type=SourceType.MINER,
                uid=1,
                hotkey="hotkey-1",
                verified=True,
                task_id="task-verified",
            )

            challenges.record_outcome(
                task_id="task-verified",
                uid=1,
                hotkey="hotkey-1",
                prompt_id=prompt_id,
                modality="image",
                status="verified",
                media_id=media_id,
            )
            challenges.record_outcome(
                task_id="task-rejected",
                uid=1,
                hotkey="hotkey-1",
                prompt_id=prompt_id,
                modality="image",
                status="failed",
                failure_reason="C2PA verification failed",
            )

            stats = challenges.get_outcome_stats_last_n_hours(lookback_hours=1)

            self.assertEqual(stats["hotkey-1"]["total_verified"], 1)
            self.assertEqual(stats["hotkey-1"]["total_failed"], 1)
            self.assertEqual(stats["hotkey-1"]["total_evaluated"], 2)
            self.assertEqual(stats["hotkey-1"]["pass_rate"], 0.5)
            self.assertEqual(stats["hotkey-1"]["media_ids"], [media_id])
            # Per-modality fields (hard cutover)
            self.assertEqual(stats["hotkey-1"]["image_verified"], 1)
            self.assertEqual(stats["hotkey-1"]["image_failed"], 1)
            self.assertEqual(stats["hotkey-1"]["image_pass_rate"], 0.5)
            self.assertEqual(stats["hotkey-1"]["video_verified"], 0)
            self.assertEqual(stats["hotkey-1"]["video_failed"], 0)
            self.assertEqual(stats["hotkey-1"]["video_pass_rate"], 0.0)
            self.assertEqual(stats["hotkey-1"]["image_model_names"], [])
            self.assertEqual(stats["hotkey-1"]["video_model_names"], [])


if __name__ == "__main__":
    unittest.main()
