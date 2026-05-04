import tempfile
import unittest
from pathlib import Path

from gas.cache.content_db import ContentDB
from gas.types import MediaType, Modality, SourceType


class ChallengeOutcomeStatsTest(unittest.TestCase):
    def test_challenge_outcomes_count_pre_storage_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ContentDB(Path(tmpdir) / "prompts.db")
            prompt_id = db.add_prompt_entry("a test prompt", modality="image")
            media_id = db.add_media_entry(
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

            db.record_challenge_outcome(
                task_id="task-verified",
                uid=1,
                hotkey="hotkey-1",
                prompt_id=prompt_id,
                modality="image",
                status="verified",
                media_id=media_id,
            )
            db.record_challenge_outcome(
                task_id="task-rejected",
                uid=1,
                hotkey="hotkey-1",
                prompt_id=prompt_id,
                modality="image",
                status="failed",
                failure_reason="C2PA verification failed",
            )

            stats = db.get_challenge_outcome_stats_last_n_hours(lookback_hours=1)

            self.assertEqual(stats["hotkey-1"]["total_verified"], 1)
            self.assertEqual(stats["hotkey-1"]["total_failed"], 1)
            self.assertEqual(stats["hotkey-1"]["total_evaluated"], 2)
            self.assertEqual(stats["hotkey-1"]["pass_rate"], 0.5)
            self.assertEqual(stats["hotkey-1"]["media_ids"], [media_id])


if __name__ == "__main__":
    unittest.main()
