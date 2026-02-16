"""
Comprehensive tests for gas.evaluation.rewards module.

Tests cover:
- Discriminator reward calculation with various inputs
- Generator base reward computation from verification stats
- Generator reward multipliers with fool rate and sample size scaling
- Edge cases: empty inputs, invalid data types, missing fields
- Liveness filtering for generator rewards
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch


from gas.evaluation.rewards import (
    get_discriminator_rewards,
    get_generator_base_rewards,
    get_generator_reward_multipliers,
)


def _make_metagraph(hotkeys):
    """Create a mock metagraph with the given hotkey list."""
    mg = MagicMock()
    mg.hotkeys = list(hotkeys)
    return mg


class TestGetDiscriminatorRewards(unittest.TestCase):
    """Tests for get_discriminator_rewards."""

    def test_empty_runs_returns_empty(self):
        mg = _make_metagraph(["hk0", "hk1"])
        result = get_discriminator_rewards([], mg)
        self.assertEqual(result, {})

    def test_none_runs_returns_empty(self):
        mg = _make_metagraph(["hk0"])
        result = get_discriminator_rewards(None, mg)
        self.assertEqual(result, {})

    def test_single_image_result(self):
        mg = _make_metagraph(["hk0", "hk1"])
        runs = [
            {
                "discriminator_address": "hk1",
                "modality": "image",
                "binary_mcc": 0.8,
                "multiclass_mcc": 0.6,
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        # image reward = 0.5 * (0.5*0.8 + 0.5*0.6) = 0.5 * 0.7 = 0.35
        self.assertIn(1, result)
        self.assertAlmostEqual(result[1], 0.35, places=5)

    def test_image_and_video_combined(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "modality": "image",
                "binary_mcc": 1.0,
                "multiclass_mcc": 1.0,
            },
            {
                "discriminator_address": "hk0",
                "modality": "video",
                "binary_mcc": 0.5,
                "multiclass_mcc": 0.5,
            },
        ]
        result = get_discriminator_rewards(runs, mg)
        # image_reward = 0.5*1.0 + 0.5*1.0 = 1.0
        # video_reward = 0.5*0.5 + 0.5*0.5 = 0.5
        # final = 0.5*1.0 + 0.5*0.5 = 0.75
        self.assertAlmostEqual(result[0], 0.75, places=5)

    def test_negative_mcc_clamped_to_zero(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "modality": "image",
                "binary_mcc": -0.5,
                "multiclass_mcc": -1.0,
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        # max(0, -0.5) = 0, max(0, -1.0) = 0 → reward = 0
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_none_mcc_values_default_to_zero(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "modality": "image",
                "binary_mcc": None,
                "multiclass_mcc": None,
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_unknown_hotkey_ignored(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "unknown_key",
                "modality": "image",
                "binary_mcc": 0.9,
                "multiclass_mcc": 0.9,
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        self.assertEqual(result, {})

    def test_missing_modality_skipped(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "binary_mcc": 0.9,
                "multiclass_mcc": 0.9,
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        self.assertEqual(result, {})

    def test_invalid_result_type_skipped(self):
        mg = _make_metagraph(["hk0"])
        runs = ["not_a_dict", 42, None]
        result = get_discriminator_rewards(runs, mg)
        self.assertEqual(result, {})

    def test_string_mcc_values_converted(self):
        """MCC values that come as strings should be converted to float."""
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "modality": "image",
                "binary_mcc": "0.7",
                "multiclass_mcc": "0.3",
            }
        ]
        result = get_discriminator_rewards(runs, mg)
        expected = 0.5 * (0.5 * 0.7 + 0.5 * 0.3)
        self.assertAlmostEqual(result[0], expected, places=5)

    def test_custom_weights(self):
        mg = _make_metagraph(["hk0"])
        runs = [
            {
                "discriminator_address": "hk0",
                "modality": "image",
                "binary_mcc": 1.0,
                "multiclass_mcc": 0.0,
            },
            {
                "discriminator_address": "hk0",
                "modality": "video",
                "binary_mcc": 0.0,
                "multiclass_mcc": 1.0,
            },
        ]
        result = get_discriminator_rewards(
            runs,
            mg,
            image_score_weight=0.7,
            video_score_weight=0.3,
            binary_score_weight=1.0,
            multiclass_score_weight=0.0,
        )
        # image: 1.0*1.0 + 0.0*0.0 = 1.0
        # video: 1.0*0.0 + 0.0*1.0 = 0.0
        # final = 0.7*1.0 + 0.3*0.0 = 0.7
        self.assertAlmostEqual(result[0], 0.7, places=5)

    def test_multiple_miners(self):
        mg = _make_metagraph(["hk0", "hk1", "hk2"])
        runs = [
            {"discriminator_address": "hk0", "modality": "image", "binary_mcc": 0.5, "multiclass_mcc": 0.5},
            {"discriminator_address": "hk2", "modality": "image", "binary_mcc": 1.0, "multiclass_mcc": 1.0},
        ]
        result = get_discriminator_rewards(runs, mg)
        self.assertIn(0, result)
        self.assertIn(2, result)
        self.assertNotIn(1, result)


class TestGetGeneratorBaseRewards(unittest.TestCase):
    """Tests for get_generator_base_rewards."""

    def test_empty_stats_returns_empty(self):
        rewards, media_ids = get_generator_base_rewards({})
        self.assertEqual(rewards, {})
        self.assertEqual(media_ids, [])

    def test_none_stats_returns_empty(self):
        rewards, media_ids = get_generator_base_rewards(None)
        self.assertEqual(rewards, {})
        self.assertEqual(media_ids, [])

    def test_single_miner_perfect_pass_rate(self):
        stats = {
            "hotkey_a": {
                "uid": 5,
                "total_verified": 8,
                "total_failed": 0,
                "total_evaluated": 8,
                "pass_rate": 1.0,
                "media_ids": ["m1", "m2"],
            }
        }
        rewards, media_ids = get_generator_base_rewards(stats)
        # reward = 1.0 * min(8, 10) = 8.0
        self.assertEqual(rewards[5], 8.0)
        self.assertEqual(media_ids, ["m1", "m2"])

    def test_volume_bonus_capped_at_ten(self):
        stats = {
            "hk": {
                "uid": 0,
                "total_verified": 50,
                "total_failed": 0,
                "total_evaluated": 50,
                "pass_rate": 1.0,
                "media_ids": ["m1"],
            }
        }
        rewards, _ = get_generator_base_rewards(stats)
        # Capped: 1.0 * min(50, 10) = 10.0
        self.assertEqual(rewards[0], 10.0)

    def test_zero_pass_rate(self):
        stats = {
            "hk": {
                "uid": 1,
                "total_verified": 0,
                "total_failed": 5,
                "total_evaluated": 5,
                "pass_rate": 0.0,
                "media_ids": ["m1"],
            }
        }
        rewards, _ = get_generator_base_rewards(stats)
        self.assertEqual(rewards[1], 0.0)

    def test_multiple_miners_media_ids_aggregated(self):
        stats = {
            "hk0": {
                "uid": 0,
                "total_verified": 3,
                "total_failed": 1,
                "total_evaluated": 4,
                "pass_rate": 0.75,
                "media_ids": ["a", "b"],
            },
            "hk1": {
                "uid": 1,
                "total_verified": 5,
                "total_failed": 0,
                "total_evaluated": 5,
                "pass_rate": 1.0,
                "media_ids": ["c"],
            },
        }
        rewards, media_ids = get_generator_base_rewards(stats)
        self.assertEqual(len(rewards), 2)
        self.assertIn("a", media_ids)
        self.assertIn("c", media_ids)


class TestGetGeneratorRewardMultipliers(unittest.TestCase):
    """Tests for get_generator_reward_multipliers."""

    def test_empty_results_returns_empty(self):
        mg = _make_metagraph(["hk0"])
        result = get_generator_reward_multipliers([], mg)
        self.assertEqual(result, {})

    def test_none_results_returns_empty(self):
        mg = _make_metagraph(["hk0"])
        result = get_generator_reward_multipliers(None, mg)
        self.assertEqual(result, {})

    def test_basic_fool_rate(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 15, "not_fooled_count": 5},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        # fool_rate = 15/20 = 0.75, total=20 >= ref=20 → multiplier = 1.0
        # reward = 0.75 * 1.0 = 0.75
        self.assertIn(0, rewards)
        self.assertAlmostEqual(rewards[0], 0.75, places=3)

    def test_small_sample_penalized(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 5, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        # total=5, fool_rate=1.0, multiplier = max(0.5, 5/20) = 0.5 → but 5/20=0.25 < 0.5
        # reward = 1.0 * 0.5 = 0.5
        self.assertAlmostEqual(rewards[0], 0.5, places=3)

    def test_large_sample_bonus(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 100, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        # total=100, fool_rate=1.0
        # multiplier = min(2.0, 1.0 + log(100/20)) = min(2.0, 1.0+1.609) = min(2.0, 2.609) = 2.0
        # reward = 1.0 * 2.0 = 2.0
        self.assertAlmostEqual(rewards[0], 2.0, places=3)

    def test_reward_clamped_to_two(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 10000, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        self.assertLessEqual(rewards[0], 2.0)

    def test_zero_total_count(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 0, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        self.assertNotIn(0, rewards)

    def test_aggregation_across_multiple_results(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 10, "not_fooled_count": 5},
            {"ss58_address": "hk0", "fooled_count": 5, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        # total_fooled=15, total_not_fooled=5, total=20
        # fool_rate=0.75, multiplier=1.0 → reward=0.75
        self.assertAlmostEqual(rewards[0], 0.75, places=3)

    def test_liveness_filter_excludes_inactive(self):
        mg = _make_metagraph(["hk0", "hk1"])
        current = time.time()
        results = [
            {"ss58_address": "hk0", "fooled_count": 20, "not_fooled_count": 0},
            {"ss58_address": "hk1", "fooled_count": 20, "not_fooled_count": 0},
        ]
        liveness = {
            "hk0": current - 3600,       # active (1h ago)
            "hk1": current - 100 * 3600,  # inactive (100h ago)
        }
        rewards = get_generator_reward_multipliers(
            results, mg, generator_liveness=liveness, max_inactive_hours=24
        )
        self.assertIn(0, rewards)
        self.assertNotIn(1, rewards)

    def test_liveness_filter_none_means_no_filtering(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": 20, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(
            results, mg, generator_liveness=None
        )
        self.assertIn(0, rewards)

    def test_invalid_counts_default_to_zero(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "hk0", "fooled_count": "abc", "not_fooled_count": None},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        # Both default to 0 → total=0 → no reward
        self.assertNotIn(0, rewards)

    def test_unknown_address_ignored(self):
        mg = _make_metagraph(["hk0"])
        results = [
            {"ss58_address": "unknown", "fooled_count": 20, "not_fooled_count": 0},
        ]
        rewards = get_generator_reward_multipliers(results, mg)
        self.assertEqual(rewards, {})


class TestRewardEdgeCases(unittest.TestCase):
    """Edge cases and integration-style tests."""

    def test_discriminator_last_modality_wins_on_overwrite(self):
        """When same uid has multiple results for same modality, last one wins."""
        mg = _make_metagraph(["hk0"])
        runs = [
            {"discriminator_address": "hk0", "modality": "image", "binary_mcc": 0.1, "multiclass_mcc": 0.1},
            {"discriminator_address": "hk0", "modality": "image", "binary_mcc": 0.9, "multiclass_mcc": 0.9},
        ]
        result = get_discriminator_rewards(runs, mg)
        # Last image result overwrites: 0.5*0.9 + 0.5*0.9 = 0.9
        # final = 0.5 * 0.9 = 0.45
        self.assertAlmostEqual(result[0], 0.45, places=5)

    def test_generator_base_rewards_uid_as_string(self):
        """uid field as string should be converted to int."""
        stats = {
            "hk": {
                "uid": "3",
                "total_verified": 5,
                "total_failed": 0,
                "total_evaluated": 5,
                "pass_rate": 0.8,
                "media_ids": [],
            }
        }
        rewards, _ = get_generator_base_rewards(stats)
        self.assertIn(3, rewards)


if __name__ == "__main__":
    unittest.main()
