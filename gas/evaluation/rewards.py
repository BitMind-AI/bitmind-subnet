import math
import time
from typing import Dict, Iterable, Optional

import bittensor as bt

from collections import Counter

# Model generation cost in USD per second of video (720p, no audio unless noted).
# Used to compute reward multipliers: miner gets baseline_ratio * (model_price / baseline_price).
#
# These are defaults.  If OPEN_ROUTER_API_KEY is set, live prices are fetched from
# the /videos/models endpoint at module load and merged on top (so newly added
# models or price changes are picked up automatically).
#
# C2PA blindness note: several providers do NOT expose the model variant in their
# C2PA manifests.  All Veo (any provider) returns None.  All Runway proprietary
# models share the same "RunwayML" softwareAgent.  We conservatively use the
# cheapest variant as baseline and accept the ambiguity in the reward multiplier.
GENERATOR_MODEL_PRICES: Dict[str, float] = {
    # Google Veo family (C2PA: no variant exposed — all return None → baseline)
    "google/veo-3.1-lite":  0.03,   # cheapest Veo — used as baseline
    "google/veo-3.1-fast":  0.08,
    "google/veo-3.1":       0.20,
    # ByteDance Seedance (C2PA: params.model_name — variant IS exposed)
    "dreamina-seedance-2-0-fast":  0.05,
    "dreamina-seedance-2-0":       0.12,
    "dreamina-seedance-1-5-pro":   0.25,
    # Runway proprietary models (C2PA: softwareAgent = "RunwayML Video Generation")
    # All share the same C2PA signature; variant is not exposed.
    # Keyed at cheapest variant (gen3a_turbo/gen4_turbo/act_two, 5 credits/s).
    # Full Runway pricing at ~$0.01/credit:
    #   gen4.5 (12cr/s)  gen4_turbo (5cr/s)  gen4_aleph (15cr/s)
    #   gen3a_turbo (5cr/s)  act_two (5cr/s)
    # Runway-resold Veo models return None (same as Google Veo) → baseline.
    "RunwayML": 0.05,
}

# Cheapest model price — all multipliers are relative to this.
_GENERATOR_BASELINE_PRICE: float = min(GENERATOR_MODEL_PRICES.values())

# Cached live prices — refreshed at most once per process lifetime on first import.
# The /videos/models endpoint is free and public (no auth).
_LIVE_PRICES_FETCHED = False


def _fetch_openrouter_prices() -> Dict[str, float]:
    """Pull live video model prices from OpenRouter /videos/models (free, no auth).

    Only runs once per process — result is cached at module level."""
    global _LIVE_PRICES_FETCHED
    if _LIVE_PRICES_FETCHED:
        return {}

    _LIVE_PRICES_FETCHED = True
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/videos/models",
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        models = resp.json().get("data", [])
        prices: Dict[str, float] = {}
        for m in models:
            model_id = m.get("id", "")
            skus = m.get("pricing_skus", {}) or {}
            # Find the cheapest per-second price for text-to-video
            candidates = []
            for key, val in skus.items():
                if "text_to_video" not in key and "duration_seconds" not in key:
                    continue
                try:
                    candidates.append(float(val))
                except (ValueError, TypeError):
                    pass
            if candidates:
                prices[model_id] = min(candidates)
        bt.logging.info(f"Fetched {len(prices)} live OpenRouter video model prices")
        return prices
    except Exception as e:
        bt.logging.debug(f"Could not fetch OpenRouter prices: {e}")
        return {}


# Merge live prices on top of defaults (live wins over hardcoded for same key).
_live = _fetch_openrouter_prices()
if _live:
    GENERATOR_MODEL_PRICES = {**GENERATOR_MODEL_PRICES, **_live}
    _GENERATOR_BASELINE_PRICE = min(GENERATOR_MODEL_PRICES.values())


def _get_model_price(model_name: str) -> float:
    """Return the USD/second price for a model name, or the baseline price if unknown."""
    if not model_name:
        return _GENERATOR_BASELINE_PRICE
    lower = model_name.lower()
    # Direct match first
    for key, price in GENERATOR_MODEL_PRICES.items():
        if key.lower() == lower:
            return price
    # Substring match (for C2PA-extracted names like "Google")
    for key, price in GENERATOR_MODEL_PRICES.items():
        if key.lower() in lower:
            return price
    return _GENERATOR_BASELINE_PRICE


def _compute_average_model_multiplier(model_names: list[str]) -> float:
    """Average price multiplier across a miner's verified submissions.

    Uses sqrt(price/baseline) to taper extreme ratios (e.g. 8.33x → 2.89x).
    """
    if not model_names:
        return 1.0
    total = 0.0
    for name in model_names:
        price = _get_model_price(name)
        total += math.sqrt(price / _GENERATOR_BASELINE_PRICE)
    return total / len(model_names)


def get_discriminator_rewards(
    runs,
    metagraph,
    image_score_weight: float = 0.5,
    video_score_weight: float = 0.5,
    binary_score_weight: float = 0.5,
    multiclass_score_weight: float = 0.5,
):
    """
    Process discriminator results to extract rewards from MCC scores.
    Handles separate image and video detection models per miner.

    Args:
        runs: List of DiscriminatorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping
        image_score_weight: Weight for image modality rewards
        video_score_weight: Weight for video modality rewards
        binary_score_weight: Weight for binary MCC
        multiclass_score_weight: Weight for multiclass MCC

    Returns:
        dict: Mapping of UID to combined reward score for discriminators
    """
    
    # Store rewards by UID and modality
    miner_modality_rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    if not runs:
        bt.logging.warning("No discriminator runs data provided")
        return {}

    try:
        for result in runs:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue

            ss58_address = result.get("discriminator_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            uid = ss58_to_uid[ss58_address]
            modality = result.get("modality")
            if not modality:
                bt.logging.warning(f"Missing modality for UID {uid}")
                continue

            if uid not in miner_modality_rewards:
                miner_modality_rewards[uid] = {}

            # Handle potential None or non-numeric values
            binary_mcc = result.get("binary_mcc", 0)
            multiclass_mcc = result.get("multiclass_mcc", 0)

            try:
                binary_mcc = float(binary_mcc) if binary_mcc is not None else 0
                multiclass_mcc = float(multiclass_mcc) if multiclass_mcc is not None else 0
            except (ValueError, TypeError):
                bt.logging.warning(f"Invalid MCC values for UID {uid}: binary={binary_mcc}, multiclass={multiclass_mcc}")
                binary_mcc = 0
                multiclass_mcc = 0

            miner_modality_rewards[uid][modality] = binary_score_weight * max(
                0, binary_mcc
            ) + multiclass_score_weight * max(0, multiclass_mcc)

    except Exception as e:
        bt.logging.error(f"Error processing discriminator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    # Combine image and video rewards for each miner
    final_multipliers = {}
    for uid, modality_rewards in miner_modality_rewards.items():
        final_multipliers[uid] = image_score_weight * modality_rewards.get(
            "image", 0.0
        ) + video_score_weight * modality_rewards.get("video", 0.0)

    return final_multipliers


def get_generator_base_rewards(verification_stats):
    """
    Compute base rewards for generators based on their verification pass rates.

    Args:
        verification_stats: Dict mapping hotkey to verification stats from
                            ContentManager.get_verification_stats_last_n_hours()
            Expected format:
            {
                "hotkey": {
                    "uid": int,
                    "total_verified": int,
                    "total_failed": int,
                    "total_evaluated": int,
                    "pass_rate": float,
                    "media_ids": List[str]
                }
            }

    Returns:
        tuple: (uid_rewards_dict, media_ids_to_mark)
            - uid_rewards_dict: Mapping of UID to base reward score
            - media_ids_to_mark: List of media IDs to mark as rewarded
    """
    try:
        if not verification_stats:
            return {}, []

        # Convert to UID-based rewards and collect media IDs
        uid_rewards = {}
        all_media_ids = []

        for hotkey, stats in verification_stats.items():
            uid = int(stats["uid"])
            pass_rate = stats["pass_rate"]
            verified_count = stats["total_verified"]

            # Base reward: pass_rate scaled by verified count with diminishing-returns
            # taper.  First 10 count linearly; beyond 10, log₂ takes over so high-
            # throughput miners still see gains but can't farm volume indefinitely.
            volume = min(verified_count, 10) + max(0.0, math.log2(max(1, verified_count - 9)))
            base_reward = pass_rate * volume

            # Model tier multiplier — rewards miners for using higher-cost models
            model_names = stats.get("model_names", [])
            model_mult = _compute_average_model_multiplier(model_names)
            base_reward *= model_mult

            uid_rewards[uid] = base_reward
            all_media_ids.extend(stats["media_ids"])

        bt.logging.info(f"Computed base rewards for {len(uid_rewards)} miners: {uid_rewards}")

        return uid_rewards, all_media_ids

    except Exception as e:
        bt.logging.error(f"Error in get_generator_base_rewards: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        return {}, []


def get_generator_reward_multipliers(
    generator_results, 
    metagraph,
    generator_liveness: Optional[Dict[str, float]] = None,
    max_inactive_hours: int = 24,
):
    """
    Process generator results to extract rewards from fool counts.
    
    Optionally filters out inactive generators based on liveness tracking.

    Args:
        generator_results: List of GeneratorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping
        generator_liveness: Optional dict mapping hotkey to last activity timestamp.
                           If provided, generators not seen within max_inactive_hours will get 0 rewards.
        max_inactive_hours: Maximum hours of inactivity before generator is considered inactive (default: 48)

    Returns:
        dict: Mapping of UID to reward score for generators
    """    
    rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    if not generator_results:
        bt.logging.warning("No generator results data provided")
        return {}

    # Build set of active generators if liveness data provided
    active_generators = None
    inactive_count = 0
    if generator_liveness:
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600
        active_generators = {
            hotkey for hotkey, last_seen in generator_liveness.items()
            if (current_time - last_seen) <= max_inactive_seconds
        }
        bt.logging.info(f"Liveness filter: {len(active_generators)} active generators (within {max_inactive_hours}h)")

    # Aggregate fool counts by generator (ss58_address)
    generator_fooled_counts = {}
    generator_not_fooled_counts = {}

    try:
        for result in generator_results:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue                
            ss58_address = result.get("ss58_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            # Skip inactive generators if liveness filter is enabled
            if active_generators is not None and ss58_address not in active_generators:
                inactive_count += 1
                bt.logging.debug(f"Skipping inactive generator {ss58_address[:16]}...")
                continue

            fooled_count = result.get("fooled_count", 0)
            not_fooled_count = result.get("not_fooled_count", 0)
            
            try:
                fooled_count = int(fooled_count) if fooled_count is not None else 0
                not_fooled_count = int(not_fooled_count) if not_fooled_count is not None else 0
            except (ValueError, TypeError):
                bt.logging.warning(f"Invalid counts for {ss58_address}: fooled={fooled_count}, not_fooled={not_fooled_count}")
                fooled_count = 0
                not_fooled_count = 0

            if ss58_address not in generator_fooled_counts:
                generator_fooled_counts[ss58_address] = 0
                generator_not_fooled_counts[ss58_address] = 0

            generator_fooled_counts[ss58_address] += fooled_count
            generator_not_fooled_counts[ss58_address] += not_fooled_count

        # Calculate fool rate for each generator from accumulated counts with sample size bonus
        for ss58_address in generator_fooled_counts:
            if ss58_address in ss58_to_uid:
                uid = ss58_to_uid[ss58_address]
                total_fooled = generator_fooled_counts[ss58_address]
                total_not_fooled = generator_not_fooled_counts[ss58_address]
                total_count = total_fooled + total_not_fooled

                if total_count > 0:
                    # Base fool rate
                    fool_rate = total_fooled / total_count

                    # Sample size multiplier: rewards higher sample sizes
                    # Uses logarithmic scaling to provide diminishing returns
                    # Reference count of 20 gives multiplier of 1.0, higher counts get bonus
                    reference_count = 20
                    max_multiplier = 2.0  # Cap the maximum multiplier

                    if total_count >= reference_count:
                        sample_size_multiplier = min(max_multiplier, 1.0 + math.log(total_count / reference_count))
                    else:
                        # Penalize very small sample sizes
                        sample_size_multiplier = max(0.5, total_count / reference_count)

                    # Final reward combines fool rate with sample size bonus
                    base_reward = fool_rate * sample_size_multiplier
                    rewards[uid] = max(0, min(2.0, base_reward))  # Allow rewards up to 2.0 for high sample sizes

                    bt.logging.debug(f"Generator {ss58_address[:8]}... UID {uid}: fool_rate={fool_rate:.3f}, "
                                   f"sample_size={total_count}, sample_size_multiplier={sample_size_multiplier:.3f}, "
                                   f"final_multiplier={rewards[uid]:.3f}")
                else:
                    bt.logging.warning(f"Zero total count for generator {ss58_address}")

        inactive_msg = f", skipped {inactive_count} inactive" if inactive_count > 0 else ""
        bt.logging.info(f"Processed {len(generator_results)} generator results, computed rewards for {len(rewards)} generators{inactive_msg}")

    except Exception as e:
        bt.logging.error(f"Error processing generator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    return rewards


def _clamp(value, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = min_value
    if math.isnan(value):
        value = min_value
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _get_artifact_rewards(
    artifact_stats,
    correctness_keys: tuple[str, ...],
    default_correctness_key: str,
    role_name: str,
) -> Dict[int, float]:
    """
    Compute DPS artifact-mechanism rewards from verification stats.

    Expected stat fields per miner:
        uid, accepted_work_units, correctness/quality rate,
        availability_rate, timeliness_multiplier, novelty_multiplier, penalties

    Compatibility aliases are accepted so the DPS validator layer can feed this
    function from existing verification summaries while full task pipelines are
    being rolled out.
    """
    rewards: Dict[int, float] = {}
    if not artifact_stats:
        return rewards

    if isinstance(artifact_stats, dict):
        records = artifact_stats.values()
    else:
        records = artifact_stats

    for stats in records:
        if not isinstance(stats, dict) or "uid" not in stats:
            continue

        try:
            uid = int(stats["uid"])
        except (TypeError, ValueError):
            continue

        accepted_work_units = _clamp(
            stats.get("accepted_work_units", stats.get("total_verified", 0))
        )
        correctness_value = stats.get(default_correctness_key)
        for key in correctness_keys:
            if correctness_value is not None:
                break
            correctness_value = stats.get(key)
        correctness_rate = _clamp(correctness_value, max_value=1.0)
        availability_rate = _clamp(
            stats.get("availability_rate", 1.0 if accepted_work_units > 0 else 0.0),
            max_value=1.0,
        )
        timeliness_multiplier = _clamp(
            stats.get("timeliness_multiplier", stats.get("timeliness", 1.0))
        )
        novelty_multiplier = _clamp(
            stats.get("novelty_multiplier", stats.get("novelty", 1.0))
        )
        penalties = _clamp(stats.get("penalties", stats.get("penalty", 0.0)))

        score = (
            accepted_work_units
            * correctness_rate
            * availability_rate
            * timeliness_multiplier
            * novelty_multiplier
            - penalties
        )
        rewards[uid] = max(0.0, score)

    bt.logging.info(f"Computed {role_name} rewards for {len(rewards)} miners: {rewards}")
    return rewards


def get_encoder_rewards(encoder_stats) -> Dict[int, float]:
    """
    Compute DPS encoder rewards from deterministic verification stats.

    Formula:
        accepted_work_units * deterministic_correctness_rate *
        availability_rate * timeliness_multiplier * novelty_multiplier - penalties
    """
    return _get_artifact_rewards(
        encoder_stats,
        correctness_keys=("correctness_rate", "pass_rate"),
        default_correctness_key="deterministic_correctness_rate",
        role_name="encoder",
    )


def get_captioner_rewards(captioner_stats) -> Dict[int, float]:
    """
    Compute DPS captioner rewards for mechanism 1.

    Captioning is quality-verifiable rather than deterministic, so validators
    should feed a semantic quality score once that pipeline is live.
    """
    return _get_artifact_rewards(
        captioner_stats,
        correctness_keys=("semantic_quality_rate", "quality_score", "pass_rate"),
        default_correctness_key="caption_quality_rate",
        role_name="captioner",
    )


def artifact_stats_with_uids(role_stats, hotkeys):
    records = artifact_stat_records(role_stats)
    hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
    normalized_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record = dict(record)
        if "uid" not in record:
            hotkey = (
                record.get("hotkey")
                or record.get("ss58_address")
                or record.get("coldkey")
            )
            if hotkey in hotkey_to_uid:
                record["uid"] = hotkey_to_uid[hotkey]
        if "uid" in record:
            normalized_records.append(record)
    return normalized_records


def artifact_stat_records(role_stats):
    if not role_stats:
        return []
    if not isinstance(role_stats, dict):
        return role_stats
    if any(
        key in role_stats
        for key in (
            "uid",
            "hotkey",
            "ss58_address",
            "accepted_work_units",
            "total_verified",
            "pass_rate",
            "quality_score",
        )
    ):
        return [role_stats]

    records = []
    for hotkey, stats in role_stats.items():
        if not isinstance(stats, dict):
            continue
        stats = dict(stats)
        stats.setdefault("hotkey", hotkey)
        records.append(stats)
    return records


def normalize_rewards_to_weight_budget(
    scores,
    active_uids: Iterable[int],
    special_uids: Iterable[int],
    budget: float,
) -> tuple[object, float]:
    """
    Normalize role scores into a fixed weight budget.

    Returns:
        (weights, unallocated_budget). Unallocated budget is non-zero when there
        are no active non-special UIDs with positive score.
    """
    import numpy as np

    weights = np.zeros_like(scores, dtype=np.float32)
    special_uids = set(special_uids)
    active_uids = [
        int(uid)
        for uid in active_uids
        if 0 <= int(uid) < len(scores) and int(uid) not in special_uids
    ]

    if not active_uids or budget <= 0:
        return weights, max(0.0, float(budget))

    role_scores = np.array([max(0.0, float(scores[uid])) for uid in active_uids])
    score_sum = float(np.sum(role_scores))
    if score_sum <= 0 or np.isnan(score_sum):
        return weights, max(0.0, float(budget))

    weights[active_uids] = role_scores / score_sum * float(budget)
    return weights, 0.0
