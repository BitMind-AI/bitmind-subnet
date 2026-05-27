import math
import time
from typing import Dict, Optional

import bittensor as bt

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


def get_generator_base_rewards(verification_stats):
    """
    Compute base rewards for generators based on their verification pass rates,
    split by modality (image/video) so they can be weighted independently.

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
                    "image_verified": int,
                    "image_failed": int,
                    "image_pass_rate": float,
                    "image_model_names": list[str],
                    "video_verified": int,
                    "video_failed": int,
                    "video_pass_rate": float,
                    "video_model_names": list[str],
                    "media_ids": List[str]
                }
            }

    Returns:
        tuple: (uid_rewards_dict, media_ids_to_mark)
            - uid_rewards_dict: Mapping of UID to {"image": float, "video": float}
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

            # --- Image modality ---
            image_verified = stats.get("image_verified", 0)
            image_pass_rate = stats.get("image_pass_rate", 0.0)
            image_volume = min(image_verified, 10) + max(0.0, math.log2(max(1, image_verified - 9)))
            image_base = image_pass_rate * image_volume
            image_model_mult = _compute_average_model_multiplier(
                stats.get("image_model_names", [])
            )
            image_base *= image_model_mult

            # --- Video modality ---
            video_verified = stats.get("video_verified", 0)
            video_pass_rate = stats.get("video_pass_rate", 0.0)
            video_volume = min(video_verified, 10) + max(0.0, math.log2(max(1, video_verified - 9)))
            video_base = video_pass_rate * video_volume
            video_model_mult = _compute_average_model_multiplier(
                stats.get("video_model_names", [])
            )
            video_base *= video_model_mult

            uid_rewards[uid] = {"image": image_base, "video": video_base}
            all_media_ids.extend(stats["media_ids"])

        bt.logging.info(f"Computed per-modality base rewards for {len(uid_rewards)} miners: {uid_rewards}")

        return uid_rewards, all_media_ids

    except Exception as e:
        bt.logging.error(f"Error in get_generator_base_rewards: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        return {}, []


def get_generator_fool_bonuses(
    generator_results, 
    metagraph,
    generator_liveness: Optional[Dict[str, float]] = None,
    max_inactive_hours: int = 24,
):
    """
    Compute fool-rate bonuses for generators.  Returns a bonus in [0, 2] that
    is applied as (1 + bonus) to base verification rewards, so miners with zero
    fool rate still receive their base rewards and fooling detectors adds extra.

    Optionally filters out inactive generators based on liveness tracking.

    Args:
        generator_results: List of GeneratorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping
        generator_liveness: Optional dict mapping hotkey to last activity timestamp.
                           If provided, generators not seen within max_inactive_hours are excluded.
        max_inactive_hours: Maximum hours of inactivity (default: 24)

    Returns:
        dict: Mapping of UID to fool-rate bonus (0.0–2.0)
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

                    # Fool-rate bonus — added on top of base rewards as (1 + bonus)
                    fool_bonus = fool_rate * sample_size_multiplier
                    rewards[uid] = max(0.0, min(2.0, fool_bonus))

                    bt.logging.debug(f"Generator {ss58_address[:8]}... UID {uid}: fool_rate={fool_rate:.3f}, "
                                   f"sample_size={total_count}, sample_size_multiplier={sample_size_multiplier:.3f}, "
                                   f"fool_bonus={rewards[uid]:.3f}")
                else:
                    bt.logging.warning(f"Zero total count for generator {ss58_address}")

        inactive_msg = f", skipped {inactive_count} inactive" if inactive_count > 0 else ""
        bt.logging.info(f"Processed {len(generator_results)} generator results, computed rewards for {len(rewards)} generators{inactive_msg}")

    except Exception as e:
        bt.logging.error(f"Error processing generator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    return rewards
