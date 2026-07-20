import math
import time
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from gas.evaluation.resolution_tiers import effective_tier

# Model generation cost in USD per second of video (720p, no audio unless noted).
# Used to compute reward multipliers: miner gets baseline_ratio * (model_price / baseline_price).
#
# These are defaults.  If OPEN_ROUTER_API_KEY is set, live prices are fetched from
# the /videos/models endpoint at module load and merged on top (so newly added
# models or price changes are picked up automatically).
#
# C2PA blindness note: several providers do NOT expose the model variant in their
# C2PA manifests.  All Veo (any provider) returns None.  All Runway proprietary
# models share the same "RunwayML" softwareAgent.  Veo videos are priced by a
# resolution/audio floor (see MODEL_TIER_PRICES); other unknowns get the baseline.
GENERATOR_MODEL_PRICES: Dict[str, float] = {
    # Google Veo family (C2PA: no variant exposed — all return None → tier floor)
    "google/veo-3.1-lite":  0.03,   # cheapest Veo — used as baseline
    "google/veo-3.1-fast":  0.08,
    "google/veo-3.1":       0.20,
    # ByteDance Seedance (C2PA: params.model_name — variant IS exposed)
    # Pricing is per-token; figures are 720p-with-audio references (USD/s).
    # Covers both OpenRouter (bytedance/*) and Runway (seedance2/seedance2_fast) —
    # both route through ByteDance's own C2PA signing (sig issuer: Byteplus Pte. Ltd.).
    # Confirmed via live test: seedance-1-5-pro has NO C2PA on OpenRouter and is
    # not offered on Runway — only the 2.0 variants are validator-eligible.
    "dreamina-seedance-2-0-fast":  0.12,   # ~$0.121/s at 720p
    "dreamina-seedance-2-0":       0.15,   # ~$0.151/s at 720p
    # Runway gen4.5: C2PA manifest present but claimSignature.mismatch — validators
    # reject all gen4.5 content until Runway fixes their signing infra.
    # "RunwayML": 0.05,  # re-enable when gen4.5 signature is fixed
}

# Per-(resolution tier, audio) USD/s prices for models whose C2PA manifests
# hide the variant (the Veo family).  A Veo video is priced at the cheapest
# variant able to produce its observed (tier, audio) combination — a price
# floor.  Since challenges request a tier and rewards use
# min(observed, requested), overshooting a request never pays more.
#
# Keys mirror OpenRouter pricing_skus: the un-suffixed duration_seconds SKU is
# the 1080p default; _720p/_4k suffixes are explicit tiers.  Defaults below
# mirror live values as of 2026-07; live SKUs are merged on top at module load.
MODEL_TIER_PRICES: Dict[str, Dict[Tuple[str, bool], float]] = {
    "google/veo-3.1-lite": {
        ("720p", False): 0.03, ("720p", True): 0.05,
        ("1080p", False): 0.05, ("1080p", True): 0.08,
    },
    "google/veo-3.1-fast": {
        ("720p", False): 0.08, ("720p", True): 0.10,
        ("1080p", False): 0.10, ("1080p", True): 0.12,
        ("4K", False): 0.25, ("4K", True): 0.30,
    },
    "google/veo-3.1": {
        ("1080p", False): 0.20, ("1080p", True): 0.40,
        ("4K", False): 0.40, ("4K", True): 0.60,
    },
}

# Models priced via the tier floor when C2PA gives model_name=None.
_TIER_FLOOR_FAMILY_PREFIX = "google/veo"

# Cheapest model price — all multipliers are relative to this.
_GENERATOR_BASELINE_PRICE: float = min(GENERATOR_MODEL_PRICES.values())

# Cached live prices — refreshed at most once per process lifetime on first import.
# The /videos/models endpoint is free and public (no auth).
_LIVE_PRICES_FETCHED = False


def _parse_duration_sku_key(key: str) -> Optional[Tuple[str, bool]]:
    """Map a duration_seconds SKU key to a (tier, has_audio) pair.

    "duration_seconds_with_audio" -> ("1080p", True)   # no suffix = 1080p default
    "duration_seconds_without_audio_720p" -> ("720p", False)
    "duration_seconds_with_audio_4k" -> ("4K", True)
    """
    if "duration_seconds" not in key:
        return None
    has_audio = "without_audio" not in key
    if key.endswith("_720p"):
        tier = "720p"
    elif key.endswith("_4k"):
        tier = "4K"
    else:
        tier = "1080p"
    return tier, has_audio


def _fetch_openrouter_prices() -> Tuple[Dict[str, float], Dict[str, Dict[Tuple[str, bool], float]]]:
    """Pull live video model prices from OpenRouter /videos/models (free, no auth).

    Returns (flat_prices, tier_prices).  Only runs once per process — result is
    cached at module level."""
    global _LIVE_PRICES_FETCHED
    if _LIVE_PRICES_FETCHED:
        return {}, {}

    _LIVE_PRICES_FETCHED = True
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/videos/models",
            timeout=10,
        )
        if resp.status_code != 200:
            return {}, {}
        models = resp.json().get("data", [])
        prices: Dict[str, float] = {}
        tier_prices: Dict[str, Dict[Tuple[str, bool], float]] = {}
        for m in models:
            model_id = m.get("id", "")
            skus = m.get("pricing_skus", {}) or {}
            # Find the cheapest per-second price for text-to-video
            candidates = []
            tier_table: Dict[Tuple[str, bool], float] = {}
            for key, val in skus.items():
                if "text_to_video" not in key and "duration_seconds" not in key:
                    continue
                try:
                    price = float(val)
                except (ValueError, TypeError):
                    continue
                candidates.append(price)
                tier_key = _parse_duration_sku_key(key)
                if tier_key:
                    tier_table[tier_key] = price
            if candidates:
                prices[model_id] = min(candidates)
            if tier_table:
                tier_prices[model_id] = tier_table
        bt.logging.info(f"Fetched {len(prices)} live OpenRouter video model prices")
        return prices, tier_prices
    except Exception as e:
        bt.logging.debug(f"Could not fetch OpenRouter prices: {e}")
        return {}, {}


# Merge live prices on top of defaults (live wins over hardcoded for same key).
_live, _live_tiers = _fetch_openrouter_prices()
if _live:
    GENERATOR_MODEL_PRICES = {**GENERATOR_MODEL_PRICES, **_live}
    _GENERATOR_BASELINE_PRICE = min(GENERATOR_MODEL_PRICES.values())
if _live_tiers:
    MODEL_TIER_PRICES = {**MODEL_TIER_PRICES, **_live_tiers}


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


def _get_video_generation_price(generation: Dict[str, Any]) -> float:
    """USD/s price for one verified video generation.

    Named models (Seedance et al. expose model_name via C2PA) use their flat
    price.  C2PA-blind generations (model_name=None — the Veo family) are
    priced at the cheapest Veo variant able to produce the video's
    (resolution tier, audio) combination, where the tier is
    min(observed, requested) so overshooting a challenge request never pays.
    An unknown observed resolution earns only the baseline.
    """
    model_name = generation.get("model_name")
    if model_name:
        return _get_model_price(model_name)

    tier = effective_tier(
        generation.get("observed_resolution"),
        generation.get("requested_resolution"),
    )
    if tier is None:
        return _GENERATOR_BASELINE_PRICE

    has_audio = bool(generation.get("has_audio"))
    candidates = [
        table[(tier, has_audio)]
        for model, table in MODEL_TIER_PRICES.items()
        if model.startswith(_TIER_FLOOR_FAMILY_PREFIX) and (tier, has_audio) in table
    ]
    # No variant offers this (tier, audio) — e.g. sub-720p output — so no
    # premium can be justified.
    return min(candidates) if candidates else _GENERATOR_BASELINE_PRICE


def _compute_video_generation_multiplier(generations: List[Dict[str, Any]]) -> float:
    """Average price multiplier across a miner's verified video generations.

    Same sqrt taper as _compute_average_model_multiplier, but priced per
    generation from (model_name, resolution tier, audio) instead of model
    name alone.
    """
    if not generations:
        return 1.0
    total = 0.0
    for generation in generations:
        price = _get_video_generation_price(generation)
        total += math.sqrt(price / _GENERATOR_BASELINE_PRICE)
    return total / len(generations)


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
                    "video_generations": list[dict],  # per-generation model/resolution/audio
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
            # Prefer per-generation (model, resolution tier, audio) pricing;
            # fall back to model names for stats produced before tiered pricing.
            video_generations = stats.get("video_generations")
            if video_generations:
                video_model_mult = _compute_video_generation_multiplier(video_generations)
            else:
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
