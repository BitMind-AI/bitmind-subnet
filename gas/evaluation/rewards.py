import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from gas.evaluation.resolution_tiers import effective_tier, tier_shortfall

# Model generation cost in USD per second of video (720p, no audio unless noted).
# Used to compute reward multipliers: miner gets baseline_ratio * (model_price / baseline_price).
#
# These are defaults.  If OPEN_ROUTER_API_KEY is set, live prices are fetched from
# the /videos/models endpoint at module load and merged on top (so newly added
# models or price changes are picked up automatically).
#
# C2PA blindness note: several providers do NOT expose the model variant in their
# C2PA manifests.  All Veo (any provider) returns None.  All Runway proprietary
# models share the same "RunwayML" softwareAgent.  Veo videos are priced
# per delivered tier (see _VEO_TIER_PRICES); other unknowns get the baseline.
GENERATOR_MODEL_PRICES: Dict[str, float] = {
    # Google Veo family (C2PA: no variant exposed — all return None → tier floor)
    "google/veo-3.1-lite":  0.03,   # cheapest Veo — used as baseline
    "google/veo-3.1-fast":  0.08,
    "google/veo-3.1":       0.20,
    # ByteDance Seedance (C2PA: params.model_name — variant IS exposed)
    # 720p reference (USD/s); per-tier prices in NAMED_MODEL_TIER_PRICES take
    # precedence — these flat values only serve the legacy model-name path.
    # Covers both OpenRouter (bytedance/*) and Runway (seedance2/seedance2_fast) —
    # both route through ByteDance's own C2PA signing (sig issuer: Byteplus Pte. Ltd.).
    # Confirmed via live test: seedance-1-5-pro has NO C2PA on OpenRouter and is
    # not offered on Runway — only the 2.0 variants are validator-eligible.
    "dreamina-seedance-2-0-fast":  0.152,  # $0.76 per 5s at 720p (provider quote, 2026-07)
    "dreamina-seedance-2-0":       0.40,   # $2.00 per 5s at 720p (provider quote, 2026-07)
    # Runway gen4.5: C2PA manifest present but claimSignature.mismatch — validators
    # reject all gen4.5 content until Runway fixes their signing infra.
    # "RunwayML": 0.05,  # re-enable when gen4.5 signature is fixed
}

# Per-(resolution tier, audio) USD/s prices parsed from OpenRouter
# pricing_skus (the un-suffixed duration_seconds SKU is the 1080p default;
# _720p/_4k suffixes are explicit tiers).  Defaults mirror live values as of
# 2026-07; live SKUs are merged on top at module load.
#
# NOTE: not currently consulted for pricing — C2PA-blind Veo is priced via
# _VEO_TIER_PRICES instead.  Retained as the live
# price reference, and it becomes the pricing source again if Veo variants
# ever become distinguishable (manifest signal or a 4K capability probe).
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

# Per-tier USD/s prices for C2PA-blind Veo generations (model_name=None).
# Google's manifests never expose the variant, so pricing is by delivered
# tier only (audio-independent): 480p/720p match Seedance fast's rates
# (same resolution, same multiplier), and 1080p reflects that Veo's real
# price rises with resolution — $0.40/s is the veo-3.1 standard 1080p rate
# (also Seedance full's 720p rate), i.e. a 3.65x multiplier.
_VEO_TIER_PRICES: Dict[str, float] = {
    "480p": 0.07,
    "720p": 0.152,
    "1080p": 0.40,
}

# Per-tier USD/s prices for named models (C2PA exposes model_name).
# OpenRouter publishes no Seedance pricing SKUs, so these come from provider
# quotes (2026-07, per 5s video): seedance-2-0 $0.50 / $2.00 / $2.40 at
# 480p / 720p / 1080p; seedance-2-0-fast $0.35 / $0.76 at 480p / 720p.
# Note the spread is NOT pixel-proportional (full jumps ~4x from 480p to
# 720p but only ~1.2x from 720p to 1080p), which is why these are explicit
# tables rather than a shared scale.  Keys are matched against the C2PA
# model_name the same way as GENERATOR_MODEL_PRICES (exact, then substring).
NAMED_MODEL_TIER_PRICES: Dict[str, Dict[str, float]] = {
    "dreamina-seedance-2-0-fast": {
        "480p": 0.07,    # $0.35 / 5s
        "720p": 0.152,   # $0.76 / 5s
    },
    "dreamina-seedance-2-0": {
        "480p": 0.10,    # $0.50 / 5s
        "720p": 0.40,    # $2.00 / 5s
        "1080p": 0.48,   # $2.40 / 5s
    },
}

# Fallback resolution scaling for named models with no NAMED_MODEL_TIER_PRICES
# table: their flat 720p reference is scaled by the delivered tier's pixel
# ratio (480p is ~0.44x the 720p pixel count, 1080p ~2.25x).
NAMED_MODEL_TIER_SCALE: Dict[str, float] = {
    "480p": 0.44,
    "720p": 1.0,
    "1080p": 2.25,
}


def _get_named_model_tier_table(model_name: str) -> Optional[Dict[str, float]]:
    """Per-tier price table for a C2PA model name (exact, then substring match)."""
    lower = model_name.lower()
    for key, table in NAMED_MODEL_TIER_PRICES.items():
        if key.lower() == lower:
            return table
    for key, table in NAMED_MODEL_TIER_PRICES.items():
        if key.lower() in lower:
            return table
    return None

# Relative per-image prices by resolution tier (1K is the image baseline).
# Image C2PA manifests rarely expose a usable model variant and image APIs
# price per image (often tiered by output size), so images are priced purely
# by delivered resolution tier: min(observed, requested), same as video.
# Ratios approximate the resolution-tier spreads of the trusted image APIs
# (e.g. GPT Image 2 spans ~3x from 1K to 4K); tune here as providers change.
IMAGE_TIER_PRICES: Dict[str, float] = {
    "1K": 1.0,
    "2K": 2.0,
    "4K": 4.0,
}

_IMAGE_BASELINE_PRICE: float = min(IMAGE_TIER_PRICES.values())

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

    In both branches the tier is min(observed, requested) so overshooting a
    challenge request never pays, and an unknown observed resolution earns
    only the baseline — no premium without a verifiable resolution.

    Named models (Seedance et al. expose model_name via C2PA) use their
    per-tier quoted prices (NAMED_MODEL_TIER_PRICES), falling back to
    pixel-ratio scaling of their flat 720p reference.  C2PA-blind
    generations (model_name=None — the Veo family) are priced per
    delivered tier via _VEO_TIER_PRICES, audio-independent.
    """
    tier = effective_tier(
        generation.get("observed_resolution"),
        generation.get("requested_resolution"),
    )
    model_name = generation.get("model_name")
    if model_name:
        if tier is None:
            return _GENERATOR_BASELINE_PRICE
        table = _get_named_model_tier_table(model_name)
        if table and tier in table:
            price = table[tier]
        elif table:
            # Tier absent from the table (e.g. a fast variant somehow observed
            # at 1080p): pay the highest tier the table does price — the model
            # can't officially produce more, so no extrapolated premium.
            price = max(table.values())
        else:
            price = _get_model_price(model_name) * NAMED_MODEL_TIER_SCALE.get(tier, 1.0)
        # Baseline is the floor: a named model unknown to the price table
        # resolves to the baseline price, and tier down-scaling must not push
        # it (or any cheap model) below what an unpriceable generation earns.
        return max(price, _GENERATOR_BASELINE_PRICE)

    if tier is None:
        return _GENERATOR_BASELINE_PRICE

    price = _VEO_TIER_PRICES.get(tier, max(_VEO_TIER_PRICES.values()))
    return max(price, _GENERATOR_BASELINE_PRICE)


# Multiplier discount per tier of video undershoot (delivered below the
# requested tier).  min(observed, requested) pricing alone lets miners serve
# every challenge at the cheapest tier: under the sqrt taper, 480p Seedance
# earns ~1.8x for $0.50/5s while 720p earns ~3.65x for $2.00/5s — better
# reward-per-dollar at the bottom.  The discount keeps degrade-not-fail
# semantics (undershooting still pays, unlike a hard failure) but makes
# honoring the requested tier the highest-reward choice per challenge slot:
# e.g. on a 1080p request, 1080p Seedance earns 4.0x, 720p 3.65*0.6=2.19x,
# 480p 1.83*0.36=0.66x.  Declining instead of undershooting earns 0 and
# costs a volume slot, so the discount is not dodgeable by refusal.
VIDEO_UNDERSHOOT_TIER_DISCOUNT = 0.6


def _compute_video_generation_multiplier(generations: List[Dict[str, Any]]) -> float:
    """Average price multiplier across a miner's verified video generations.

    Same sqrt taper as _compute_average_model_multiplier, but priced per
    generation from (model_name, resolution tier, audio) instead of model
    name alone.  Deliveries below the requested tier are discounted per tier
    of shortfall (see VIDEO_UNDERSHOOT_TIER_DISCOUNT).
    """
    if not generations:
        return 1.0
    total = 0.0
    for generation in generations:
        price = _get_video_generation_price(generation)
        shortfall = tier_shortfall(
            generation.get("observed_resolution"),
            generation.get("requested_resolution"),
        )
        total += math.sqrt(price / _GENERATOR_BASELINE_PRICE) * (
            VIDEO_UNDERSHOOT_TIER_DISCOUNT ** shortfall
        )
    return total / len(generations)


def _get_image_generation_price(generation: Dict[str, Any]) -> float:
    """Relative price for one verified image generation, by delivered tier.

    Tier is min(observed, requested) so overshooting a challenge request never
    pays more.  An unknown observed resolution earns only the 1K baseline.
    """
    tier = effective_tier(
        generation.get("observed_resolution"),
        generation.get("requested_resolution"),
        modality="image",
    )
    if tier is None:
        return _IMAGE_BASELINE_PRICE
    return IMAGE_TIER_PRICES.get(tier, _IMAGE_BASELINE_PRICE)


def _compute_image_generation_multiplier(generations: List[Dict[str, Any]]) -> float:
    """Average resolution-tier multiplier across a miner's verified images.

    Same sqrt taper as video: a tier 4x the baseline price earns 2x, not 4x.
    """
    if not generations:
        return 1.0
    total = 0.0
    for generation in generations:
        price = _get_image_generation_price(generation)
        total += math.sqrt(price / _IMAGE_BASELINE_PRICE)
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
                    "image_generations": list[dict],  # per-generation resolution tiers
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
            # Prefer per-generation resolution-tier pricing; fall back to model
            # names for stats produced before tiered pricing (all baseline today,
            # since no image models carry a flat price).
            image_generations = stats.get("image_generations")
            if image_generations:
                image_model_mult = _compute_image_generation_multiplier(image_generations)
            else:
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


@dataclass
class GeneratorQualification:
    """Per-UID 7-day sample-weighted fool-rate gate for image and video."""

    image_n: int = 0
    image_fooled: int = 0
    video_n: int = 0
    video_fooled: int = 0
    qualified_image: bool = False
    qualified_video: bool = False

    @property
    def image_rate(self) -> Optional[float]:
        if self.image_n <= 0:
            return None
        return self.image_fooled / self.image_n

    @property
    def video_rate(self) -> Optional[float]:
        if self.video_n <= 0:
            return None
        return self.video_fooled / self.video_n


def _as_nonneg_int(value: Any) -> int:
    try:
        parsed = int(value) if value is not None else 0
    except (ValueError, TypeError):
        return 0
    return max(0, parsed)


def get_generator_qualification(
    generator_results,
    metagraph,
    image_fool_cutoff: float = 0.02,
    video_fool_cutoff: float = 0.01,
    min_fool_samples: int = 20,
) -> Dict[int, GeneratorQualification]:
    """Qualify generators on last-week sample-weighted fool rate, per modality.

    A modality clears when (fooled + not_fooled) >= min_fool_samples and
    fooled / n is strictly greater than that modality's cutoff. Counts come
    from benchmark evals (generator_result_benchmark), not answered challenges.
    Unknown or unregistered hotkeys are omitted; callers treat missing UIDs
    as onboarding.
    """
    if not generator_results:
        bt.logging.warning("No generator results data provided")
        return {}

    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}
    tallies: Dict[int, Dict[str, int]] = {}

    try:
        for result in generator_results:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue

            ss58_address = result.get("ss58_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            modality = str(result.get("modality") or "").strip().lower()
            if modality not in ("image", "video"):
                continue

            fooled = _as_nonneg_int(result.get("fooled_count", 0))
            not_fooled = _as_nonneg_int(result.get("not_fooled_count", 0))
            uid = ss58_to_uid[ss58_address]
            row = tallies.setdefault(
                uid, {"image_fooled": 0, "image_n": 0, "video_fooled": 0, "video_n": 0}
            )
            row[f"{modality}_fooled"] += fooled
            row[f"{modality}_n"] += fooled + not_fooled

        qualifications: Dict[int, GeneratorQualification] = {}
        n_image = n_video = 0
        for uid, row in tallies.items():
            image_n = row["image_n"]
            video_n = row["video_n"]
            image_fooled = row["image_fooled"]
            video_fooled = row["video_fooled"]
            image_rate = (image_fooled / image_n) if image_n else None
            video_rate = (video_fooled / video_n) if video_n else None
            qualified_image = (
                image_n >= min_fool_samples
                and image_rate is not None
                and image_rate > image_fool_cutoff
            )
            qualified_video = (
                video_n >= min_fool_samples
                and video_rate is not None
                and video_rate > video_fool_cutoff
            )
            qualifications[uid] = GeneratorQualification(
                image_n=image_n,
                image_fooled=image_fooled,
                video_n=video_n,
                video_fooled=video_fooled,
                qualified_image=qualified_image,
                qualified_video=qualified_video,
            )
            n_image += int(qualified_image)
            n_video += int(qualified_video)

        bt.logging.info(
            f"Qualified {n_image} image and {n_video} video generators "
            f"from {len(generator_results)} result rows "
            f"(cutoffs image>{image_fool_cutoff:.3f} video>{video_fool_cutoff:.3f}, "
            f"n>={min_fool_samples})"
        )
        return qualifications
    except Exception as e:
        bt.logging.error(f"Error processing generator qualification: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        return {}


def combine_generator_rewards(
    base_rewards: Dict[int, Dict[str, float]],
    qualification: Dict[int, GeneratorQualification],
    image_weight: float = 0.30,
    video_weight: float = 0.70,
) -> Dict[int, float]:
    """Pay only in modalities that cleared the fool-rate gate.

    R = 0.30 * R_image * I_image + 0.70 * R_video * I_video
    I_* is 1 if that modality is qualified, else 0. UIDs with R == 0 are omitted
    so they do not share the generator pot.
    """
    rewards: Dict[int, float] = {}
    for uid, base in base_rewards.items():
        q = qualification.get(uid)
        image_term = float(base.get("image", 0.0) or 0.0)
        video_term = float(base.get("video", 0.0) or 0.0)
        if q is None or not q.qualified_image:
            image_term = 0.0
        if q is None or not q.qualified_video:
            video_term = 0.0
        reward = image_weight * image_term + video_weight * video_term
        if reward > 0:
            rewards[int(uid)] = reward
    return rewards
