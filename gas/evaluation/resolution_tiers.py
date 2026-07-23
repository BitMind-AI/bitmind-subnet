"""
Resolution tiers for generation challenges.

Validators request a tier per challenge and reward at the tier actually
delivered, capped at the tier requested: reward_tier = min(observed, requested).
Overshooting a request never pays more, and undershooting (model can't reach
the requested tier) degrades to the lower tier's price — discounted per tier
of shortfall for video (see rewards.VIDEO_UNDERSHOOT_TIER_DISCOUNT) — instead
of failing the challenge.

Video tiers ("480p", "720p", "1080p") match the OpenRouter /videos API
resolution values that the reference miner forwards to providers. 4K video is
deliberately not a tier for now: it is never requested, and 4K output
classifies (and prices) as 1080p.

Image tiers ("1K", "2K", "4K") refer to the minimum pixel dimension
(nominal 1024 / 2048 / 4096).
"""

import random
from typing import Dict, List, Optional, Tuple

VIDEO_MODALITY = "video"
IMAGE_MODALITY = "image"

# Lowest to highest, per modality.
TIER_ORDER: Dict[str, List[str]] = {
    VIDEO_MODALITY: ["480p", "720p", "1080p"],
    IMAGE_MODALITY: ["1K", "2K", "4K"],
}

_TIER_RANK: Dict[str, Dict[str, int]] = {
    modality: {tier: rank for rank, tier in enumerate(order)}
    for modality, order in TIER_ORDER.items()
}

# Weighted distributions for sampling the requested tier of a challenge.
# The knob for dataset resolution mix and how often top-tier capability is
# probed (top tiers are the most expensive for miners to serve).
CHALLENGE_TIER_WEIGHTS: Dict[str, Dict[str, float]] = {
    VIDEO_MODALITY: {
        "480p": 0.20,
        "720p": 0.40,
        "1080p": 0.40,
    },
    IMAGE_MODALITY: {
        "1K": 0.40,
        "2K": 0.40,
        "4K": 0.20,
    },
}

# Video classification thresholds on *total pixel count*, which is both
# orientation-invariant and aspect-ratio-invariant. Min-dimension thresholds
# were exploitable via square output: a 640x640 video has the pixel count
# (and provider cost) of 480p but the min dimension of a 720p portrait
# encode, so it classified — and was paid — as 720p. Pixel count matches how
# providers themselves bill resolution tiers. Thresholds sit at ~85% of the
# nominal 16:9 pixel count to tolerate provider variations like 704-pixel
# encodes. The lowest tier is a catch-all floor for undersized output.
_VIDEO_TIER_PIXEL_THRESHOLDS: List[Tuple[str, int]] = [
    ("1080p", 1_760_000),  # nominal 1920x1080 = 2,073,600; also absorbs 4K (no 4K tier yet)
    ("720p", 780_000),     # nominal 1280x720 = 921,600
    ("480p", 0),
]

# Image tiers are defined by minimum pixel dimension (nominal 1024/2048/4096),
# so min-dim classification is the definition, not a proxy. Thresholds sit
# below nominal to tolerate provider variations like 2156-pixel encodes.
_IMAGE_TIER_MIN_DIM_THRESHOLDS: List[Tuple[str, int]] = [
    ("4K", 3500),     # nominal 3840/4096
    ("2K", 1900),     # nominal 2048
    ("1K", 0),        # nominal 1024; catch-all for smaller output
]


def sample_challenge_tier(modality: str = VIDEO_MODALITY, rng: random.Random = random) -> str:
    """Sample a requested resolution tier from the modality's weight distribution."""
    weights_table = CHALLENGE_TIER_WEIGHTS[modality]
    tiers = list(weights_table.keys())
    weights = list(weights_table.values())
    return rng.choices(tiers, weights=weights, k=1)[0]


def normalize_tier(tier: Optional[str], modality: str = VIDEO_MODALITY) -> Optional[str]:
    """Map a tier string to its canonical form ("4k" -> "4K"), or None if unknown."""
    if not tier:
        return None
    for canonical in TIER_ORDER[modality]:
        if canonical.lower() == tier.strip().lower():
            return canonical
    return None


def tier_from_resolution(
    resolution: Optional[Tuple[int, int]], modality: str = VIDEO_MODALITY
) -> Optional[str]:
    """Classify observed (width, height) pixel dimensions into a tier.

    Video classifies by total pixel count; images by minimum dimension (their
    tiers are defined that way). See the threshold tables above for why.
    """
    if not resolution or len(resolution) != 2:
        return None
    try:
        width, height = int(resolution[0]), int(resolution[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if modality == VIDEO_MODALITY:
        pixels = width * height
        for tier, threshold in _VIDEO_TIER_PIXEL_THRESHOLDS:
            if pixels >= threshold:
                return tier
    else:
        min_dim = min(width, height)
        for tier, threshold in _IMAGE_TIER_MIN_DIM_THRESHOLDS:
            if min_dim >= threshold:
                return tier
    return None


def tier_shortfall(
    observed_resolution: Optional[Tuple[int, int]],
    requested_tier: Optional[str],
    modality: str = VIDEO_MODALITY,
) -> int:
    """How many tiers below the requested tier the delivery landed.

    0 when the request was met or exceeded, or when either side is unknown
    (unknown observed resolutions already price at the baseline, and outcomes
    recorded before tiered challenges have no requested tier).
    """
    observed = tier_from_resolution(observed_resolution, modality)
    requested = normalize_tier(requested_tier, modality)
    if observed is None or requested is None:
        return 0
    rank = _TIER_RANK[modality]
    return max(0, rank[requested] - rank[observed])


def effective_tier(
    observed_resolution: Optional[Tuple[int, int]],
    requested_tier: Optional[str],
    modality: str = VIDEO_MODALITY,
) -> Optional[str]:
    """
    Tier a generation is priced at: min(observed, requested).

    Returns None when the observed resolution is unknown — pricing then falls
    back to the baseline, since an unverifiable resolution earns no premium.
    A missing requested tier (e.g. outcomes recorded before tiered challenges
    shipped) prices at the observed tier alone.
    """
    observed = tier_from_resolution(observed_resolution, modality)
    if observed is None:
        return None
    requested = normalize_tier(requested_tier, modality)
    if requested is None:
        return observed
    rank = _TIER_RANK[modality]
    return observed if rank[observed] <= rank[requested] else requested
