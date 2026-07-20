"""
Resolution tiers for generation challenges.

Validators request a tier per challenge and reward at the tier actually
delivered, capped at the tier requested: reward_tier = min(observed, requested).
Overshooting a request never pays more, and undershooting (model can't reach
the requested tier) degrades gracefully to the lower tier's price instead of
failing the challenge.

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

# Classification thresholds on the *minimum* dimension so portrait and
# landscape orientations classify identically (e.g. 720x1280 and 1280x720 are
# both 720p). Thresholds sit below nominal values to tolerate provider
# variations like 704 or 2156-pixel encodes. The lowest tier is a catch-all
# floor for undersized output.
_TIER_MIN_DIM_THRESHOLDS: Dict[str, List[Tuple[str, int]]] = {
    VIDEO_MODALITY: [
        ("1080p", 1000),  # nominal 1080; also absorbs 4K output (no 4K tier yet)
        ("720p", 620),    # nominal 720
        ("480p", 0),
    ],
    IMAGE_MODALITY: [
        ("4K", 3500),     # nominal 3840/4096
        ("2K", 1900),     # nominal 2048
        ("1K", 0),        # nominal 1024; catch-all for smaller output
    ],
}


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
    """Classify observed (width, height) pixel dimensions into a tier."""
    if not resolution or len(resolution) != 2:
        return None
    try:
        min_dim = min(int(resolution[0]), int(resolution[1]))
    except (TypeError, ValueError):
        return None
    if min_dim <= 0:
        return None
    for tier, threshold in _TIER_MIN_DIM_THRESHOLDS[modality]:
        if min_dim >= threshold:
            return tier
    return None


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
