"""
Resolution tiers for video generation challenges.

Validators request a tier per video challenge and reward at the tier actually
delivered, capped at the tier requested: reward_tier = min(observed, requested).
Overshooting a request never pays more, and undershooting (model can't reach
the requested tier) degrades gracefully to the lower tier's price instead of
failing the challenge.

Tier strings match the OpenRouter /videos API resolution values ("720p",
"1080p", "4K") that the reference miner forwards to providers.
"""

import random
from typing import Optional, Tuple

# Lowest to highest. "480p" is a catch-all floor for sub-720p output; it is
# never requested in challenges but observed output can classify into it.
TIER_ORDER = ["480p", "720p", "1080p", "4K"]

_TIER_RANK = {tier: rank for rank, tier in enumerate(TIER_ORDER)}

# Weighted distribution for sampling the requested tier of a video challenge.
# The knob for dataset resolution mix and how often top-tier capability is probed.
CHALLENGE_TIER_WEIGHTS = {
    "720p": 0.35,
    "1080p": 0.45,
    "4K": 0.20,
}

# Classification thresholds on the *minimum* dimension so portrait and
# landscape orientations classify identically (e.g. 720x1280 and 1280x720 are
# both 720p). Thresholds sit below nominal values to tolerate provider
# variations like 704 or 2156-pixel encodes.
_TIER_MIN_DIM_THRESHOLDS = [
    ("4K", 2000),     # nominal 2160
    ("1080p", 1000),  # nominal 1080
    ("720p", 620),    # nominal 720
    ("480p", 0),
]


def sample_challenge_tier(rng: random.Random = random) -> str:
    """Sample a requested resolution tier from the challenge weight distribution."""
    tiers = list(CHALLENGE_TIER_WEIGHTS.keys())
    weights = list(CHALLENGE_TIER_WEIGHTS.values())
    return rng.choices(tiers, weights=weights, k=1)[0]


def normalize_tier(tier: Optional[str]) -> Optional[str]:
    """Map a tier string to its canonical form ("4k" -> "4K"), or None if unknown."""
    if not tier:
        return None
    for canonical in TIER_ORDER:
        if canonical.lower() == tier.strip().lower():
            return canonical
    return None


def tier_from_resolution(resolution: Optional[Tuple[int, int]]) -> Optional[str]:
    """Classify observed (width, height) pixel dimensions into a tier."""
    if not resolution or len(resolution) != 2:
        return None
    try:
        min_dim = min(int(resolution[0]), int(resolution[1]))
    except (TypeError, ValueError):
        return None
    if min_dim <= 0:
        return None
    for tier, threshold in _TIER_MIN_DIM_THRESHOLDS:
        if min_dim >= threshold:
            return tier
    return None


def effective_tier(
    observed_resolution: Optional[Tuple[int, int]],
    requested_tier: Optional[str],
) -> Optional[str]:
    """
    Tier a video generation is priced at: min(observed, requested).

    Returns None when the observed resolution is unknown — pricing then falls
    back to the baseline, since an unverifiable resolution earns no premium.
    A missing requested tier (e.g. outcomes recorded before tiered challenges
    shipped) prices at the observed tier alone.
    """
    observed = tier_from_resolution(observed_resolution)
    if observed is None:
        return None
    requested = normalize_tier(requested_tier)
    if requested is None:
        return observed
    return observed if _TIER_RANK[observed] <= _TIER_RANK[requested] else requested
