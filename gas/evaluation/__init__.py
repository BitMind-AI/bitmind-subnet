from .challenge_allocation import (
    allocate_challenge_slots,
    classify_modality_bucket,
)
from .miner_type_tracker import MinerTypeTracker
from .rewards import (
    GeneratorQualification,
    combine_generator_rewards,
    get_generator_base_rewards,
    get_generator_qualification,
    resolve_generator_qualification,
)

__all__ = [
    "GenerativeChallengeManager",
    "MinerTypeTracker",
    "GeneratorQualification",
    "allocate_challenge_slots",
    "classify_modality_bucket",
    "combine_generator_rewards",
    "get_generator_base_rewards",
    "get_generator_qualification",
    "resolve_generator_qualification",
]


def __getattr__(name):
    # GenerativeChallengeManager pulls C2PA/CLIP (torch). CI installs no GPU
    # stack, so keep that import lazy for unit tests of rewards/allocation.
    if name == "GenerativeChallengeManager":
        from .generative_challenge_manager import GenerativeChallengeManager

        return GenerativeChallengeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
