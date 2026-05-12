from .rewards import (
    artifact_stats_with_uids,
    get_discriminator_rewards,
    get_captioner_rewards,
    get_encoder_rewards,
    get_generator_base_rewards,
    get_generator_reward_multipliers,
    normalize_rewards_to_weight_budget,
)


def __getattr__(name):
    if name == "ArtifactTaskManager":
        from .artifact_task_manager import ArtifactTaskManager

        return ArtifactTaskManager
    if name == "GenerativeChallengeManager":
        from .generative_challenge_manager import GenerativeChallengeManager

        return GenerativeChallengeManager
    if name == "MinerTypeTracker":
        from .miner_type_tracker import MinerTypeTracker

        return MinerTypeTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArtifactTaskManager",
    "GenerativeChallengeManager",
    "MinerTypeTracker",
    "artifact_stats_with_uids",
    "get_discriminator_rewards",
    "get_captioner_rewards",
    "get_encoder_rewards",
    "get_generator_base_rewards",
    "get_generator_reward_multipliers",
    "normalize_rewards_to_weight_budget",
]
