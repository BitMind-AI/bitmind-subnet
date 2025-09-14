def get_discriminator_rewards(
    runs,
    metagraph,
    image_score_weight: float = 0.5,
    video_score_weight: float = 0.5,
    binary_score_weight: float = 0.5,
    multiclass_score_weight: float = 0.5,
):
    """
    Process benchmark runs to extract discriminator rewards from MCC scores.
    Handles separate image and video detection models per miner.

    Args:
        runs: List of benchmark run data from API
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

    for run in runs:
        if run.get("status") != "success":
            continue

        for result in run.get("results", []):
            ss58_address = result.get("ss58_address")
            if ss58_address not in ss58_to_uid:
                continue

            uid = ss58_to_uid[ss58_address]
            modality = result["modality"]

            if uid not in miner_modality_rewards:
                miner_modality_rewards[uid] = {}

            miner_modality_rewards[uid][modality] = binary_score_weight * max(
                0, result.get("binary_mcc", 0)
            ) + multiclass_score_weight * max(0, result.get("multiclass_mcc", 0))

    # Combine image and video rewards for each miner
    final_rewards = {}
    for uid, modality_rewards in miner_modality_rewards.items():
        final_rewards[uid] = image_score_weight * modality_rewards.get(
            "image", 0.0
        ) + video_score_weight * modality_rewards.get("video", 0.0)

    return final_rewards


def get_generator_base_rewards(verification_stats):
    """
    Compute base rewards for generators based on their verification pass rates.

    Args:
        verification_stats: Dict mapping hotkey to verification stats from
                            ContentManager.get_unrewarded_verification_stats()
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
            uid = stats["uid"]
            pass_rate = stats["pass_rate"]
            verified_count = stats["total_verified"]

            # Simple base reward calculation (can be customized)
            # Higher pass rate = higher reward, with bonus for volume
            base_reward = pass_rate * min(verified_count, 10)  # Cap volume bonus at 10

            uid_rewards[uid] = base_reward
            all_media_ids.extend(stats["media_ids"])

        import bittensor as bt

        bt.logging.info(f"Computed base rewards for {len(uid_rewards)} miners")

        return uid_rewards, all_media_ids

    except Exception as e:
        import bittensor as bt

        bt.logging.error(f"Error in get_generator_base_rewards: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        return {}, []


def get_generator_reward_multipliers(generator_results, metagraph):
    """
    Process generator results to extract rewards from fool rates.

    Args:
        generator_results: List of generator result data from API
        metagraph: Bittensor metagraph for SS58 to UID mapping

    Returns:
        dict: Mapping of UID to reward score for generators
    """
    rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    # Aggregate fool rates by generator (ss58_address)
    generator_scores = {}
    generator_counts = {}

    for result in generator_results:
        ss58_address = result.get("ss58_address")
        if ss58_address in ss58_to_uid:
            fool_rate = result.get("fool_rate", 0)

            if ss58_address not in generator_scores:
                generator_scores[ss58_address] = 0
                generator_counts[ss58_address] = 0

            generator_scores[ss58_address] += fool_rate
            generator_counts[ss58_address] += 1

    # Calculate average fool rate for each generator
    for ss58_address, total_score in generator_scores.items():
        if ss58_address in ss58_to_uid:
            uid = ss58_to_uid[ss58_address]
            avg_fool_rate = total_score / generator_counts[ss58_address]

            rewards[uid] = max(0, min(1.0, avg_fool_rate))

    return rewards
