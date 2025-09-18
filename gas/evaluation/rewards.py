import bittensor as bt


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
        generator_results: List of GeneratorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping

    Returns:
        dict: Mapping of UID to reward score for generators
    """
    import bittensor as bt
    
    rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    if not generator_results:
        bt.logging.warning("No generator results data provided")
        return {}

    # Aggregate fool rates by generator (ss58_address)
    generator_scores = {}
    generator_counts = {}

    try:
        for result in generator_results:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue                
            ss58_address = result.get("ss58_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            fool_rate = result.get("fool_rate", 0)
            try:
                fool_rate = float(fool_rate) if fool_rate is not None else 0
            except (ValueError, TypeError):
                bt.logging.warning(f"Invalid fool_rate for {ss58_address}: {fool_rate}")
                fool_rate = 0

            if ss58_address not in generator_scores:
                generator_scores[ss58_address] = 0
                generator_counts[ss58_address] = 0

            generator_scores[ss58_address] += fool_rate
            generator_counts[ss58_address] += 1

        # Calculate average fool rate for each generator
        for ss58_address, total_score in generator_scores.items():
            if ss58_address in ss58_to_uid:
                uid = ss58_to_uid[ss58_address]
                count = generator_counts[ss58_address]

                if count > 0:
                    avg_fool_rate = total_score / count
                    rewards[uid] = max(0, min(1.0, avg_fool_rate))
                else:
                    bt.logging.warning(f"Zero count for generator {ss58_address}")

        bt.logging.info(f"Processed {len(generator_results)} generator results, computed rewards for {len(rewards)} generators")

    except Exception as e:
        bt.logging.error(f"Error processing generator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    return rewards
