from typing import Dict, List, Optional, Tuple
import bittensor as bt
import numpy as np
from sklearn.metrics import matthews_corrcoef

from gas.types import Modality, MinerType


def get_discriminator_rewards(
    label: int,
    predictions: List[np.ndarray],
    uids: List[int], 
    hotkeys: List[str],
    challenge_modality: Modality,
    discriminator_tracker: 'DiscriminatorTracker',
    window: Optional[int] = 200,
    image_score_weight: float = 0.6,
    video_score_weight: float = 0.4,
    binary_score_weight: float = 0.5,
    multiclass_score_weight: float = 0.5
) -> Tuple[Dict[int, float], Dict[int, Dict[str, Dict[str, float]]], Dict[int, bool]]:
    """
    Calculate rewards for detection challenge

    Args:
        label: The true label (0 for real, 1 for synthetic, 2 for semi-synthetic)
        predictions: List of probability vectors from miners, each shape (3,)
        uids: List of miner UIDs
        hotkeys: List of miner hotkeys
        challenge_modality: Type of challenge (Modality.VIDEO or Modality.IMAGE)
        discriminator_tracker: Tracker object for storing miner predictions
        window: Number of recent predictions to consider, defaults to 200
        image_score_weight: Weight for image modality rewards, defaults to 0.6
        video_score_weight: Weight for video modality rewards, defaults to 0.4
        binary_score_weight: Weight for binary classification rewards, defaults to 0.5
        multiclass_score_weight: Weight for multiclass classification rewards, defaults to 0.5

    Returns:
        Dict containing:
            - rewards: Dict mapping UIDs to reward values
            - metrics: Dict mapping UIDs to modality metrics
            - correct: Dict mapping UIDs to correctness booleans
    """
    miner_predictions = {}
    miner_rewards = {}
    miner_metrics = {}
    miner_correct = {}

    for hotkey, uid, pred_probs in zip(hotkeys, uids, predictions):
        miner_modality_rewards = {}
        miner_modality_metrics = {}

        for modality in Modality:
            try:
                # update tracker with prediction for current challenge for appropriate modality
                if modality == challenge_modality:
                    if pred_probs is not None and isinstance(pred_probs[0], (list, tuple, np.ndarray)):
                        pred_probs = pred_probs[0]

                    discriminator_tracker.update(
                        uid=uid,
                        prediction=pred_probs,
                        label=label,
                        modality=challenge_modality,
                        miner_hotkey=hotkey,
                    )

                # get current valid/invalid prediction counts, skip if there are no valid preds
                invalid_count = discriminator_tracker.get_invalid_prediction_count(uid, modality, window)
                preds, labels = discriminator_tracker.get_predictions_and_labels(uid, modality, window)
                if not len(preds) or not len(labels):
                    continue

                # take argmax of stored probabilities 
                predictions = np.argmax(preds, axis=1)

                # compute correctness of last prediction if evaluating current challenge modality
                if modality == challenge_modality:
                    miner_correct[uid] = predictions[-1] == label
                    miner_predictions[uid] = int(predictions[-1])

                # Always compute MCC for both modalities if > 5 preds made with > 2 unique labels
                pred_count = len(preds)
                if pred_count < 5 or len(np.unique(labels)) < 2:
                    continue

                # multi-class MCC (real vs synthetic vs semi-synthetic)
                multi_class_mcc = matthews_corrcoef(labels, predictions)

                # Binary MCC (real vs any synthetic)
                binary_labels = (labels > 0).astype(int)
                binary_preds = (predictions > 0).astype(int)
                binary_mcc = matthews_corrcoef(binary_labels, binary_preds)

                # penalize for invalid predictions by reducing MCC
                total_attempts = pred_count + invalid_count
                invalid_penalty = invalid_count / total_attempts if total_attempts > 0 else 0

                modality_reward = (
                    binary_score_weight * max(0, binary_mcc) * (1 - invalid_penalty)
                    + multiclass_score_weight * max(0,  multi_class_mcc) * (1 - invalid_penalty)
                )

                miner_modality_rewards[modality.value] = modality_reward
                miner_modality_metrics[modality.value] = {
                    'binary_mcc': binary_mcc,
                    'multi_class_mcc': multi_class_mcc,
                    'sample_size': len(preds),
                    'invalid_count': invalid_count,
                    'total_attempts': total_attempts
                }
 
            except Exception as e:
                bt.logging.error(
                    f"Couldn't compute detection rewards for miner {uid}, "
                    f"prediction: {pred_probs}, label: {label}, modality: {modality.value}"
                )
                bt.logging.exception(e)

        image_rewards = miner_modality_rewards.get(Modality.IMAGE, 0.0)
        video_rewards = miner_modality_rewards.get(Modality.VIDEO, 0.0)
        total_reward = (
            image_score_weight * image_rewards + 
            video_score_weight * video_rewards
        )

        miner_rewards[uid] = total_reward
        miner_metrics[uid] = miner_modality_metrics

    return {
       'rewards': miner_rewards, 
       'metrics': miner_metrics, 
       'correct': miner_correct,
       'predictions': miner_predictions
    }