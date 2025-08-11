from typing import Dict, List, Optional, Tuple
import bittensor as bt
import numpy as np
from sklearn.metrics import matthews_corrcoef

from gas.types import Modality, MinerType


def get_discriminator_rewards(
    label: int,
    predictions: List[np.ndarray],
    uids: List[int],
    hotkeys: List[bt.axon],
    challenge_modality: Modality,
    discriminator_tracker,
    window=200,
) -> Tuple[np.ndarray, List[Dict[Modality, Dict[str, float]]]]:
    """
    Calculate rewards for detection challenge

    Args:
        label: The true label (0 for real, 1 for synthetic, 2 for semi-synthetic)
        predictions: List of probability vectors from miners, each shape (3,)
        uids: List of miner UIDs
        axons: List of miner axons
        challenge_modality: Type of challenge (Modality.VIDEO or Modality.IMAGE)

    Returns:
        Tuple containing:
            - np.ndarray: Array of rewards for each miner
            - List[Dict]: List of performance metrics for each miner
    """
    miner_rewards = {}
    miner_metrics = {}

    for hotkey, uid, pred_probs in zip(hotkeys, uids, predictions):
        miner_modality_rewards = {}
        miner_modality_metrics = {}

        for modality in Modality:
            try:
                miner_modality_rewards[modality.value] = 0.0
                miner_modality_metrics[modality.value] = {"multi_class_mcc": 0, "binary_mcc": 0}

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

                pred_count = discriminator_tracker.get_prediction_count(uid).get(modality, 0)
                if pred_count < 5:
                    #bt.logging.trace(f"Not enough {modality} attempts ({pred_count} < 5) to compute rewards for uid {uid}")
                    continue

                recent_preds, recent_labels = discriminator_tracker.get_predictions_and_labels(uid, modality)

                if window is not None:
                    window = min(window, len(recent_preds))
                    recent_preds = recent_preds[-window:]
                    recent_labels = recent_labels[-window:]

                predictions = np.argmax(recent_preds, axis=1)

                # Multi-class MCC (real vs synthetic vs semi-synthetic)
                multi_class_mcc = matthews_corrcoef(recent_labels, predictions)

                # Binary MCC (real vs any synthetic)
                binary_labels = (recent_labels > 0).astype(int)
                binary_preds = (predictions > 0).astype(int)
                binary_mcc = matthews_corrcoef(binary_labels, binary_preds)

                # Penalize for invalid predictions by reducing MCC
                invalid_count = discriminator_tracker.get_invalid_prediction_count(uid, modality)
                total_attempts = pred_count + invalid_count
                invalid_penalty = invalid_count / total_attempts if total_attempts > 0 else 0

                binary_weight = 0.5  #self.config.scoring.binary_weight
                multiclass_weight = 0.5  #self.config.scoring.multiclass_weight
    
                miner_modality_rewards[modality.value] = (
                    binary_weight * max(0, binary_mcc) * (1 - invalid_penalty)
                    + multiclass_weight * max(0,  multi_class_mcc) * (1 - invalid_penalty)
                )
                miner_modality_metrics[modality.value]['binary_mcc'] = binary_mcc
                miner_modality_metrics[modality.value]['multi_class_mcc'] = multi_class_mcc
                miner_modality_metrics[modality.value]['sample_size'] = len(recent_preds)
                miner_modality_metrics[modality.value]['invalid_count'] = invalid_count
                miner_modality_metrics[modality.value]['total_attempts'] = total_attempts


            except Exception as e:
                bt.logging.error(
                    f"Couldn't compute detection rewards for miner {uid}, "
                    f"prediction: {pred_probs}, label: {label}, modality: {modality.value}"
                )
                bt.logging.exception(e)

        image_weight = 0.6 #self.config.scoring.image_weight
        video_weight = 0.4 #self.config.scoring.video_weight
        image_rewards = miner_modality_rewards.get(Modality.IMAGE, 0.0)
        video_rewards = miner_modality_rewards.get(Modality.VIDEO, 0.0)
        total_reward = (image_weight * image_rewards) + (
            video_weight * video_rewards
        )

        miner_rewards[uid] = total_reward
        miner_metrics[uid] = miner_modality_metrics

    return miner_rewards, miner_metrics
