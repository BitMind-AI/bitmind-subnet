# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 BitMind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Dict, Tuple, Any
import bittensor as bt
import numpy as np


def compute_penalty_multiplier(y_pred: np.ndarray) -> float:
    """
    Compute penalty for predictions outside valid range.

    Args:
        y_pred (np.ndarray): Predicted probabilities for each class, shape (3,)

    Returns:
        float: 0.0 if prediction is invalid, 1.0 if valid
    """
    sum_check = np.abs(np.sum(y_pred) - 1.0) < 1e-6
    range_check = np.all((y_pred >= 0.0) & (y_pred <= 1.0))
    return 1.0 if (sum_check and range_check) else 0.0


def transform_rational(mcc, pole=1.01):
    return 1 / (pole - np.array(mcc))


def get_rewards(
    label: int,
    responses: List[np.ndarray],
    uids: List[int],
    axons: List[bt.axon],
    challenge_modality: str,
    performance_trackers: Dict[str, Any]
) -> Tuple[np.ndarray, List[Dict[str, Dict[str, float]]]]:
    """
    Calculate rewards for miner responses based on performance metrics.

    Args:
        label: The true label (0 for real, 1 for synthetic, 2 for semi-synthetic)
        responses: List of probability vectors from miners, each shape (3,)
        uids: List of miner UIDs
        axons: List of miner axons
        challenge_modality: Type of challenge ('video' or 'image')
        performance_trackers: Dict mapping modality to performance tracker

    Returns:
        Tuple containing:
            - np.ndarray: Array of rewards for each miner
            - List[Dict]: List of performance metrics for each miner
    """
    miner_rewards = []
    miner_metrics = []

    for axon, uid, pred_probs in zip(axons, uids, responses):
        miner_modality_rewards = {}
        miner_modality_metrics = {}

        for modality in ['image', 'video']:
            tracker = performance_trackers[modality]
            try:
                miner_hotkey = axon.hotkey

                if uid in tracker.miner_hotkeys and tracker.miner_hotkeys[uid] != miner_hotkey:
                    bt.logging.info(f"Miner hotkey changed for UID {uid}. Resetting performance metrics.")
                    tracker.reset_miner_history(uid, miner_hotkey)

                if modality == challenge_modality:
                    tracker.update(uid, pred_probs, label, miner_hotkey)

                metrics = tracker.get_metrics(uid, window=100)
                reward = (0.9 * metrics['binary_mcc'] + 0.1 * metrics['multi_class_mcc'])
                reward *= compute_penalty_multiplier(pred_probs)
                
                miner_modality_rewards[modality] = reward
                miner_modality_metrics[modality] = metrics

            except Exception as e:
                bt.logging.error(f"Couldn't calculate reward for miner {uid}, prediction: {pred_probs}, label: {label}")
                bt.logging.exception(e)
                miner_rewards.append(0.0)
                continue

        total_reward = (
            0.4 * miner_modality_rewards.get('video', 0.0) +
            0.6 * miner_modality_rewards.get('image', 0.0)
        )
        miner_rewards.append(total_reward)
        miner_metrics.append(miner_modality_metrics)

    return np.array(miner_rewards), miner_metrics
