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


def compute_penalty(y_pred: float) -> float:
    """
    Compute penalty for predictions outside valid range.

    Args:
        y_pred (float): Predicted value

    Returns:
        float: 0.0 if prediction is invalid, 1.0 if valid
    """
    bad = (y_pred < 0.0) or (y_pred > 1.0)
    return 0.0 if bad else 1.0


def get_rewards(
    label: float,
    responses: List[float],
    uids: List[int],
    axons: List[bt.axon],
    challenge_modality: str,
    performance_trackers: Dict[str, Any]
) -> Tuple[np.ndarray, List[Dict[str, Dict[str, float]]]]:
    """
    Calculate rewards for miner responses based on performance metrics.

    Args:
        label: The true label (1.0 for fake, 0.0 for real)
        responses: List of responses from the miners
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

    for axon, uid, pred_prob in zip(axons, uids, responses):
        miner_modality_rewards = {}
        miner_modality_metrics = {}

        for modality in ['image', 'video']:
            tracker = performance_trackers[modality]
            try:
                miner_hotkey = axon.hotkey

                tracked_hotkeys = tracker.miner_hotkeys
                if uid in tracked_hotkeys and tracked_hotkeys[uid] != miner_hotkey:
                    bt.logging.info(
                        f"Miner hotkey changed for UID {uid}. Resetting performance metrics."
                    )
                    tracker.reset_miner_history(uid, miner_hotkey)

                if modality == challenge_modality:
                    performance_trackers[modality].update(
                        uid, pred_prob, label, miner_hotkey
                    )

                metrics_100 = tracker.get_metrics(uid, window=100)
                metrics_10 = tracker.get_metrics(uid, window=10)
                reward = 0.5 * metrics_100['mcc'] + 0.5 * metrics_10['accuracy']
                reward *= compute_penalty(pred_prob)
                miner_modality_rewards[modality] = reward
                miner_modality_metrics[modality] = metrics_100

            except Exception as e:
                bt.logging.error(
                    f"Couldn't calculate reward for miner {uid}, "
                    f"prediction: {pred_prob}, label: {label}"
                )
                bt.logging.exception(e)
                miner_rewards.append(0.0)

        total_reward = (
            0.1 * miner_modality_rewards['video'] +
            0.9 * miner_modality_rewards['image']
        )
        miner_rewards.append(total_reward)
        miner_metrics.append(miner_modality_metrics)

    return np.array(miner_rewards), miner_metrics
