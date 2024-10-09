# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import List
import bittensor as bt
import numpy as np


def compute_penalty(y_pred: float) -> float:
    bad = (y_pred < 0.0) or (y_pred > 1.0)
    return 0.0 if bad else 1.0


def get_rewards(
        label: float,
        responses: List[float],
        uids: List[int],
        axons: List[bt.axon],
        performance_tracker,
    ) -> np.array:
    """
    Returns an array of rewards for the given label and miner responses.

    Args:
    - label (float): The true label (1.0 for fake, 0.0 for real).
    - responses (List[float]): A list of responses from the miners.
    - uids (List[int]): List of miner UIDs.
    - axons (List[bt.axon]): List of miner axons.
    - performance_tracker (MinerPerformanceTracker): Tracks historical performance metrics per miner.

    Returns:
    - np.array: An array of rewards for the given label and responses.
    """
    miner_rewards = []
    miner_metrics = []
    for axon, uid, pred_prob in zip(axons, uids, responses):
        try:
            miner_hotkey = axon.hotkey
            if uid in performance_tracker.miner_hotkeys and performance_tracker.miner_hotkeys[uid] != miner_hotkey:
                bt.logging.info(f"Miner hotkey changed for UID {uid}. Resetting performance metrics.")
                performance_tracker.reset_miner_history(uid, miner_hotkey)

            performance_tracker.update(uid, pred_prob, label, miner_hotkey)
            metrics = performance_tracker.get_metrics(uid, window=100)
            reward = metrics['auc'] ** 2
            reward *= compute_penalty(pred_prob)

            miner_rewards.append(reward)
            miner_metrics.append(metrics)

        except Exception as e:
            bt.logging.error(f"Couldn't calculate reward for miner {uid}, prediction: {pred_prob}, label: {label}")
            bt.logging.exception(e)
            miner_rewards.append(0.0)

    return np.array(miner_rewards), miner_metrics