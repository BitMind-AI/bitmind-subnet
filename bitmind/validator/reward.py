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
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score
)


def count_penalty(y_pred: float) -> float:
    bad = (y_pred < 0) | (y_pred > 1)
    return 0. if bad else 1.


def get_rewards(
        label: float,
        responses: List,
    ) -> np.array:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - label (float): 1 if image was fake, 0 if real.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - np.array: A tensor of rewards for the given query and responses.
    """
    miner_rewards = []
    for uid in range(len(responses)):
        try:
            pred = responses[uid]
            reward = 1. if np.round(pred) == label else 0.
            reward *= count_penalty(pred)
            miner_rewards.append(reward)

        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(
                uid, responses[uid], label))
            bt.logging.exception(e)
            miner_rewards.append(0)

    return np.array(miner_rewards)
