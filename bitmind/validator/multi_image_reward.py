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


import torch
from typing import List, Tuple
import bittensor as bt
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score
)


def reward(y_pred: np.array, y_true: np.array) -> Tuple[float, dict]:
    """
    Reward the miner response to the request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """

    preds = np.round(y_pred)
    #print(preds, '\n', y_true)
    # accuracy = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    ap_score = average_precision_score(y_true, y_pred)

    metrics = {
        'fp_score': 1 - fp / len(y_pred),
        #'precision': prec,
        #'recall': rec,
        'f1_score': f1,
        'ap_score': ap_score
    }
    # TODO: should we use some linear combination of these metrics
    reward_val = sum([v for v in metrics.values()]) / len(metrics)
    return reward_val, metrics


def count_penalty(y_pred: np.array) -> float:
    bad = np.any((y_pred < 0) | (y_pred > 1))
    return 0 if bad else 1


def get_rewards(
        labels: torch.FloatTensor,
        responses: List,
) -> Tuple[torch.FloatTensor, List[dict]]:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    miner_rewards = []
    miner_metrics = []

    for uid in range(len(responses)):
        try:
            if not responses[uid] or len(responses[uid]) != len(labels):
                miner_rewards.append(0)
                miner_metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})
                continue

            predictions_array = np.array(responses[uid])
            miner_reward, metrics = reward(predictions_array, labels)
            penalty = count_penalty(predictions_array)
            miner_reward *= penalty
            miner_rewards.append(miner_reward)
            metrics['penalty'] = penalty
            miner_metrics.append(metrics)

        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(
                uid, responses[uid], labels))

            bt.logging.exception(e)
            miner_rewards.append(0)
            miner_metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})

    return torch.FloatTensor(miner_rewards), miner_metrics
