from sklearn.metrics import f1_score
from typing import Dict, List
from collections import deque
import bittensor as bt
import numpy as np


class MinerPerformanceTracker:
    """
    Tracks all recent miner performance to facilitate reward computation.
    """
    def __init__(self, store_last_n_predictions: int = 100):
        self.prediction_history: Dict[int, deque] = {}
        self.label_history: Dict[int, deque] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.store_last_n_predictions = store_last_n_predictions

    def reset_miner_history(self, uid: int, miner_hotkey: str):
        """
        Reset the history for a miner.
        """
        self.prediction_history[uid] = deque(maxlen=self.store_last_n_predictions)
        self.label_history[uid] = deque(maxlen=self.store_last_n_predictions)
        self.miner_hotkeys[uid] = miner_hotkey

    def update(self, uid: int, prediction: np.ndarray, label: int, miner_hotkey: str):
        """
        Update the miner prediction history

        Args:
        - prediction: numpy array of shape (3,) containing probabilities for [real, synthetic, semi-synthetic]
        - label: integer label (0 for real, 1 for synthetic, 2 for semi-synthetic)
        """
        if uid not in self.prediction_history or self.miner_hotkeys.get(uid) != miner_hotkey:
            self.reset_miner_history(uid, miner_hotkey)

        self.prediction_history[uid].append(prediction)   # store full probability vector
        self.label_history[uid].append(label)

    def get_metrics(self, uid: int, window: int = None):
        """
        Get the performance metrics for a miner based on their last n predictions
        """
        if uid not in self.prediction_history:
            return self._empty_metrics()

        recent_preds = list(self.prediction_history[uid])
        recent_labels = list(self.label_history[uid])

        if window is not None:
            window = min(window, len(recent_preds))
            recent_preds = recent_preds[-window:]
            recent_labels = recent_labels[-window:]

        pred_probs = np.array([p for p in recent_preds if not np.array_equal(p, -1)])
        labels = np.array([l for i, l in enumerate(recent_labels) if not np.array_equal(recent_preds[i], -1)])

        if len(labels) == 0 or len(pred_probs) == 0:
            return self._empty_metrics()

        try:
            predictions = np.argmax(pred_probs, axis=1)

            # multiclass F1 (real vs synthetic vs semi-synthetic)
            multi_class_f1 = f1_score(labels, predictions, average='weighted')

            # binary f-1 (real vs any synthetic)
            binary_labels = (labels > 0).astype(int)
            binary_preds = (predictions > 0).astype(int)
            binary_f1 = f1_score(binary_labels, binary_preds)

            return {
                'multi_class_f1': multi_class_f1,
                'binary_f1': binary_f1
            }

        except Exception as e:
            bt.logging.warning(f'Error in reward computation: {e}')
            return self._empty_metrics()

    def _empty_metrics(self):
        """
        Return a dictionary of empty metrics
        """
        return {
            'multi_class_f1': 0,
            'binary_f1': 0
        }

    def get_prediction_count(self, uid: int) -> int:
        """
        Get the number of predictions made by a specific miner.
        """
        if uid not in self.prediction_history:
            return 0
        return len(self.prediction_history[uid])