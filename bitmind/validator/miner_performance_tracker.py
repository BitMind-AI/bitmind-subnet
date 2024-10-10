from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
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

    def update(self, uid: int, prediction: int, label: int, miner_hotkey: str):
        """
        Update the miner prediction history
        """
        # Reset histories if miner is new or miner address has changed
        if uid not in self.prediction_history or self.miner_hotkeys.get(uid) != miner_hotkey:
            self.reset_miner_history(uid, miner_hotkey)

        # Update histories
        self.prediction_history[uid].append(prediction)
        self.label_history[uid].append(label)

    def get_metrics(self, uid: int, window: int = None):
        """
        Get the performance metrics for a miner based on their last n predictions

        Args:
        - uid (int): The unique identifier of the miner
        - window (int, optional): The number of recent predictions to consider. 
        - If None, all stored predictions are used.

        Returns:
        - dict: A dictionary containing various performance metrics
        """
        if uid not in self.prediction_history:
            return self._empty_metrics()

        recent_preds = list(self.prediction_history[uid])
        recent_labels = list(self.label_history[uid])

        # If window is larger than available data, use all available data
        if window is not None:
            window = min(window, len(recent_preds))
            recent_preds = recent_preds[-window:]
            recent_labels = recent_labels[-window:]

        keep_idx = [i for i, p in enumerate(recent_preds) if p != -1]
        pred_probs = np.array([recent_preds[i] for i in keep_idx])
        predictions = np.round(pred_probs)
        labels = np.array([recent_labels[i] for i in keep_idx])

        if len(labels) == 0 or len(predictions) == 0:
            return self._empty_metrics()

        try:
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            mcc = matthews_corrcoef(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
            auc = roc_auc_score(labels, pred_probs) if len(np.unique(labels)) > 1 else 0.0
        except Exception as e:
            bt.logging.warning(f'Error in reward computation: {e}')
            return self._empty_metrics()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'auc': auc
        }

    def _empty_metrics(self):
        """
        Return a dictionary of empty metrics
        """
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'mcc': 0,
            'auc': 0
        }

    def get_prediction_count(self, uid: int) -> int:
        """
        Get the number of predictions made by a specific miner.

        Args:
        - uid (int): The unique identifier of the miner

        Returns:
        - int: The number of predictions made by the miner
        """
        if uid not in self.prediction_history:
            return 0
        return len(self.prediction_history[uid])