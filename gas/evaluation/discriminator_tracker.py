from typing import Dict
from collections import deque
import bittensor as bt
import numpy as np
import joblib
import traceback
import os

from gas.types import Modality


class DiscriminatorTracker:
    """Tracks all recent discriminator results to facilitate reward computation"""

    VERSION = 4

    def __init__(self, store_last_n: int = 200):
        self.predictions: Dict[int, Dict[Modality, deque]] = {}
        self.labels: Dict[int, Dict[Modality, deque]] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.store_last_n = store_last_n
        self.version = self.VERSION

    def update(
        self,
        uid: int,
        prediction: np.ndarray,
        label: int,
        modality: Modality,
        miner_hotkey: str,
    ):
        """Update the miner prediction history.

        Args:
            prediction: numpy array of shape (3,) containing probabilities for
                [real, synthetic, semi-synthetic]
            label: integer label (0 for real, 1 for synthetic, 2 for semi-synthetic)
        """
        if (
            uid not in self.miner_hotkeys
            or self.miner_hotkeys.get(uid) != miner_hotkey
        ):
            self.reset_miner_history(uid, miner_hotkey)
            bt.logging.info(f"Reset history for {uid} {miner_hotkey}")

        self.predictions[uid][modality].append(np.array(prediction))
        self.labels[uid][modality].append(label)
  
    def reset_miner_history(self, uid: int, miner_hotkey: str):

        self.miner_hotkeys[uid] = miner_hotkey

        self.labels[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n),
            Modality.VIDEO: deque(maxlen=self.store_last_n),
        }
        self.predictions[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n),
            Modality.VIDEO: deque(maxlen=self.store_last_n),
        }

    def get_prediction_count(self, uid: int) -> int:
        return {
            modality: len(self.get_predictions_and_labels(uid, modality)[0])
             for modality in [Modality.IMAGE, Modality.VIDEO]
        }

    def get_predictions_and_labels(self, uid, modality, window=None):
        if uid not in self.predictions or modality not in self.predictions[uid]:
            return [], []
            
        predictions = list(self.predictions[uid][modality])
        labels = list(self.labels[uid][modality])

        if window is not None:
            window = min(window, len(predictions))
            predictions = predictions[-window:]
            labels = labels[-window:]
        
        valid_indices = [
            i for i, p in enumerate(predictions)
            if p is not None and (isinstance(p, (list, np.ndarray)) and not np.any(p == None))
        ]
        predictions = [predictions[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        return np.array(predictions), np.array(labels)
        
    def get_invalid_prediction_count(self, uid, modality, window=None):
        if uid not in self.predictions or modality not in self.predictions[uid]:
            return 0
            
        predictions = list(self.predictions[uid][modality])
        if window is not None:
            window = min(window, len(predictions))
            predictions = predictions[-window:]

        none_count = sum(
            1 for p in predictions 
            if p is None or (isinstance(p, (list, np.ndarray)) and np.any(p == None))
        )
        return none_count

    def save_state(self, save_dir, filename="history.pkl"):
        path = os.path.join(save_dir, filename)
        state = {
            "version": self.version,
            "store_last_n": self.store_last_n,
            "miner_hotkeys": self.miner_hotkeys,
            "predictions": self.predictions,
            "labels": self.labels,
        }
        joblib.dump(state, path)

    def load_state(self, save_dir, filename="history.pkl"):
        path = os.path.join(save_dir, filename)
        if not os.path.isfile(path):
            bt.logging.warning(f"No saved state found at {path}")
            return False

        try:
            state = joblib.load(path)
            if state["version"] != self.VERSION:
                bt.logging.warning(
                    f"Loading state from different version: {state['version']} != {self.VERSION}"
                )

            attributes = [
                'version', 
                'store_last_n', 
                'miner_hotkeys',
                'predictions', 
                'labels', 
            ]

            for attr in attributes:
                default = self.VERSION if attr == 'version' else getattr(self, attr)
                setattr(self, attr, state.get(attr, default))

            collections = {
                'miner_hotkeys': 'miner hotkeys',
                'predictions': 'predictions',
                'labels': 'labels',
            }

            for attr, desc in collections.items():
                if len(getattr(self, attr)) == 0:
                    bt.logging.warning(f"Loaded state has no {desc}")

            bt.logging.debug(
                f"Successfully loaded history for {len(self.miner_hotkeys)} miners"
            )
            return True

        except Exception as e:
            bt.logging.error(f"Error deserializing MinerHistory state: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False

    def clear_miner_predictions(self, uid: int):
        """Reset specific miner to recover from inconsistency in validator state"""
        if uid in self.predictions:
            self.predictions[uid] = {
                Modality.IMAGE: deque(maxlen=self.store_last_n),
                Modality.VIDEO: deque(maxlen=self.store_last_n),
            }
        if uid in self.labels:
            self.labels[uid] = {
                Modality.IMAGE: deque(maxlen=self.store_last_n),
                Modality.VIDEO: deque(maxlen=self.store_last_n),
            }

        bt.logging.info(f"Cleared prediction history for miner {uid} due to data inconsistency")