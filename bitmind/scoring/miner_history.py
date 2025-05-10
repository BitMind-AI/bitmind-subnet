from typing import Dict
from collections import deque
import bittensor as bt
import numpy as np
import joblib
import traceback
import os

from bitmind.types import Modality


class MinerHistory:
    """Tracks all recent miner performance to facilitate reward computation."""

    VERSION = 2

    def __init__(self, store_last_n_predictions: int = 100):
        self.predictions: Dict[int, Dict[Modality, deque]] = {}
        self.labels: Dict[int, Dict[Modality, deque]] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.store_last_n_predictions = store_last_n_predictions
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
        if uid not in self.miner_hotkeys or self.miner_hotkeys[uid] != miner_hotkey:
            self.reset_miner_history(uid, miner_hotkey)
            bt.logging.info(f"Reset history for {uid} {miner_hotkey}")

        self.predictions[uid][modality].append(np.array(prediction))
        self.labels[uid][modality].append(label)

    def _reset_predictions(self, uid: int):
        self.predictions[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n_predictions),
            Modality.VIDEO: deque(maxlen=self.store_last_n_predictions),
        }

    def _reset_labels(self, uid: int):
        self.labels[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n_predictions),
            Modality.VIDEO: deque(maxlen=self.store_last_n_predictions),
        }

    def reset_miner_history(self, uid: int, miner_hotkey: str):
        """Reset the history for a miner."""
        self._reset_predictions(uid)
        self._reset_labels(uid)
        self.miner_hotkeys[uid] = miner_hotkey

    def get_prediction_count(self, uid: int) -> int:
        """Get the number of predictions made by a specific miner."""
        counts = {}
        for modality in [Modality.IMAGE, Modality.VIDEO]:
            if uid not in self.predictions or modality not in self.predictions[uid]:
                counts[modality] = 0
            else:
                counts[modality] = len(self.predictions[uid][modality])
        return counts

    def save_state(self, save_dir):
        path = os.path.join(save_dir, "history.pkl")
        state = {
            "version": self.version,
            "store_last_n_predictions": self.store_last_n_predictions,
            "miner_hotkeys": self.miner_hotkeys,
            "predictions": self.predictions,
            "labels": self.labels,
        }
        joblib.dump(state, path)

    def load_state(self, save_dir):
        path = os.path.join(save_dir, "history.pkl")
        if not os.path.isfile(path):
            bt.logging.warning(f"No saved state found at {path}")
            return False

        try:
            state = joblib.load(path)
            if state["version"] != self.VERSION:
                bt.logging.warning(
                    f"Loading state from different version: {state['version']} != {self.VERSION}"
                )

            self.version = state["version"]
            self.store_last_n_predictions = state["store_last_n_predictions"]
            self.miner_hotkeys = state["miner_hotkeys"]
            self.predictions = state["predictions"]
            self.labels = state["labels"]
            bt.logging.debug(
                f"Successfully loaded history for {len(self.miner_hotkeys)} miners"
            )
            return True

        except Exception as e:
            bt.logging.error(f"Error deserializing MinerHistory state: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False
