from typing import Dict
from collections import deque
import bittensor as bt
import numpy as np
import joblib
import traceback
import os

from bitmind.types import Modality


class MinerHistory:
    """Tracks all recent miner performance to facilitate reward computation.
    Will be replaced with Redis in a future release """

    VERSION = 2

    def __init__(self, store_last_n_predictions: int = 200):
        self.predictions: Dict[int, Dict[Modality, deque]] = {}
        self.labels: Dict[int, Dict[Modality, deque]] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.health: Dict[int: int] = {}
        self.store_last_n_predictions = store_last_n_predictions
        self.version = self.VERSION

    def update(
        self,
        uid: int,
        prediction: np.ndarray,
        error: str,
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

        if not error:
            self.predictions[uid][modality].append(np.array(prediction))
            self.labels[uid][modality].append(label)
            self.health[uid] = 1 
        else:
            self.health[uid] = 0

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
        self._reset_predictions(uid)
        self._reset_labels(uid)
        self.miner_hotkeys[uid] = miner_hotkey

    def get_prediction_count(self, uid: int) -> int:
        counts = {}
        for modality in [Modality.IMAGE, Modality.VIDEO]:
            counts[modality] = len(self.get_recent_predictions_and_labels(uid, modality)[0])
        return counts

    def get_recent_predictions_and_labels(self, uid, modality):
        if uid not in self.predictions or modality not in self.predictions[uid]:
            return [], []
        valid_indices = [
            i for i, p in enumerate(self.predictions[uid][modality])
            if p is not None and (isinstance(p, (list, np.ndarray)) and not np.any(p == None))
        ]
        valid_preds = np.array([
            p for i, p in enumerate(self.predictions[uid][modality]) if i in valid_indices
        ])
        labels_with_valid_preds = np.array([
            p for i, p in enumerate(self.labels[uid][modality]) if i in valid_indices
        ])
        return valid_preds, labels_with_valid_preds

    def get_healthy_miner_uids(self) -> list:
        return [uid for uid, healthy in self.health.items() if healthy]

    def get_unhealthy_miner_uids(self) -> list:
        return [uid for uid, healthy in self.health.items() if not healthy]

    def save_state(self, save_dir):
        path = os.path.join(save_dir, "history.pkl")
        state = {
            "version": self.version,
            "store_last_n_predictions": self.store_last_n_predictions,
            "miner_hotkeys": self.miner_hotkeys,
            "predictions": self.predictions,
            "labels": self.labels,
            "health": self.health
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

            self.version = state.get("version", self.VERSION)
            self.store_last_n_predictions = state.get("store_last_n_predictions", self.store_last_n_predictions)
            self.miner_hotkeys = state.get("miner_hotkeys", self.miner_hotkeys)
            self.predictions = state.get("predictions", self.predictions)
            self.labels = state.get("labels", self.labels)
            self.health = state.get("health", self.health)

            if len(self.miner_hotkeys) == 0:
                bt.logging.warning("Loaded state has no miner hotkeys")
            if len(self.predictions) == 0:
                bt.logging.warning("Loaded state has no predictions")
            if len(self.labels) == 0:
                bt.logging.warning("Loaded state has no labels")
            if len(self.health) == 0:
                bt.logging.warning("Loaded state has no health records")

            bt.logging.debug(
                f"Successfully loaded history for {len(self.miner_hotkeys)} miners"
            )
            return True

        except Exception as e:
            bt.logging.error(f"Error deserializing MinerHistory state: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False
