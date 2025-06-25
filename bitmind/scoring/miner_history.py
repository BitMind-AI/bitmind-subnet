from typing import Dict
from collections import deque
import bittensor as bt
import numpy as np
import joblib
import traceback
import os

from bitmind.types import Modality, MinerType


class MinerHistory:
    """Tracks all recent miner performance to facilitate reward computation.
    Will be replaced with Redis in a future release """

    VERSION = 3

    def __init__(self, store_last_n: int = 200):
        self.segmentation_scores: Dict[int, Dict[Modality, deque]] = {}
        self.predictions: Dict[int, Dict[Modality, deque]] = {}
        self.labels: Dict[int, Dict[Modality, deque]] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.health: Dict[int: int] = {}
        self.miner_types: Dict[int: MinerType] = {}
        self.store_last_n = store_last_n
        self.version = self.VERSION

    def update(
        self,
        uid: int,
        segmentation_score: float,
        prediction: np.ndarray,
        label: int,
        error: str,
        modality: Modality,
        miner_hotkey: str,
        miner_type: str,
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
            or self.get_miner_type(uid) != miner_type
            or (miner_type == MinerType.SEGMENTER and uid not in self.segmentation_scores) \
            or (miner_type == MinerType.DETECTOR and uid not in self.predictions)
        ):
            self.reset_miner_history(uid, miner_hotkey, miner_type)
            bt.logging.info(f"Reset history for {uid} {miner_hotkey}")

        if miner_type == MinerType.SEGMENTER and modality != Modality.IMAGE:
            bt.logging.warning(f"SEGMENTER miner {uid} received unsupported modality {modality}, skipping update")
            return

        if not error:
            self.health[uid] = 1 
            if miner_type == MinerType.DETECTOR:
                self.predictions[uid][modality].append(np.array(prediction))
                self.labels[uid][modality].append(label)
            elif miner_type == MinerType.SEGMENTER:
                self.segmentation_scores[uid][modality].append(segmentation_score)
        else:
            self.health[uid] = 0

    def reset_miner_history(self, uid: int, miner_hotkey: str, miner_type: str):

        self.miner_hotkeys[uid] = miner_hotkey
        self.miner_types[uid] = miner_type

        ## for classification 
        self.labels[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n),
            Modality.VIDEO: deque(maxlen=self.store_last_n),
        }
        self.predictions[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n),
            Modality.VIDEO: deque(maxlen=self.store_last_n),
        }

        ## for segmentation
        self.segmentation_scores[uid] = {
            Modality.IMAGE: deque(maxlen=self.store_last_n),
        }

    def get_miner_type(self, uid: int) -> MinerType:
        return self.miner_types.get(uid)

    def get_prediction_count(self, uid: int) -> int:
        counts = {}
        miner_type = self.get_miner_type(uid)
        
        if miner_type == MinerType.DETECTOR:
            # DETECTOR supports both IMAGE and VIDEO
            for modality in [Modality.IMAGE, Modality.VIDEO]:
                counts[modality] = len(self.get_predictions_and_labels(uid, modality)[0])
        elif miner_type == MinerType.SEGMENTER:
            # SEGMENTER only supports IMAGE
            counts[Modality.IMAGE] = len(self.get_segmentation_scores(uid, Modality.IMAGE))
        
        return counts

    def get_segmentation_scores(self, uid, modality):
        if uid in self.segmentation_scores:
            return np.array([
                p for p in self.segmentation_scores[uid].get(modality, []) if p is not None
            ])
        return np.array([])

    def get_predictions_and_labels(self, uid, modality):
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
            "store_last_n": self.store_last_n,
            "miner_hotkeys": self.miner_hotkeys,
            "miner_types": self.miner_types,
            "segmentation_scores": self.segmentation_scores,
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

            attributes = [
                'version', 'store_last_n', 'miner_hotkeys', 'miner_types',
                'segmentation_scores', 'predictions', 'labels', 'health'
            ]

            for attr in attributes:
                default = self.VERSION if attr == 'version' else getattr(self, attr)
                setattr(self, attr, state.get(attr, default))

            collections = {
                'miner_hotkeys': 'miner hotkeys',
                'predictions': 'predictions',
                'labels': 'labels',
                'health': 'health records'
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
        if uid in self.segmentation_scores:
            self.segmentation_scores[uid] = {
                Modality.IMAGE: deque(maxlen=self.store_last_n),
            }
        bt.logging.info(f"Cleared prediction history for miner {uid} due to data inconsistency")
