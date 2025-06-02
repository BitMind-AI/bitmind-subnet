from typing import List, Dict, Tuple, Any, Optional
import bittensor as bt
import numpy as np
import json
from sklearn.metrics import matthews_corrcoef
import os

from bitmind.types import Modality
from bitmind.scoring.miner_history import MinerHistory


class EvalEngine:
    """
    A class to track rewards and compute weights for miners based on their
    prediction performance.
    """

    def __init__(
        self,
        metagraph: bt.metagraph,
        config: bt.config,
    ):
        assert config.neuron.full_path
        assert (
            abs(config.scoring.image_weight + config.scoring.video_weight - 1.0) < 1e-6
        ), "Modality weights must sum to 1.0"
        assert (
            abs(config.scoring.binary_weight + config.scoring.multiclass_weight - 1.0)
            < 1e-6
        ), "Binary/Multiclass weights must sum to 1.0"

        self.metagraph = metagraph
        self.config = config
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.tracker = MinerHistory()
        self.miner_metrics = {}

    def get_weights(self):
        """Returns an L1 normalized vector of scores (rewards EMA)."""

        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        normed_weights = self.scores / norm
        # uncomment to burn emissions 
        #normed_weights = np.array([v * 0.6 for v in normed_weights])
        #normed_weights[135] = 0.4
        bt.logging.debug(normed_weights)
        return normed_weights

    def score_challenge(
        self,
        challenge_results: dict,
        label: int,
        challenge_modality: Modality,
    ) -> Tuple[np.ndarray, List[Dict[Modality, Dict[str, float]]]]:
        """Update miner prediction history, compute instantaneous rewards, update score EMA"""

        predictions = [np.array(r["prediction"]) for r in challenge_results]
        hotkeys = [r["hotkey"] for r in challenge_results]
        uids = [r["uid"] for r in challenge_results]

        rewards = self._get_rewards_for_challenge(
            label, predictions, uids, hotkeys, challenge_modality
        )
        self._update_scores(rewards)
        return rewards

    def _update_scores(self, rewards: dict):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        uids = list(rewards.keys())
        rewards = np.array([rewards[uid] for uid in uids])
        bt.logging.trace(f"updating scores {uids} : {rewards}")

        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = np.nan_to_num(rewards, nan=0)

        rewards = np.asarray(rewards)
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        self.maybe_extend_scores(np.max(uids) + 1)
        scattered_rewards: np.ndarray = np.full_like(self.scores, 0.5)
        vali_uids = [
            uid
            for uid in range(len(scattered_rewards))
            if self.metagraph.validator_permit[uid]
            and self.metagraph.S[uid] > self.config.vpermit_tao_limit
        ]
        no_response_uids = [
            uid
            for uid in range(len(scattered_rewards))
            if all(
                [
                    count == 0
                    for modality, count in self.tracker.get_prediction_count(
                        uid
                    ).items()
                ]
            )
        ]
        scattered_rewards[vali_uids] = 0.0
        scattered_rewards[no_response_uids] = 0.0
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {scattered_rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.scoring.moving_average_alpha
        self.scores: np.ndarray = alpha * scattered_rewards + (1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def _get_rewards_for_challenge(
        self,
        label: int,
        responses: List[np.ndarray],
        uids: List[int],
        hotkeys: List[bt.axon],
        challenge_modality: Modality,
    ) -> Tuple[np.ndarray, List[Dict[Modality, Dict[str, float]]]]:
        """
        Calculate rewards for miner responses based on performance metrics.

        Args:
            label: The true label (0 for real, 1 for synthetic, 2 for semi-synthetic)
            responses: List of probability vectors from miners, each shape (3,)
            uids: List of miner UIDs
            axons: List of miner axons
            challenge_modality: Type of challenge (Modality.VIDEO or Modality.IMAGE)

        Returns:
            Tuple containing:
                - np.ndarray: Array of rewards for each miner
                - List[Dict]: List of performance metrics for each miner
        """
        miner_rewards = {}
        for hotkey, uid, pred_probs in zip(hotkeys, uids, responses):
            miner_modality_rewards = {}
            miner_modality_metrics = {}

            self.tracker.update(uid, pred_probs, label, challenge_modality, hotkey)

            for modality in Modality:
                try:
                    modality = modality.value
                    pred_count = self.tracker.get_prediction_count(uid).get(modality, 0)
                    if pred_count < 5:
                        miner_modality_rewards[modality] = 0.0
                        miner_modality_metrics[modality] = self._empty_metrics()
                        continue

                    metrics = self._get_metrics(uid, modality, window=100)

                    binary_weight = self.config.scoring.binary_weight
                    multiclass_weight = self.config.scoring.multiclass_weight
                    reward = (
                        binary_weight * metrics["binary_mcc"]
                        + multiclass_weight * metrics["multi_class_mcc"]
                    )

                    if modality == challenge_modality:
                        reward *= self.compute_penalty_multiplier(pred_probs)

                    miner_modality_rewards[modality] = reward
                    miner_modality_metrics[modality] = metrics

                except Exception as e:
                    bt.logging.error(
                        f"Couldn't calculate reward for miner {uid}, "
                        f"prediction: {pred_probs}, label: {label}, modality: {modality}"
                    )
                    bt.logging.exception(e)
                    miner_modality_rewards[modality] = 0.0
                    miner_modality_metrics[modality] = self._empty_metrics()

            image_weight = self.config.scoring.image_weight
            video_weight = self.config.scoring.video_weight
            image_rewards = miner_modality_rewards.get(Modality.IMAGE, 0.0)
            video_rewards = miner_modality_rewards.get(Modality.VIDEO, 0.0)
            total_reward = (image_weight * image_rewards) + (
                video_weight * video_rewards
            )

            miner_rewards[uid] = total_reward
            self.miner_metrics[uid] = miner_modality_metrics

        return miner_rewards

    def _get_metrics(
        self, uid: int, modality: Modality, window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a specific miner and modality.

        Args:
            uid: The miner's UID
            modality: The modality to calculate metrics for
            window: Number of recent predictions to consider (default: None = all)

        Returns:
            Dict containing performance metrics
        """
        if (
            uid not in self.tracker.predictions
            or modality not in self.tracker.predictions[uid]
        ):
            return self._empty_metrics()

        recent_preds = list(self.tracker.predictions[uid][modality])
        recent_labels = list(self.tracker.labels[uid][modality])

        if len(recent_labels) != len(recent_preds):
            bt.logging.critical(
                f"Prediction and label array size mismatch ({len(recent_preds)} and {len(recent_labels)})"
            )
            bt.logging.critical(
                f"Clearing miner history for {uid} to allow scoring to resume"
            )
            self.tracker.reset_miner_history(uid)
            return self._empty_metrics()

        if window is not None:
            window = min(window, len(recent_preds))
            recent_preds = recent_preds[-window:]
            recent_labels = recent_labels[-window:]

        pred_probs = np.array([p for p in recent_preds if not np.array_equal(p, -1)])
        labels = np.array(
            [
                l
                for i, l in enumerate(recent_labels)
                if not np.array_equal(recent_preds[i], -1)
            ]
        )

        if len(labels) == 0 or len(pred_probs) == 0:
            return self._empty_metrics()

        try:
            predictions = np.argmax(pred_probs, axis=1)

            # Multi-class MCC (real vs synthetic vs semi-synthetic)
            multi_class_mcc = matthews_corrcoef(labels, predictions)

            # Binary MCC (real vs any synthetic)
            binary_labels = (labels > 0).astype(int)
            binary_preds = (predictions > 0).astype(int)
            binary_mcc = matthews_corrcoef(binary_labels, binary_preds)

            return {"multi_class_mcc": multi_class_mcc, "binary_mcc": binary_mcc}
        except Exception as e:
            bt.logging.warning(f"Error in reward computation: {e}")
            return self._empty_metrics()

    def get_miner_metrics(self, uid):
        return self.miner_metrics.get(uid, self._empty_metrics())

    def _empty_metrics(self):
        """Return a dictionary of empty metrics."""
        return {"multi_class_mcc": 0, "binary_mcc": 0}

    def sync_to_metagraph(self):
        """Just zeros out scores for dereg'd miners. MinerHistory class
        handles clearing predictio history in `update` when a new hotkey
        is detected"""
        hotkeys = self.tracker.miner_hotkeys
        for uid, hotkey in hotkeys.items():
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced
        self.maybe_extend_scores(self.metagraph.n)

    def maybe_extend_scores(self, n):
        """Only for the case where metagraph.n is still growing"""
        if n > len(self.scores):
            n_before = len(self.scores)
            new_moving_average = np.zeros((n))
            new_moving_average[:n_before] = self.scores[:n_before]
            self.scores = new_moving_average

    def save_state(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        scores_path = os.path.join(save_dir, "scores.npy")
        np.save(scores_path, self.scores)
        self.tracker.save_state(save_dir)
        bt.logging.trace(f"Saved state to {save_dir}")

    def load_state(self, save_dir):
        self.tracker.load_state(save_dir)
        scores_path = os.path.join(save_dir, "scores.npy")
        if not os.path.isfile(scores_path):
            bt.logging.info(f"No saved scores found at {scores_path}")
            return False
        try:
            self.scores = np.load(scores_path)
            return True
        except Exception as e:
            bt.logging.error(f"Error deserializing scores: {str(e)}")
            return False

    @staticmethod
    def compute_penalty_multiplier(y_pred: np.ndarray) -> float:
        """
        Compute penalty for predictions outside valid range.

        Args:
            y_pred: Predicted probabilities for each class, shape (3,)

        Returns:
            float: 0.0 if prediction is invalid, 1.0 if valid
        """
        sum_check = np.abs(np.sum(y_pred) - 1.0) < 1e-6
        range_check = np.all((y_pred >= 0.0) & (y_pred <= 1.0))
        return 1.0 if (sum_check and range_check) else 0.0

    @staticmethod
    def transform_reward(reward: float, pole: float = 1.01) -> float:
        """
        Transform reward using an inverse function.

        Args:
            reward: Raw reward value
            pole: Pole parameter for transformation

        Returns:
            float: Transformed reward
        """
        if reward == 0:
            return 0
        return 1 / (pole - np.array(reward))
