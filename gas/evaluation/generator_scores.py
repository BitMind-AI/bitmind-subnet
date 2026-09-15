"""Hotkey-owned, independently gated generator score histories."""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

from .rewards import GeneratorQualification


class GeneratorScoreState:
    """Persist unweighted image/video EMAs; never seed from legacy scalar scores."""

    def __init__(self):
        self.by_hotkey: Dict[str, Dict[str, float]] = {}

    def update(
        self,
        base_rewards: Dict[int, Dict[str, float]],
        qualification: Dict[str, GeneratorQualification],
        hotkeys: Sequence[str],
        *,
        image_weight: float = 0.30,
        video_weight: float = 0.70,
        alpha: float = 0.5,
        last_seen: Optional[Dict[str, float]] = None,
        inactive_cutoff: float = 0.0,
    ) -> Dict[int, float]:
        """Update every epoch, including epochs with no payable base rewards.

        Disqualified lanes and inactive/deregistered hotkeys lose their history
        immediately. Qualified lanes with no new reward decay normally. Callers
        still restrict payout to UIDs with positive current gated base rewards.
        """
        next_history = {}
        scores = {}
        for uid, hotkey in enumerate(hotkeys):
            q = qualification.get(hotkey)
            if q is None or (last_seen and last_seen.get(hotkey, 0) < inactive_cutoff):
                continue
            previous = self.by_hotkey.get(hotkey, {})
            base = base_rewards.get(uid, {})
            lanes = {}
            for modality in ("image", "video"):
                if getattr(q, f"qualified_{modality}"):
                    current = float(base.get(modality, 0.0) or 0.0)
                    lanes[modality] = alpha * current + (1 - alpha) * previous.get(modality, 0.0)
                else:
                    lanes[modality] = 0.0
            if any(value > 0 for value in lanes.values()):
                next_history[hotkey] = lanes
                scores[uid] = image_weight * lanes["image"] + video_weight * lanes["video"]
        self.by_hotkey = next_history
        return scores

    def save_state(self, save_dir: str, filename: str) -> None:
        # StateManager writes this into its temporary snapshot before swapping.
        with (Path(save_dir) / filename).open("w") as stream:
            json.dump({"version": 1, "by_hotkey": self.by_hotkey}, stream, allow_nan=False)

    def load_state(self, save_dir: str, filename: str) -> bool:
        # Missing/invalid new-format state must never reuse legacy scalar EMA.
        self.by_hotkey = {}
        path = Path(save_dir) / filename
        if not path.exists():
            return False
        try:
            with path.open() as stream:
                payload = json.load(stream)
            if payload.get("version") != 1 or not isinstance(payload.get("by_hotkey"), dict):
                return False
            restored = {}
            for hotkey, lanes in payload["by_hotkey"].items():
                if not isinstance(hotkey, str) or not isinstance(lanes, dict):
                    return False
                values = {modality: float(lanes[modality]) for modality in ("image", "video")}
                if any(not math.isfinite(value) or value < 0 for value in values.values()):
                    return False
                restored[hotkey] = values
            self.by_hotkey = restored
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False
