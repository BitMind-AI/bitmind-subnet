"""Hotkey-owned, independently gated generator score histories."""

import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Optional, Sequence

from .rewards import GeneratorQualification


class GeneratorScoreState:
    """Persist modality EMAs, qualification, and the last payable hotkey scores."""

    def __init__(self):
        self.by_hotkey: Dict[str, Dict[str, float]] = {}
        self.qualification: Dict[str, GeneratorQualification] = {}
        # None means no safe fallback exists; {} means a successful empty payout.
        # Unlike the EMA history, this contains only miners eligible for payout.
        self.last_payout: Optional[Dict[str, float]] = None

    def resolve_last_payout(self, hotkeys: Sequence[str]) -> Optional[Dict[int, float]]:
        """Resolve the last good payout against current identities, never old UIDs."""
        if self.last_payout is None:
            return None
        return {
            uid: self.last_payout[hotkey]
            for uid, hotkey in enumerate(hotkeys)
            if hotkey in self.last_payout
        }

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
            json.dump({
                "version": 1,
                "by_hotkey": self.by_hotkey,
                "qualification": {hotkey: asdict(q) for hotkey, q in self.qualification.items()},
                "last_payout": self.last_payout,
            }, stream, allow_nan=False)

    def load_state(self, save_dir: str, filename: str) -> bool:
        # Missing/invalid new-format state must never reuse legacy scalar EMA.
        self.by_hotkey = {}
        self.qualification = {}
        self.last_payout = None
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
            # Older EMA snapshots have no qualification cache. Keep them
            # readable, but do not invent eligibility from historical scores.
            cached = payload.get("qualification", {})
            if not isinstance(cached, dict):
                return False
            qualification = {}
            expected_fields = {field.name for field in fields(GeneratorQualification)}
            for hotkey, row in cached.items():
                if not isinstance(hotkey, str) or not hotkey or not isinstance(row, dict):
                    return False
                if set(row) != expected_fields:
                    return False
                for modality in ("image", "video"):
                    n, fooled = row[f"{modality}_n"], row[f"{modality}_fooled"]
                    qualified = row[f"qualified_{modality}"]
                    if type(n) is not int or type(fooled) is not int or type(qualified) is not bool:
                        return False
                    if not 0 <= fooled <= n or (qualified and n == 0):
                        return False
                qualification[hotkey] = GeneratorQualification(**row)
            # Older snapshots cannot establish payout eligibility from EMA alone.
            payout = payload.get("last_payout")
            if payout is not None:
                if not isinstance(payout, dict):
                    return False
                for hotkey, score in payout.items():
                    if (not isinstance(hotkey, str) or not hotkey
                            or type(score) not in (int, float)
                            or not math.isfinite(score) or score <= 0):
                        return False
            self.by_hotkey = restored
            self.qualification = qualification
            self.last_payout = payout
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False
