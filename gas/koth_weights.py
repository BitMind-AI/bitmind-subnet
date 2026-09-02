"""Build on-chain weights for King-of-the-Hill discriminator lanes."""

from typing import Callable, Dict, Iterable, Optional

import numpy as np

KOTH_SPLIT = {
    "image": 0.40,
    "video": 0.40,
    "audio": 0.04,
    "generator": 0.16,
}


def kings_by_modality(payload: Optional[dict]) -> Dict[str, str]:
    """Map modality -> hotkey from a /current-kings response."""
    out: Dict[str, str] = {}
    if not payload:
        return out
    for king in payload.get("kings") or []:
        modality = king.get("modality")
        hotkey = king.get("ss58_address")
        if modality in ("image", "video", "audio") and hotkey:
            out[modality] = hotkey
    return out


def build_koth_weights(
    n: int,
    scores: np.ndarray,
    generator_uids: Iterable[int],
    kings: Dict[str, str],
    uid_for_hotkey: Callable[[str], Optional[int]],
    burn_uid: Optional[int] = None,
    split: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Return a length-n weight vector. Missing kings go to burn_uid.

    `uid_for_hotkey` must resolve at current chain head. Escrow addresses are
    never used.
    """
    split = dict(split or KOTH_SPLIT)
    weights = np.zeros(n, dtype=np.float64)
    if len(scores) < n:
        scores = np.append(scores, np.zeros(n - len(scores)))
    elif len(scores) > n:
        scores = scores[:n]

    king_uids = set()
    burned = 0.0
    for modality in ("image", "video", "audio"):
        pct = float(split.get(modality, 0.0))
        hotkey = kings.get(modality)
        if not hotkey:
            burned += pct
            continue
        uid = uid_for_hotkey(hotkey)
        if uid is None or uid < 0 or uid >= n:
            burned += pct
            continue
        weights[uid] += pct
        king_uids.add(uid)

    generator_pct = float(split.get("generator", 0.16))
    active = [uid for uid in generator_uids if 0 <= uid < n and uid not in king_uids]
    if active and generator_pct > 0:
        gen_scores = np.array([max(float(scores[uid]), 0.0) for uid in active])
        total = float(np.sum(gen_scores))
        if total > 0:
            for uid, score in zip(active, gen_scores):
                weights[uid] += generator_pct * (score / total)
        else:
            burned += generator_pct
    else:
        burned += generator_pct

    if (
        burned > 0
        and burn_uid is not None
        and 0 <= burn_uid < n
    ):
        weights[burn_uid] += burned

    return weights
