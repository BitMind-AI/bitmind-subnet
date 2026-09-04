"""Build on-chain weights for King-of-the-Hill discriminator lanes."""

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

KOTH_SPLIT = {
    "image": 0.40,
    "video": 0.40,
    "audio": 0.04,
    "generator": 0.16,
}
# Per-lane residual after the current king: previous, then two-back.
# Unused slots (no prior distinct king) roll up to the current king.
KOTH_LANE_RESIDUAL = (0.85, 0.10, 0.05)
KOTH_CHAIN_ROLES = ("current", "previous", "two_back")


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


def assign_residual_shares(hotkeys: Iterable[str]) -> List[Dict[str, object]]:
    """Map last-N distinct hotkeys to 85/10/5, rolling unused slots to current."""
    keys = []
    seen = set()
    for key in hotkeys:
        if not key or key in seen:
            continue
        keys.append(key)
        seen.add(key)
        if len(keys) >= len(KOTH_LANE_RESIDUAL):
            break
    if not keys:
        return []
    shares = list(KOTH_LANE_RESIDUAL[: len(keys)])
    shares[0] += sum(KOTH_LANE_RESIDUAL[len(keys) :])
    return [
        {
            "ss58_address": key,
            "share": shares[index],
            "role": KOTH_CHAIN_ROLES[index],
        }
        for index, key in enumerate(keys)
    ]


def chains_by_modality(payload: Optional[dict]) -> Dict[str, List[Dict[str, object]]]:
    """Last-3 distinct kings per lane, with residual shares recomputed locally."""
    if not payload:
        return {}
    raw = payload.get("chain") or {}
    kings = kings_by_modality(payload)
    out: Dict[str, List[Dict[str, object]]] = {}
    for modality in ("image", "video", "audio"):
        members = raw.get(modality) or []
        hotkeys = []
        for member in members:
            if isinstance(member, str):
                hotkeys.append(member)
            elif isinstance(member, dict):
                hotkeys.append(member.get("ss58_address"))
        assigned = assign_residual_shares(hotkeys)
        if not assigned and kings.get(modality):
            assigned = assign_residual_shares([kings[modality]])
        if assigned:
            out[modality] = assigned
    return out


def build_koth_weights(
    n: int,
    scores: np.ndarray,
    generator_uids: Iterable[int],
    kings: Dict[str, str],
    uid_for_hotkey: Callable[[str], Optional[int]],
    burn_uid: Optional[int] = None,
    split: Optional[Dict[str, float]] = None,
    chains: Optional[Dict[str, List[Dict[str, object]]]] = None,
) -> np.ndarray:
    """Return a length-n weight vector. Missing kings go to burn_uid.

    `uid_for_hotkey` must resolve at current chain head. Escrow addresses are
    never used. Each discriminator lane is 85/10/5 across the current king and
    the previous two distinct kings. Unused residual slots roll to the current
    king. An unresolvable current king burns its share; unresolvable previous
    kings roll to the current king when that UID resolved.
    """
    split = dict(KOTH_SPLIT if split is None else split)
    try:
        split = {key: float(split[key]) for key in KOTH_SPLIT}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid KOTH split") from exc
    if any(
        not np.isfinite(value) or value < 0 for value in split.values()
    ) or not np.isclose(sum(split.values()), 1.0):
        raise ValueError("Invalid KOTH split")
    weights = np.zeros(n, dtype=np.float64)
    if len(scores) < n:
        scores = np.append(scores, np.zeros(n - len(scores)))
    elif len(scores) > n:
        scores = scores[:n]

    king_uids = set()
    burned = 0.0
    for modality in ("image", "video", "audio"):
        pct = split[modality]
        members = list((chains or {}).get(modality) or [])
        if not members:
            hotkey = kings.get(modality)
            members = assign_residual_shares([hotkey] if hotkey else [])
        if not members:
            burned += pct
            continue

        current_uid = None
        leftover = 0.0
        current_resolved = False
        for index, member in enumerate(members):
            hotkey = member.get("ss58_address")
            share = float(member.get("share") or 0.0)
            if share <= 0:
                continue
            uid = uid_for_hotkey(hotkey) if hotkey else None
            if uid is None or uid < 0 or uid >= n:
                if index == 0:
                    burned += pct * share
                else:
                    leftover += share
                continue
            weights[uid] += pct * share
            king_uids.add(uid)
            if index == 0:
                current_uid = uid
                current_resolved = True

        if leftover > 0:
            if current_resolved and current_uid is not None:
                weights[current_uid] += pct * leftover
            else:
                burned += pct * leftover

    generator_pct = split["generator"]
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

    if burned > 0:
        if burn_uid is None or not 0 <= burn_uid < n:
            raise ValueError(
                f"Cannot allocate {burned:.4f} burn weight: burn UID is unavailable"
            )
        weights[burn_uid] += burned

    return weights
