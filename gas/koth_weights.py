"""Build on-chain weights for King-of-the-Hill discriminator lanes."""

from numbers import Real
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from bittensor.utils import is_valid_ss58_address

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
KOTH_MODALITIES = ("image", "video", "audio")


def resolve_koth_split(split: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Overlay an API split on protocol defaults and validate the result."""
    if split is not None and not isinstance(split, dict):
        raise ValueError("Invalid KOTH split")

    resolved = dict(KOTH_SPLIT)
    for key in KOTH_SPLIT:
        if split is None or key not in split:
            continue
        value = split[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Invalid KOTH split")
        try:
            resolved[key] = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Invalid KOTH split") from exc

    if any(
        not np.isfinite(value) or value < 0 for value in resolved.values()
    ) or not np.isclose(sum(resolved.values()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Invalid KOTH split")
    return resolved


def validate_koth_payload(payload: object) -> dict:
    """Return the trusted subset of a current-kings API response."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid current-kings payload")

    raw_kings = payload.get("kings")
    if raw_kings is None:
        raw_kings = []
    if not isinstance(raw_kings, list):
        raise ValueError("Invalid current-kings payload")

    kings = []
    king_addresses = {}
    for king in raw_kings:
        if not isinstance(king, dict):
            raise ValueError("Invalid current-kings payload")
        modality = king.get("modality")
        address = king.get("ss58_address")
        if modality not in KOTH_MODALITIES or modality in king_addresses:
            raise ValueError("Invalid current-kings payload")
        if not isinstance(address, str) or not is_valid_ss58_address(address):
            raise ValueError("Invalid current-kings payload")
        king_addresses[modality] = address
        kings.append({"modality": modality, "ss58_address": address})

    raw_chain = payload.get("chain")
    if raw_chain is None:
        raw_chain = {}
    if not isinstance(raw_chain, dict):
        raise ValueError("Invalid current-kings payload")

    chain = {}
    for modality in KOTH_MODALITIES:
        raw_members = raw_chain.get(modality)
        if raw_members is None:
            continue
        if not isinstance(raw_members, list):
            raise ValueError("Invalid current-kings payload")

        addresses = []
        for member in raw_members:
            if isinstance(member, str):
                address = member
            elif isinstance(member, dict):
                address = member.get("ss58_address")
            else:
                raise ValueError("Invalid current-kings payload")
            if not isinstance(address, str) or not is_valid_ss58_address(address):
                raise ValueError("Invalid current-kings payload")
            addresses.append(address)

        distinct_addresses = list(dict.fromkeys(addresses))
        if len(distinct_addresses) > len(KOTH_LANE_RESIDUAL):
            raise ValueError("Invalid current-kings payload")
        if (
            distinct_addresses
            and modality in king_addresses
            and distinct_addresses[0] != king_addresses[modality]
        ):
            raise ValueError("Invalid current-kings payload")
        if distinct_addresses:
            chain[modality] = [
                {"ss58_address": address} for address in distinct_addresses
            ]

    return {
        "kings": kings,
        "chain": chain,
        "split": resolve_koth_split(payload.get("split")),
    }


def kings_by_modality(payload: Optional[dict]) -> Dict[str, str]:
    """Map modality -> hotkey from a /current-kings response."""
    out: Dict[str, str] = {}
    if not payload:
        return out
    for king in payload.get("kings") or []:
        modality = king.get("modality")
        hotkey = king.get("ss58_address")
        if modality in KOTH_MODALITIES and hotkey:
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
    for modality in KOTH_MODALITIES:
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
    split = resolve_koth_split(split)
    weights = np.zeros(n, dtype=np.float64)
    if burn_uid is None or not 0 <= burn_uid < n:
        raise ValueError("Owner/burn UID is unavailable")
    if len(scores) < n:
        scores = np.append(scores, np.zeros(n - len(scores)))
    elif len(scores) > n:
        scores = scores[:n]

    king_uids = set()
    burned = 0.0
    for modality in KOTH_MODALITIES:
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
    active = list(
        dict.fromkeys(
            uid
            for uid in generator_uids
            if 0 <= uid < n and uid not in king_uids
        )
    )
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
        weights[burn_uid] += burned

    if (
        len(weights) != n
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("Invalid final KOTH weight vector")

    return weights
