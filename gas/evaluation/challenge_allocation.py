"""Allocate generator challenge slots across qualified / onboarding / probe buckets."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .rewards import GeneratorQualification

Bucket = str  # "qualified" | "onboarding" | "probe" | "unresponsive"
ModalityResponseStats = Dict[str, Dict[str, int]]

# Local challenge outcomes that mean "we asked, they never produced media."
NO_ANSWER_REASONS = frozenset({"no_answer", "challenge_timeout"})


def _is_unresponsive(
    response_stats: Optional[ModalityResponseStats],
    modality: str,
    min_no_answer_attempts: int,
) -> bool:
    """True when this validator asked often enough and got zero answers."""
    if not response_stats or min_no_answer_attempts <= 0:
        return False
    row = response_stats.get(modality) or {}
    answered = int(row.get("answered") or 0)
    no_answer = int(row.get("no_answer") or 0)
    return answered == 0 and no_answer >= min_no_answer_attempts


def classify_modality_bucket(
    qualification: Optional[GeneratorQualification],
    modality: str,
    min_fool_samples: int = 20,
    response_stats: Optional[ModalityResponseStats] = None,
    min_no_answer_attempts: int = 5,
) -> Bucket:
    """Classify one UID for one modality.

    Repeated no-answers on this validator (never accepted, or accepted and
    never delivered) are unresponsive: they do not occupy onboarding slots.
    Missing qualification or n < min_fool_samples is onboarding. Over the
    cutoff is qualified. n >= min and under the cutoff is probe.
    """
    if _is_unresponsive(response_stats, modality, min_no_answer_attempts):
        return "unresponsive"
    if qualification is None:
        return "onboarding"
    if modality == "image":
        n = qualification.image_n
        qualified = qualification.qualified_image
    elif modality == "video":
        n = qualification.video_n
        qualified = qualification.qualified_video
    else:
        return "onboarding"
    if n < min_fool_samples:
        return "onboarding"
    if qualified:
        return "qualified"
    return "probe"


def resolve_challenge_response_stats(
    response_stats: Optional[Dict[str, ModalityResponseStats]],
    metagraph,
) -> Optional[Dict[int, ModalityResponseStats]]:
    """Map hotkey-keyed no-answer counts onto current UIDs.

    A replacement at a recycled UID starts at zero; the prior occupant's
    totals stay on the old hotkey and do not transfer.
    """
    if not response_stats:
        return None
    return {
        uid: response_stats[hotkey]
        for uid, hotkey in enumerate(list(metagraph.hotkeys))
        if hotkey in response_stats
    }


def _slot_targets(
    qualified_slots: int,
    onboarding_slots: int,
    probe_slots: int,
    any_onboarding: bool,
) -> Tuple[int, int, int, bool]:
    """Return (qualified, onboarding, probe, rolled_onboarding)."""
    if any_onboarding:
        return qualified_slots, onboarding_slots, probe_slots, False
    extra_qualified = onboarding_slots // 2
    extra_probe = onboarding_slots - extra_qualified
    return (
        qualified_slots + extra_qualified,
        0,
        probe_slots + extra_probe,
        True,
    )


def allocate_random_slots(
    miner_uids: Sequence[int],
    available_modalities: Sequence[str],
    *,
    sample_size: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, object]]:
    """Pick unique generators uniformly, each with a random image/video modality."""
    rng = rng or np.random.default_rng()
    uids = [int(u) for u in dict.fromkeys(miner_uids)]
    mods = [str(m).strip().lower() for m in available_modalities]
    mods = [m for m in mods if m in ("image", "video")]
    stats = {
        "mode": "random",
        "pool": len(uids),
        "image_qualified": 0,
        "video_qualified": 0,
        "onboarding": 0,
        "probe": 0,
        "image_unresponsive": 0,
        "video_unresponsive": 0,
        "rolled_onboarding": False,
    }
    if not uids or not mods or sample_size <= 0:
        return [], stats
    n = min(int(sample_size), len(uids))
    chosen = rng.choice(np.array(uids, dtype=int), size=n, replace=False)
    slot_mods = [mods[int(i)] for i in rng.integers(0, len(mods), size=n)]
    assignments = [(int(uid), slot_mods[i]) for i, uid in enumerate(chosen)]
    return assignments, stats


def allocate_challenge_slots(
    miner_uids: Sequence[int],
    available_modalities: Sequence[str],
    qualification: Optional[Dict[int, GeneratorQualification]],
    *,
    sample_size: int = 50,
    qualified_slots: int = 36,
    onboarding_slots: int = 8,
    probe_slots: int = 6,
    min_fool_samples: int = 20,
    response_stats: Optional[Dict[int, ModalityResponseStats]] = None,
    min_no_answer_attempts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, object]]:
    """Pick unique (uid, modality) pairs for one validator challenge round.

    Modality is chosen first for each slot, then a UID from the matching
    bucket. If qualification is None (stale/missing API), every UID is
    onboarding unless local no-answer stats mark that modality unresponsive.
    Unused onboarding slots when that set is empty become 4+4 extra
    qualified/probe (40/10 at the defaults). Remaining unused slots overflow
    qualified → probe → qualified, then any leftover miner who still answers
    that modality.

    Returns (assignments, stats) where stats includes pool sizes and
    rolled_onboarding.
    """
    rng = rng or np.random.default_rng()
    uids = [int(u) for u in dict.fromkeys(miner_uids)]
    mods = [str(m).strip().lower() for m in available_modalities]
    mods = [m for m in mods if m in ("image", "video")]
    empty_stats = {
        "image_qualified": 0,
        "video_qualified": 0,
        "onboarding": 0,
        "probe": 0,
        "image_unresponsive": 0,
        "video_unresponsive": 0,
        "rolled_onboarding": False,
    }
    if not uids or not mods or sample_size <= 0:
        return [], empty_stats

    def bucket_for(uid: int, modality: str) -> Bucket:
        q = None if qualification is None else qualification.get(uid)
        stats = None if response_stats is None else response_stats.get(uid)
        return classify_modality_bucket(
            q,
            modality,
            min_fool_samples,
            response_stats=stats,
            min_no_answer_attempts=min_no_answer_attempts,
        )

    image_qualified = video_qualified = 0
    image_unresponsive = video_unresponsive = 0
    onboarding_uids = set()
    probe_uids = set()
    for uid in uids:
        for mod in mods:
            kind = bucket_for(uid, mod)
            if kind == "qualified":
                if mod == "image":
                    image_qualified += 1
                else:
                    video_qualified += 1
            elif kind == "onboarding":
                onboarding_uids.add(uid)
            elif kind == "probe":
                probe_uids.add(uid)
            elif kind == "unresponsive":
                if mod == "image":
                    image_unresponsive += 1
                else:
                    video_unresponsive += 1
    onboarding_n = len(onboarding_uids)
    probe_n = len(probe_uids)

    n_q, n_o, n_p, rolled = _slot_targets(
        qualified_slots, onboarding_slots, probe_slots, bool(onboarding_uids)
    )
    n = min(int(sample_size), len(uids))
    slot_mods = [mods[int(i)] for i in rng.integers(0, len(mods), size=n)]
    assigned: List[Optional[int]] = [None] * n
    used = set()

    def fill(kind: Bucket, count: int) -> int:
        if count <= 0:
            return 0
        filled = 0
        for i in rng.permutation(n):
            if filled >= count:
                break
            if assigned[i] is not None:
                continue
            pool = [
                uid
                for uid in uids
                if uid not in used and bucket_for(uid, slot_mods[i]) == kind
            ]
            if not pool:
                continue
            uid = int(pool[int(rng.integers(0, len(pool)))])
            assigned[i] = uid
            used.add(uid)
            filled += 1
        return count - filled

    leftover_onboarding = fill("onboarding", n_o)
    leftover_qualified = fill("qualified", n_q + leftover_onboarding)
    leftover_probe = fill("probe", n_p + leftover_qualified)
    leftover_after_probe = fill("qualified", leftover_probe)
    leftover_after_onb = fill("onboarding", leftover_after_probe)
    fill("probe", leftover_after_onb)

    for i in range(n):
        if assigned[i] is not None:
            continue
        pool = [
            uid
            for uid in uids
            if uid not in used and bucket_for(uid, slot_mods[i]) != "unresponsive"
        ]
        if not pool:
            continue
        uid = int(pool[int(rng.integers(0, len(pool)))])
        assigned[i] = uid
        used.add(uid)

    assignments = [
        (assigned[i], slot_mods[i])
        for i in range(n)
        if assigned[i] is not None
    ]
    stats = {
        "image_qualified": image_qualified,
        "video_qualified": video_qualified,
        "onboarding": onboarding_n,
        "probe": probe_n,
        "image_unresponsive": image_unresponsive,
        "video_unresponsive": video_unresponsive,
        "rolled_onboarding": rolled,
    }
    return assignments, stats


def summarize_assignment_buckets(
    assignments: Iterable[Tuple[int, str]],
    qualification: Optional[Dict[int, GeneratorQualification]],
    min_fool_samples: int = 20,
    response_stats: Optional[Dict[int, ModalityResponseStats]] = None,
    min_no_answer_attempts: int = 5,
) -> Dict[str, int]:
    """Count assigned UIDs by the bucket used for their chosen modality."""
    counts = {"qualified": 0, "onboarding": 0, "probe": 0, "unresponsive": 0}
    for uid, modality in assignments:
        q = None if qualification is None else qualification.get(uid)
        stats = None if response_stats is None else response_stats.get(uid)
        kind = classify_modality_bucket(
            q,
            modality,
            min_fool_samples,
            response_stats=stats,
            min_no_answer_attempts=min_no_answer_attempts,
        )
        counts[kind] += 1
    return counts
