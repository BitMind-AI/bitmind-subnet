"""Allocate generator challenge slots across qualified / onboarding / probe buckets."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gas.evaluation.rewards import GeneratorQualification

Bucket = str  # "qualified" | "onboarding" | "probe"


def classify_modality_bucket(
    qualification: Optional[GeneratorQualification],
    modality: str,
    min_fool_samples: int = 20,
) -> Bucket:
    """Classify one UID for one modality.

    Missing qualification or n < min_fool_samples is onboarding. Over the
    cutoff is qualified. n >= min and under the cutoff is probe.
    """
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
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, object]]:
    """Pick unique (uid, modality) pairs for one validator challenge round.

    Modality is chosen first for each slot, then a UID from the matching
    bucket. If qualification is None (stale/missing API), every UID is
    onboarding. Unused onboarding slots when that set is empty become
    4+4 extra qualified/probe (40/10 at the defaults). Remaining unused
    slots overflow qualified → probe → qualified, then any leftover miner.

    Returns (assignments, stats) where stats includes pool sizes and
    rolled_onboarding.
    """
    rng = rng or np.random.default_rng()
    uids = [int(u) for u in dict.fromkeys(miner_uids)]
    mods = [str(m).strip().lower() for m in available_modalities]
    mods = [m for m in mods if m in ("image", "video")]
    if not uids or not mods or sample_size <= 0:
        return [], {
            "image_qualified": 0,
            "video_qualified": 0,
            "onboarding": 0,
            "probe": 0,
            "rolled_onboarding": False,
        }

    def bucket_for(uid: int, modality: str) -> Bucket:
        if qualification is None:
            return "onboarding"
        return classify_modality_bucket(
            qualification.get(uid), modality, min_fool_samples
        )

    image_qualified = video_qualified = onboarding_n = probe_n = 0
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
            else:
                probe_uids.add(uid)
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
        pool = [uid for uid in uids if uid not in used]
        if not pool:
            break
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
        "rolled_onboarding": rolled,
    }
    return assignments, stats


def summarize_assignment_buckets(
    assignments: Iterable[Tuple[int, str]],
    qualification: Optional[Dict[int, GeneratorQualification]],
    min_fool_samples: int = 20,
) -> Dict[str, int]:
    """Count assigned UIDs by the bucket used for their chosen modality."""
    counts = {"qualified": 0, "onboarding": 0, "probe": 0}
    for uid, modality in assignments:
        if qualification is None:
            kind = "onboarding"
        else:
            kind = classify_modality_bucket(
                qualification.get(uid), modality, min_fool_samples
            )
        counts[kind] += 1
    return counts
