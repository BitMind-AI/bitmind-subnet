"""Per-prompt register / structure sampling for gasstation prompt composition.

This module is the diversity SOURCE for prompt generation. The composer LLM is
a strong renderer but a weak source of variety: left to choose register, shot
structure, length, and style on its own it collapses into a single mode
(locked-off shallow-DoF close-ups in literary prose). So those axes are
sampled HERE, explicitly and auditably, and injected into the composition
prompt as committed facts the LLM must render.

Pure stdlib, no model dependencies; fully unit-testable.

The ``weights_override`` argument of :func:`sample_spec` is the future hook
for a discriminator-feedback loop (bias register weights toward regions where
detectors are currently strong). Keep it a plain dict of register-name ->
weight so a downstream job can compute it from challenge outcomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Register table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Register:
    """A visual register: how the clip/photo was plausibly captured."""

    name: str
    weight: float
    directives: str                       # rendered into the committed spec block
    camera_motions: Tuple[str, ...]       # sampled per prompt (video only)
    strictness: str = "free"              # "plain" bans literary devices
    image_ok: bool = True                 # register applies to still images too


# Weights lean toward what actually circulates in real media, not uniform.
REGISTERS: Tuple[Register, ...] = (
    Register(
        "phone_casual", 0.20,
        "casual phone capture; imperfect framing, slight tilt, ordinary "
        "lighting, plain colloquial description",
        ("handheld drift", "slow pan", "walking motion", "static handheld"),
        strictness="plain",
    ),
    Register(
        "phone_vertical", 0.10,
        "vertical phone video; held at arm's length or chest height, "
        "everyday context, plain language",
        ("handheld drift", "walking motion", "static handheld"),
        strictness="plain",
    ),
    Register(
        "cctv_surveillance", 0.08,
        "fixed security camera; high mounted angle, wide coverage, flat "
        "utilitarian color, no artistic treatment, plain factual language",
        ("static",),
        strictness="plain",
    ),
    Register(
        "dashcam", 0.05,
        "dashboard camera; wide lens, hood or dashboard edge visible, "
        "scene moves past the fixed mount, plain factual language",
        ("fixed mount, scene moves past",),
        strictness="plain",
        image_ok=False,
    ),
    Register(
        "home_video", 0.07,
        "home video; amateur operator, occasional zoom or reframe, warm "
        "domestic context, plain language",
        ("shaky handheld", "amateur zoom", "reframing pans"),
        strictness="plain",
    ),
    Register(
        "news_broadcast", 0.07,
        "broadcast news; tripod mid-shot or standup framing, even "
        "professional lighting, correspondent or b-roll register",
        ("static tripod", "slow pan", "slow zoom"),
    ),
    Register(
        "documentary", 0.08,
        "observational documentary; deliberate but unpolished, natural "
        "light, patient framing",
        ("static tripod", "slow pan", "handheld follow"),
    ),
    Register(
        "drone_aerial", 0.05,
        "aerial drone; high vantage, gliding movement, wide landscape "
        "or overhead geometry",
        ("gliding forward", "slow orbit", "rising reveal", "static hover"),
    ),
    Register(
        "screen_recording", 0.04,
        "screen recording; software UI, cursor movement, flat rendering, "
        "plain technical description",
        ("static screen", "scrolling content"),
        strictness="plain",
    ),
    Register(
        "animation_3d", 0.05,
        "3D rendered animation; stylized materials and lighting, "
        "deliberate camera",
        ("smooth dolly", "orbit", "static", "push-in"),
    ),
    Register(
        "webcam_stream", 0.06,
        "webcam or stream capture; fixed near framing, compressed look, "
        "desk or room context, plain language",
        ("static",),
        strictness="plain",
    ),
    Register(
        "cinema_polished", 0.15,
        "polished narrative cinematography; full control of lens, light, "
        "and movement; cinematographic vocabulary appropriate",
        ("static tripod", "slow push-in", "dolly", "tracking", "crane",
         "handheld vérité"),
    ),
)

_TOTAL_WEIGHT = sum(r.weight for r in REGISTERS)

# Phrases banned in "plain" registers — the house-style tells measured by
# scripts/prompt_diversity_report.py.
PLAIN_BANNED_PHRASES: Tuple[str, ...] = (
    "as if",
    "as though",
    "no breath",
    "monochrome",
    "palpable",
    "charged",
    "unspoken",
    "the weight of",
    "stillness",
)

# Length bands: (probability, (min_words, max_words))
_LENGTH_BANDS: Dict[str, Tuple[float, Tuple[int, int]]] = {
    "short": (0.25, (80, 130)),
    "medium": (0.45, (130, 180)),
    "long": (0.30, (180, 240)),
}

# Event count distribution for video prompts.
_EVENT_DIST: Tuple[Tuple[int, float], ...] = ((0, 0.35), (1, 0.45), (2, 0.20))


@dataclass
class PromptSpec:
    """Committed per-prompt axes, sampled before composition."""

    modality: str
    register: str
    register_directives: str
    camera_motion: str
    length_band: str
    length_words: Tuple[int, int]
    event_count: int
    style_strictness: str
    banned_phrases: Tuple[str, ...] = field(default_factory=tuple)
    origin: str = "vlm"  # "vlm" | "remix"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["length_words"] = list(self.length_words)
        d["banned_phrases"] = list(self.banned_phrases)
        return d


def _weighted_choice(rng: random.Random, items, weights) -> int:
    total = sum(weights)
    x = rng.uniform(0, total)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x <= acc:
            return i
    return len(items) - 1


def sample_spec(
    modality: str,
    rng_seed: Optional[int] = None,
    weights_override: Optional[Dict[str, float]] = None,
) -> PromptSpec:
    """Sample the committed axes for one prompt.

    Args:
        modality: "image" or "video".
        rng_seed: Optional seed for deterministic sampling (tests).
        weights_override: Optional register-name -> weight map replacing the
            default weights (feedback-loop hook). Registers absent from the
            map get weight 0.
    """
    rng = random.Random(rng_seed)

    candidates: List[Register] = [
        r for r in REGISTERS if (modality == "video" or r.image_ok)
    ]
    if weights_override:
        weights = [weights_override.get(r.name, 0.0) for r in candidates]
        if sum(weights) <= 0:
            weights = [r.weight for r in candidates]
    else:
        weights = [r.weight for r in candidates]

    reg = candidates[_weighted_choice(rng, candidates, weights)]

    band_names = list(_LENGTH_BANDS)
    band_probs = [_LENGTH_BANDS[b][0] for b in band_names]
    band = band_names[_weighted_choice(rng, band_names, band_probs)]
    length_words = _LENGTH_BANDS[band][1]

    if modality == "video":
        camera_motion = rng.choice(reg.camera_motions)
        counts = [c for c, _ in _EVENT_DIST]
        probs = [p for _, p in _EVENT_DIST]
        event_count = counts[_weighted_choice(rng, counts, probs)]
    else:
        camera_motion = ""
        event_count = 0

    banned = PLAIN_BANNED_PHRASES if reg.strictness == "plain" else ()

    return PromptSpec(
        modality=modality,
        register=reg.name,
        register_directives=reg.directives,
        camera_motion=camera_motion,
        length_band=band,
        length_words=length_words,
        event_count=event_count,
        style_strictness=reg.strictness,
        banned_phrases=banned,
    )
