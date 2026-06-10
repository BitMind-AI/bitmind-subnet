"""Scene recombination: semi-synthetic scenes from real VLM extractions.

Takes a pixel-grounded SceneDescription and resamples 1-3 contextual fields
(time of day, weather, setting, environment, mood) while preserving the
subject. This breaks the scraper's distributional bias (whatever the source
images over-represent) without giving up the correlational texture of real
scenes — most of the structure stays grounded; only a few axes move.

Used for a fraction of each prompt batch (see generator_service). Remixed
scenes flow through the same committed-spec composition and QC as VLM scenes.

Pure stdlib; deterministic under a seeded random.Random.
"""

from __future__ import annotations

import random
from dataclasses import replace as dc_replace
from typing import List, Optional, Sequence

from gas.generation.prompts.scene import SceneDescription

_TIMES = [
    "dawn", "morning", "midday", "afternoon", "golden hour",
    "sunset", "dusk", "evening", "night",
]
_WEATHER = [
    "clear", "partly cloudy", "overcast", "foggy", "rainy", "snowy", "stormy",
]
_ENVIRONMENTS = ["indoor", "outdoor", "mixed"]
_MOODS = [
    "serene", "tense", "joyful", "melancholic", "mundane", "chaotic",
    "festive", "lonely", "businesslike", "playful",
]

# Fallback settings when no empirical pool is provided. Deliberately
# ordinary — the point is coverage of unglamorous reality, not spectacle.
_DEFAULT_SETTINGS = [
    "supermarket aisle", "parking garage", "suburban kitchen", "bus interior",
    "office cubicle", "laundromat", "roadside diner", "school hallway",
    "construction site", "underground station platform", "backyard patio",
    "hotel corridor", "indoor market", "riverside path", "rooftop terrace",
]

# Fields eligible for resampling, with their value pools (setting handled
# separately because its pool can be empirical).
_FIELD_POOLS = {
    "time_of_day": _TIMES,
    "weather": _WEATHER,
    "environment_type": _ENVIRONMENTS,
    "mood": _MOODS,
}

_MUTABLE_FIELDS = ("time_of_day", "weather", "setting", "environment_type", "mood")


def remix(
    scene: SceneDescription,
    rng: random.Random,
    setting_pool: Optional[Sequence[str]] = None,
) -> SceneDescription:
    """Return a new SceneDescription with 1-3 contextual fields resampled.

    Args:
        scene: Source scene (not mutated).
        rng: Seeded random source (caller controls determinism).
        setting_pool: Optional empirical pool of settings (e.g. drawn from
            the prompts DB scene_json column). Falls back to a built-in
            list of ordinary locations.

    Invariants:
        - `subject` is always preserved.
        - When `time_of_day` changes, `lighting` and `color_palette` are
          cleared so the composer re-derives them instead of inheriting
          values that no longer make sense.
        - The dense `caption` is replaced with a minimal stub built from
          subject + setting, so the composer cannot copy the original
          pixels' description verbatim.
    """
    k = rng.randint(1, 3)
    fields: List[str] = list(rng.sample(_MUTABLE_FIELDS, k))

    updates = {}
    for field in fields:
        if field == "setting":
            pool = list(setting_pool) if setting_pool else _DEFAULT_SETTINGS
            choices = [s for s in pool if s and s != scene.setting]
            if choices:
                updates["setting"] = rng.choice(choices)
        else:
            pool = _FIELD_POOLS[field]
            current = getattr(scene, field)
            choices = [v for v in pool if v != current]
            updates[field] = rng.choice(choices)

    if "time_of_day" in updates:
        updates["lighting"] = ""
        updates["color_palette"] = ""

    new_setting = updates.get("setting", scene.setting)
    stub_bits = [scene.subject or "the subject"]
    if new_setting:
        stub_bits.append(f"in {new_setting}")
    updates["caption"] = " ".join(stub_bits) + "."

    # Observed cues belong to the original pixels; a remixed context can
    # keep dynamic candidates (they travel with the subject) but observed
    # motion cues may contradict the new context.
    updates["observed_motion_cues"] = []

    return dc_replace(scene, **updates)
