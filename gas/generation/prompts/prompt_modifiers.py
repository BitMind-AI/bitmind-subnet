"""Modifier word lists used by the prompt composer.

Only the categories actually referenced by `model_prompt_styles.adapt_for_local_model`
are kept here:

  * ``quality_tags`` - appended for models with `quality_tags=True`
  * ``negative_video`` - sampled for video pipelines that accept negative_prompt
  * ``negative_portrait`` - sampled for portrait-kind image pipelines
  * ``negative_landscape`` - sampled for landscape-kind image pipelines

Other modifier categories from earlier iterations (style, lighting, camera,
composition, mood, ...) were removed when their consumers (`optimize_prompt`,
`enhance_prompt`, `generate_modifier_selection`) were retired. If a future
feature needs richer per-category modifiers, reintroduce them here.
"""

from __future__ import annotations

from typing import Dict, List


MODIFIERS: Dict[str, List[str]] = {
    "quality_tags": [
        "masterpiece", "best quality", "high quality", "ultra quality",
        "highly detailed", "intricate details", "fine details",
        "professional photography", "award-winning", "featured on 500px",
        "trending on artstation", "8K wallpaper", "stunning",
    ],
    "negative_portrait": [
        "bad anatomy", "wrong anatomy", "extra limbs", "missing limbs",
        "mutated hands", "extra fingers", "missing fingers", "fused fingers",
        "deformed face", "ugly face", "asymmetric face",
        "crossed eyes", "dead eyes", "uncanny valley",
        "bad proportions", "long neck", "long body",
    ],
    "negative_landscape": [
        "oversaturated", "unrealistic colors", "bad composition",
        "cluttered", "messy", "chaotic", "unbalanced",
        "artificial looking", "fake", "cgi obvious",
    ],
    "negative_video": [
        "flickering", "temporal inconsistency", "frame jumping",
        "motion blur artifacts", "ghosting", "tearing",
        "stuttering", "jerky motion", "unnatural movement",
    ],
}
