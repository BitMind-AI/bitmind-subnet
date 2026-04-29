"""Per-local-model prompt configuration and runtime adaptation.

Two layers live here:

* **Configuration data** (``ModelPromptConfig``, ``MODEL_STYLES``,
  ``DEFAULT_CONFIG``): per-family rules for how each local diffusers model
  prefers its prompts shaped (prefix, suffix, quality tags, negative
  prompt usage).
* **Runtime adaptation** (``adapt_for_local_model``, ``AdaptedPrompt``):
  applies that configuration to a canonical (LLM-composed) prompt at
  generation time, returning the adapted prompt and an optional
  ``negative_prompt`` for pipelines that accept one.

The canonical (miner-bound) prompt produced by ``PromptGenerator`` is
never altered by this module's data - it is only reshaped at gen-time
when it gets dispatched to a local pipeline inside ``GenerationPipeline``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from gas.generation.prompts.prompt_modifiers import MODIFIERS
from gas.generation.prompts.scene import SceneDescription


@dataclass
class ModelPromptConfig:
    """Configuration for model-specific prompt optimization."""
    
    # Prompt format style
    format: str = "natural_language"  # "natural_language", "tag_based", "hybrid", "action_focused"
    
    # Whether to include quality enhancement tags
    quality_tags: bool = False
    
    # Whether the model supports/benefits from composition instructions
    supports_composition: bool = True
    
    # Whether the model uses negative prompts
    negative_prompt: bool = False
    
    # Base negative prompt if applicable
    negative_base: str = ""
    
    # Optimal token length range (min, max)
    optimal_length: Tuple[int, int] = (50, 150)
    
    # Video-specific: emphasize motion descriptions
    motion_emphasis: bool = False
    
    # Video-specific: supports camera motion instructions
    camera_motion: bool = False
    
    # Video-specific: supports temporal descriptors
    temporal_descriptors: bool = False
    
    # Prefix to add to all prompts (if any)
    prefix: str = ""
    
    # Suffix to add to all prompts (if any)
    suffix: str = ""
    
    # Modifier intensity for this model
    modifier_intensity: str = "moderate"  # "minimal", "moderate", "rich"
    
    # Categories of modifiers that work well with this model
    preferred_modifier_categories: List[str] = field(default_factory=lambda: [
        "style", "lighting", "mood", "color_palette"
    ])


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_STYLES: Dict[str, ModelPromptConfig] = {
    # -------------------------------------------------------------------------
    # FLUX Models
    # -------------------------------------------------------------------------
    "flux": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,  # FLUX doesn't need quality tags
        supports_composition=True,  # Good at following spatial instructions
        negative_prompt=False,  # FLUX doesn't use negative prompts
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere", "composition"],
    ),
    
    "flux.1-dev": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere", "composition"],
    ),
    
    "flux.1-schnell": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(30, 100),  # Shorter prompts work better for schnell
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "lighting", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Stable Diffusion XL Models
    # -------------------------------------------------------------------------
    "sdxl": ModelPromptConfig(
        format="hybrid",  # Natural language + tags work well
        quality_tags=True,  # Benefits from quality tags
        supports_composition=True,
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text, logo, signature, cropped, worst quality, jpeg artifacts",
        optimal_length=(30, 77),  # CLIP token limit consideration
        modifier_intensity="rich",
        preferred_modifier_categories=["style", "lighting", "camera", "quality_tags", "mood", "color_palette"],
    ),
    
    "stable-diffusion-xl": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text, logo, signature, cropped, worst quality, jpeg artifacts",
        optimal_length=(30, 77),
        modifier_intensity="rich",
        preferred_modifier_categories=["style", "lighting", "camera", "quality_tags", "mood", "color_palette"],
    ),
    
    "realvis": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="cartoon, anime, illustration, painting, drawing, low quality, blurry, bad anatomy",
        optimal_length=(30, 77),
        modifier_intensity="rich",
        preferred_modifier_categories=["technical", "camera", "lighting", "quality_tags"],
    ),
    
    # -------------------------------------------------------------------------
    # Stable Diffusion 1.x/2.x Models
    # -------------------------------------------------------------------------
    "stable-diffusion": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,  # Less capable at composition
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text",
        optimal_length=(20, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "quality_tags"],
    ),
    
    "openjourney": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        prefix="mdjrny-v4 style",  # Model-specific trigger
        negative_base="low quality, blurry, bad anatomy",
        optimal_length=(20, 60),
        modifier_intensity="moderate",
        preferred_modifier_categories=["art_movement", "style", "mood"],
    ),
    
    "animagine": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        negative_base="low quality, worst quality, bad anatomy, bad hands",
        optimal_length=(20, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "color_palette"],
    ),
    
    # -------------------------------------------------------------------------
    # HiDream
    # -------------------------------------------------------------------------
    "hidream": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere"],
    ),
    
    # -------------------------------------------------------------------------
    # CogView
    # -------------------------------------------------------------------------
    "cogview": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Chroma
    # -------------------------------------------------------------------------
    "chroma": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "color_palette"],
    ),
    
    # -------------------------------------------------------------------------
    # DeepFloyd IF
    # -------------------------------------------------------------------------
    "deepfloyd": ModelPromptConfig(
        format="natural_language",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad anatomy, watermark",
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "quality_tags"],
    ),
    
    # -------------------------------------------------------------------------
    # Janus
    # -------------------------------------------------------------------------
    "janus": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere"],
    ),
    
    # -------------------------------------------------------------------------
    # Video Models
    # -------------------------------------------------------------------------
    "cogvideo": ModelPromptConfig(
        format="action_focused",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "cogvideox": ModelPromptConfig(
        format="action_focused",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "hunyuan": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "hunyuanvideo": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "mochi": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere"],
    ),
    
    "animatediff": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        motion_emphasis=True,
        camera_motion=False,  # Limited camera control
        temporal_descriptors=True,
        negative_base="low quality, blurry, bad anatomy, static, no motion",
        optimal_length=(30, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "quality_tags"],
    ),
    
    "wan": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "text-to-video-ms": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=False,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=False,
        temporal_descriptors=True,
        optimal_length=(30, 100),
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Inpainting Models
    # -------------------------------------------------------------------------
    "inpainting": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad blending, visible seam, artifact",
        optimal_length=(30, 77),
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "lighting"],
    ),
    
    "dreamshaper": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad anatomy, watermark",
        optimal_length=(30, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "quality_tags"],
    ),
}

# Default configuration for unknown models
DEFAULT_CONFIG = ModelPromptConfig(
    format="natural_language",
    quality_tags=False,
    supports_composition=True,
    negative_prompt=False,
    optimal_length=(50, 150),
    modifier_intensity="moderate",
    preferred_modifier_categories=["style", "lighting", "mood"],
)


# =============================================================================
# CONFIG LOOKUP
# =============================================================================


# Family-fallback table for HF paths that don't substring-match a style key.
_FAMILY_FALLBACKS: Tuple[Tuple[str, str], ...] = (
    ("flux", "flux"),
    ("sdxl", "sdxl"),
    ("xl", "sdxl"),
    ("cogvideo", "cogvideo"),
    ("hunyuan", "hunyuan"),
    ("wan", "wan"),
    ("mochi", "mochi"),
    ("hidream", "hidream"),
    ("animatediff", "animatediff"),
    ("inpaint", "inpainting"),
)


def get_model_config(model_name: str) -> ModelPromptConfig:
    """Resolve a `ModelPromptConfig` for an arbitrary model name.

    Lookup is fuzzy: exact match against ``MODEL_STYLES`` keys first,
    then substring matches, then a few known family fallbacks. This lets
    HF paths like ``tencent/HunyuanVideo`` map cleanly onto the short
    keys used in the styles table.
    """
    model_lower = model_name.lower()

    if model_lower in MODEL_STYLES:
        return MODEL_STYLES[model_lower]

    for key, config in MODEL_STYLES.items():
        if key in model_lower or model_lower in key:
            return config

    for needle, key in _FAMILY_FALLBACKS:
        if needle in model_lower:
            return MODEL_STYLES.get(key, DEFAULT_CONFIG)

    return DEFAULT_CONFIG


# =============================================================================
# RUNTIME ADAPTATION
# =============================================================================


@dataclass
class AdaptedPrompt:
    """Result of per-local-model adaptation."""

    prompt: str
    negative_prompt: Optional[str] = None


def _scene_kind_for_negatives(scene: SceneDescription) -> str:
    """Map our richer scene_kind onto the categories MODIFIERS uses."""
    kind = scene.scene_kind
    if kind in {"portrait", "group"}:
        return "portrait"
    if kind in {"landscape", "nature", "urban"}:
        return "landscape"
    return "general"


def _maybe_quality_suffix(config: ModelPromptConfig, rng: random.Random) -> str:
    if not config.quality_tags:
        return ""
    pool = MODIFIERS.get("quality_tags", [])
    if not pool:
        return ""
    n = min(2, len(pool))
    return ", ".join(rng.sample(pool, n))


def _build_negative(
    config: ModelPromptConfig,
    *,
    is_video: bool,
    scene_category: str,
    rng: random.Random,
) -> Optional[str]:
    if not config.negative_prompt:
        return None

    parts: List[str] = []
    if config.negative_base:
        parts.append(config.negative_base)

    if is_video:
        pool = MODIFIERS.get("negative_video", [])
        n = min(4, len(pool))
    elif scene_category == "portrait":
        pool = MODIFIERS.get("negative_portrait", [])
        n = min(4, len(pool))
    elif scene_category == "landscape":
        pool = MODIFIERS.get("negative_landscape", [])
        n = min(3, len(pool))
    else:
        pool = []
        n = 0

    if pool:
        parts.extend(rng.sample(pool, n))

    return ", ".join(p for p in parts if p) or None


def adapt_for_local_model(
    prompt: str,
    model_name: str,
    *,
    is_video: bool,
    scene: Optional[SceneDescription] = None,
    rng: Optional[random.Random] = None,
) -> AdaptedPrompt:
    """Apply per-model formatting + negatives for a local diffusers pipeline.

    The canonical (miner-bound) prompt is passed through unchanged where
    possible; this layer only adds prefixes/suffixes/quality tags and
    produces an accompanying negative prompt where the model wants one.
    Tokenizer-aware truncation happens later in
    ``truncate_prompt_if_too_long``.

    Returns an `AdaptedPrompt` with `.prompt` and optional `.negative_prompt`.
    """
    rng = rng or random.Random()
    config = get_model_config(model_name)

    out = prompt.strip()

    if config.prefix:
        out = f"{config.prefix}, {out}"

    suffix_pieces: List[str] = []
    quality = _maybe_quality_suffix(config, rng)
    if quality:
        suffix_pieces.append(quality)
    if config.suffix:
        suffix_pieces.append(config.suffix)
    if suffix_pieces:
        out = f"{out} {', '.join(suffix_pieces)}"

    scene_category = (
        _scene_kind_for_negatives(scene) if scene is not None else "general"
    )
    negative = _build_negative(
        config, is_video=is_video, scene_category=scene_category, rng=rng
    )

    return AdaptedPrompt(prompt=out, negative_prompt=negative)
