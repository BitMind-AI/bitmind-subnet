"""Structured scene representation extracted from a single image.

A `SceneDescription` is the grounded, model-agnostic intermediate that drives
all downstream prompt composition (image + video, per target model). The
extractor uses the vision-language model in a single forward pass that emits
JSON, keeping the VLM's pixel-level grounding (motion cues, lighting, time of
day, dynamic elements) instead of round-tripping through a flat caption.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import bittensor as bt
import torch
from PIL import Image


_VALID_ENV = {"indoor", "outdoor", "mixed", "unknown"}
_VALID_DISTANCE = {"close-up", "medium", "wide", "aerial", "unknown"}
_VALID_FRAMING = {"portrait", "landscape", "square", "unknown"}
_VALID_SCENE_KIND = {
    "portrait",
    "group",
    "landscape",
    "nature",
    "urban",
    "interior",
    "action",
    "animal",
    "food",
    "object",
    "abstract",
    "documentary",
    "unknown",
}


@dataclass
class SceneDescription:
    """Pixel-grounded structured representation of an image."""

    caption: str
    subject: str = ""
    scene_kind: str = "unknown"
    setting: str = ""
    environment_type: str = "unknown"
    time_of_day: str = "unknown"
    weather: str = "unknown"
    lighting: str = ""
    color_palette: str = ""
    static_elements: List[str] = field(default_factory=list)
    dynamic_candidates: List[str] = field(default_factory=list)
    observed_motion_cues: List[str] = field(default_factory=list)
    camera_distance: str = "unknown"
    framing: str = "unknown"
    style: str = ""
    mood: str = ""
    plausible_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneDescription":
        clean: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in cls.__dataclass_fields__:
                continue
            clean[k] = v

        caption = (clean.get("caption") or "").strip()
        if not caption:
            raise ValueError("SceneDescription requires a non-empty caption")

        for list_field in (
            "static_elements",
            "dynamic_candidates",
            "observed_motion_cues",
            "plausible_events",
        ):
            value = clean.get(list_field)
            if value is None:
                clean[list_field] = []
            elif isinstance(value, str):
                clean[list_field] = [s.strip() for s in re.split(r"[,;]", value) if s.strip()]
            elif isinstance(value, list):
                clean[list_field] = [str(s).strip() for s in value if str(s).strip()]
            else:
                clean[list_field] = []

        env = (clean.get("environment_type") or "unknown").lower().strip()
        clean["environment_type"] = env if env in _VALID_ENV else "unknown"

        dist = (clean.get("camera_distance") or "unknown").lower().strip()
        clean["camera_distance"] = dist if dist in _VALID_DISTANCE else "unknown"

        framing = (clean.get("framing") or "unknown").lower().strip()
        clean["framing"] = framing if framing in _VALID_FRAMING else "unknown"

        kind = (clean.get("scene_kind") or "unknown").lower().strip()
        clean["scene_kind"] = kind if kind in _VALID_SCENE_KIND else "unknown"

        for str_field in (
            "subject",
            "setting",
            "time_of_day",
            "weather",
            "lighting",
            "color_palette",
            "style",
            "mood",
        ):
            value = clean.get(str_field)
            clean[str_field] = "" if value is None else str(value).strip()

        return cls(**clean)


_SCHEMA_INSTRUCTION = (
    "Analyze the image and respond with a SINGLE JSON object (no prose, no "
    "markdown, no code fences). The object MUST contain these keys:\n"
    "- caption: a single dense paragraph (3-5 sentences) describing the image "
    "in vivid detail: main subject, setting, lighting, colors, composition, "
    "mood, and any notable visual elements.\n"
    "- subject: the primary subject in 1-6 words.\n"
    "- scene_kind: one of [portrait, group, landscape, nature, urban, interior, "
    "action, animal, food, object, abstract, documentary].\n"
    "- setting: the place or environment in 1-8 words.\n"
    "- environment_type: one of [indoor, outdoor, mixed].\n"
    "- time_of_day: one of [dawn, morning, midday, afternoon, golden hour, "
    "sunset, dusk, evening, night, unknown].\n"
    "- weather: one of [clear, partly cloudy, overcast, foggy, rainy, snowy, "
    "stormy, indoor, unknown].\n"
    "- lighting: a short phrase describing the dominant lighting "
    "(e.g. 'soft window light', 'harsh midday sun', 'warm tungsten').\n"
    "- color_palette: short phrase (e.g. 'warm earth tones', 'cool blues and "
    "teals', 'desaturated pastels').\n"
    "- static_elements: list of stable elements unlikely to move (e.g. ['stone "
    "wall', 'mountain', 'bookshelf']).\n"
    "- dynamic_candidates: list of elements that could plausibly move in a "
    "short video (e.g. ['water', 'hair', 'leaves', 'traffic', 'flame', "
    "'smoke', 'curtains', 'clouds']). Be generous; include anything that "
    "could naturally animate.\n"
    "- observed_motion_cues: list of motion signals already visible in the "
    "image (e.g. ['hair blowing left', 'visible motion blur on cyclist', "
    "'water surface ripples', 'tilted gait']). Use [] if none.\n"
    "- plausible_events: list of 2-4 discrete events that could believably "
    "happen in this scene within 10 seconds (e.g. ['a waiter passes behind "
    "him', 'the phone on the desk lights up', 'a gust scatters the napkins', "
    "'a car crosses the intersection']). Ground them in what the image shows.\n"
    "- camera_distance: one of [close-up, medium, wide, aerial].\n"
    "- framing: one of [portrait, landscape, square].\n"
    "- style: short phrase (e.g. 'documentary photography', 'cinematic', "
    "'studio portrait', 'street photography', '3D render').\n"
    "- mood: short phrase (e.g. 'serene', 'tense', 'joyful', 'melancholic').\n\n"
    "Output MUST be valid JSON parseable by json.loads. Do not wrap in "
    "markdown. Do not add commentary."
)


def _extract_json_blob(text: str) -> Optional[str]:
    """Pull the first balanced JSON object out of a string."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        return fence.group(1)

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_scene_response(raw: str) -> SceneDescription:
    blob = _extract_json_blob(raw)
    if blob is None:
        raise ValueError("No JSON object found in VLM response")
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("VLM JSON response is not an object")
    return SceneDescription.from_dict(data)


def _fallback_scene_from_text(raw: str) -> SceneDescription:
    """Build a minimal scene from raw VLM text when JSON parsing fails."""
    text = (raw or "").strip()
    if not text:
        text = "An image."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    caption = " ".join(sentences[:5]).strip()
    if not caption.endswith((".", "!", "?")):
        caption += "."
    return SceneDescription(caption=caption)


def extract_scene_with_vlm(
    image: Image.Image,
    vlm,
    vlm_processor,
    max_new_tokens: int = 512,
    temperature: float = 0.4,
) -> SceneDescription:
    """Run a single VLM forward pass and parse a SceneDescription.

    Lower temperature than free-form captioning to keep JSON well-formed; a
    fallback path reconstructs a minimal SceneDescription from the raw text
    if parsing fails so the caller never silently loses a sample.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _SCHEMA_INSTRUCTION},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = vlm_processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)

    with torch.inference_mode():
        generated_ids = vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw = vlm_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    try:
        scene = _parse_scene_response(raw)
        bt.logging.trace(f"Parsed scene: subject='{scene.subject}' kind={scene.scene_kind}")
        return scene
    except (ValueError, json.JSONDecodeError) as e:
        bt.logging.warning(
            f"Scene JSON parse failed ({e}); falling back to text-only scene."
        )
        return _fallback_scene_from_text(raw)
