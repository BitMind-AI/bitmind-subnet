"""Scene-grounded, SoTA-quality prompt generation from source images.

Pipeline (per source image):

    image
      └─► VLM (Qwen3-VL-4B, single forward pass)
            └─► SceneDescription (grounded perception, JSON)
                  │
                  ▼
            LLM (Qwen3-30B-A3B, two forward passes)
                  ├─► canonical IMAGE prompt
                  └─► canonical VIDEO prompt

The VLM does what it's uniquely good at: extract grounded perceptual facts
from pixels (subject, setting, lighting, dynamic candidates, observed
motion cues, etc.) as structured JSON.

The LLM does what it's uniquely good at: take those grounded facts and
write a SoTA-grade cinematographic prompt as a single fluent paragraph.
There is no rule-based intermediate "composer" step; the model picks shot
language, motion verbs, lens/depth-of-field, pacing, and weaves them into
prose end-to-end. This eliminates template footprint and leverages the
LLM's full creative capacity.

Both models are loaded warm during a prompt-gen batch and freed via
`clear_gpu()` at the end of the batch. The expected default batch size is
50 so the model load cost (~30-60s for the 60GB LLM in bf16) amortizes
well.

The composed prompts are stored verbatim in the prompts DB and dispatched
to miners (which wrap SoTA video/image APIs). Per-local-model formatting
for diffusers pipelines is applied separately at gen-time inside
`GenerationPipeline` and does not alter the canonical stored prompt.
"""

from __future__ import annotations

import gc
import json
import re
from collections import deque
from dataclasses import asdict
from typing import Callable, Deque, Dict, List, Optional

import bittensor as bt
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from gas.generation.prompts.register_sampler import PromptSpec, sample_spec
from gas.generation.prompts.prompt_qc import validate as qc_validate
from gas.generation.prompts.scene import SceneDescription, extract_scene_with_vlm
from gas.types import Modality


# Default VLM: Qwen3-VL-4B-Instruct. ~9GB VRAM in bf16.
# Used for grounded perceptual extraction (image -> structured scene JSON).
# Requires transformers >= 4.57.0.
IMAGE_ANNOTATION_MODEL: str = "Qwen/Qwen3-VL-4B-Instruct"

# Default LLM: Qwen3-30B-A3B-Instruct-2507 in bf16. ~60GB VRAM.
# Mixture-of-Experts: 30B total params, only 3B active per token, so
# decode throughput is comparable to a dense ~4B model while reasoning
# capacity is ~30B-class. This is the right tradeoff for "highest-quality
# cinematographic prose without unreasonable latency". Sized for an 80GB
# card alongside the VLM (~69GB total during prompt-gen).
TEXT_LLM_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"


# ---------------------------------------------------------------------------
# Composition system prompts
# ---------------------------------------------------------------------------

_VIDEO_SYSTEM = (
    "You write text-to-video prompts.\n\n"
    "You will receive a structured scene that a vision-language model "
    "extracted from a reference photograph, plus the same VLM's dense "
    "caption of that image. You may also receive a COMMITTED SHOT SPEC: "
    "when present it is authoritative — register, camera behavior, length, "
    "and event count are already decided; your job is only to realize them "
    "vividly and plausibly for THIS scene.\n\n"
    "Your job: imagine the scene as a 5-10 second video clip and write "
    "a SINGLE fluent paragraph describing that clip. Match the visual "
    "register implied by the source — that might be a casual phone "
    "clip, a surveillance frame, a documentary observation, a home "
    "video, a screen recording, an animation, a press capture, an "
    "editorial shot, polished narrative cinema, or many other "
    "possibilities. Do not always default to polished cinematography.\n\n"
    "REQUIREMENTS\n"
    "- Present tense, one paragraph, no line breaks. Respect the length "
    "given in the shot spec (or the user turn).\n"
    "- Make the scene visually concrete: subject + action, setting, "
    "what is moving, lighting and atmosphere. Include framing, camera "
    "behavior, lens, depth of field, color palette, or pacing only "
    "when they fit the register — a snapshot or surveillance frame "
    "may have an untreated static camera and no notion of 'lens'; a "
    "polished shot may have explicit framing, movement, and lens "
    "choices. Choose what genuinely fits THIS scene.\n"
    "- Ground subject motion in what plausibly moves: prefer the "
    "scene's `dynamic_candidates` and `observed_motion_cues`. Do not "
    "invent subjects or actions that aren't implied by the scene.\n"
    "- Use specific, vivid motion verbs (e.g. 'drifts', 'scatters', "
    "'billows', 'ripples', 'unfurls', 'spirals'); avoid weak verbs "
    "like 'moves', 'is', 'happens'. Vary your verb choices.\n"
    "- Use specific visual nouns: 'storm clouds gathering above the "
    "ridge', not 'moody sky'; 'rim light from a north-facing window', "
    "not 'soft light'.\n"
    "- Avoid clichés and marketing words: 'breathtaking', 'stunning', "
    "'masterpiece', 'cinematic masterpiece', 'epic'. Avoid hedging: "
    "'perhaps', 'might', 'seems'.\n"
    "- Do NOT use bullet points, headings, lists, quotation marks, or "
    "negative phrasing ('no flicker', 'avoid blur'). Do NOT explain "
    "your choices.\n"
    "- Output ONLY the paragraph, nothing else."
)

_IMAGE_SYSTEM = (
    "You write text-to-image prompts.\n\n"
    "You will receive a structured scene that a vision-language model "
    "extracted from a reference photograph, plus the same VLM's dense "
    "caption of that image. You may also receive a COMMITTED SHOT SPEC: "
    "when present it is authoritative — register, length, and style "
    "constraints are already decided; realize them for THIS scene.\n\n"
    "Your job: rewrite the scene as a SINGLE dense natural-language "
    "image prompt with maximum visual specificity. Match the visual "
    "register implied by the source — that might be a casual snapshot, "
    "a surveillance still, a press photo, a screen capture, an "
    "illustration or painting, a polished editorial frame, or many "
    "other possibilities. Do not always default to polished editorial "
    "photography.\n\n"
    "REQUIREMENTS\n"
    "- One paragraph, no line breaks. Respect the length given in the "
    "shot spec (or the user turn).\n"
    "- Lead with the most salient element for THIS scene; vary the "
    "structure across calls. Include framing, lens, color palette, "
    "lighting, mood, composition, or style/medium only when they fit "
    "the register implied by the source.\n"
    "- Use specific visual nouns: 'storm clouds gathering above the "
    "ridge', not 'moody sky'; 'rim light from a north-facing window', "
    "not 'soft light'.\n"
    "- Avoid clichés and marketing words: 'breathtaking', 'stunning', "
    "'masterpiece', 'award-winning'. Avoid generic adjectives: "
    "'beautiful', 'amazing', 'perfect'.\n"
    "- Do NOT use bullet points, headings, lists, quotation marks, or "
    "negative phrasing. Do NOT explain your choices.\n"
    "- Output ONLY the paragraph, nothing else."
)


class PromptGenerator:
    """Generate scene-grounded, SoTA-quality prompts from source images.

    Two warm models during a prompt-gen batch:
      * VLM (Qwen3-VL-4B) - grounded perception, ~9GB
      * LLM (Qwen3-30B-A3B MoE) - cinematographic composition, ~60GB

    Total ~69GB; sized for an 80GB card. Both are dropped after each
    batch via `clear_gpu()`.
    """

    # Sliding window cap for the per-modality prior-prompt history that
    # is re-injected into each user message for diversity pressure.
    # Capped because input context drives both per-call latency (linear
    # prefill cost) and peak VRAM: ~70GB resident across VLM + 30B-A3B
    # LLM on an 80GB card leaves only ~10GB for activations + KV cache,
    # and an unbounded history OOMs at high batch indices. 12 is well
    # above the 4-example anchor that originally biased the system
    # prompt, so the marginal diversity benefit of going higher is
    # small while the VRAM cost is linear.
    _PRIOR_WINDOW = 12

    def __init__(
        self,
        vlm_name: str = IMAGE_ANNOTATION_MODEL,
        llm_name: str = TEXT_LLM_MODEL,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            vlm_name: HF id for the vision-language model.
            llm_name: HF id for the composition LLM.
            device: Device identifier (e.g. "cuda" or "cuda:0").
        """
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.device = device

        self.vlm_processor = None
        self.vlm = None
        self.llm = None

        self._prior_image_prompts: Deque[str] = deque(maxlen=self._PRIOR_WINDOW)
        self._prior_video_prompts: Deque[str] = deque(maxlen=self._PRIOR_WINDOW)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _attn_impl(self) -> str:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            bt.logging.warning("flash_attn not available, using eager attention")
            return "eager"

    def load_vlm(self) -> None:
        bt.logging.debug(f"Loading VLM {self.vlm_name}")
        attn_impl = self._attn_impl()

        try:
            bt.logging.info(f"Attempting to load {self.vlm_name} from local cache...")
            self.vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_name, local_files_only=True
            )
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                self.vlm_name,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
                local_files_only=True,
            )
        except (OSError, ValueError, TypeError):
            bt.logging.info("VLM not in local cache, downloading from HuggingFace...")
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_name)
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                self.vlm_name,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            )

        # Clamp image processor to prevent absurd single-image allocations
        # (observed: 124 GiB tensor request on an 80 GiB card when the
        # processor is configured with unbounded max_pixels / longest_edge).
        if self.vlm_processor is not None and hasattr(
            self.vlm_processor, "image_processor"
        ):
            ip = self.vlm_processor.image_processor
            if hasattr(ip, "max_pixels") and ip.max_pixels is None:
                ip.max_pixels = 1280 * 28 * 28  # ≈ 1M pixels, safe for 80GB
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = 256 * 28 * 28  # ≈ 200K pixels

        bt.logging.info(f"Loaded VLM {self.vlm_name}")

    def load_llm(self) -> None:
        bt.logging.debug(f"Loading LLM {self.llm_name}")
        m = re.match(r"cuda:(\d+)", self.device)
        gpu_id = int(m.group(1)) if m else 0
        attn_impl = self._attn_impl()

        try:
            bt.logging.info(f"Attempting to load {self.llm_name} from local cache...")
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.bfloat16,
                device_map={"": gpu_id},
                attn_implementation=attn_impl,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name, local_files_only=True
            )
        except (OSError, ValueError):
            bt.logging.info("LLM not in local cache, downloading from HuggingFace...")
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.bfloat16,
                device_map={"": gpu_id},
                attn_implementation=attn_impl,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        self.llm = pipeline("text-generation", model=llm, tokenizer=tokenizer)
        bt.logging.info(f"Loaded LLM {self.llm_name}")

    def load_models(self) -> None:
        """Load both models if not already loaded."""
        if self.vlm is None:
            self.load_vlm()
        if self.llm is None:
            self.load_llm()
        # Fresh batch -> fresh diversity baseline.
        self._prior_image_prompts.clear()
        self._prior_video_prompts.clear()

    def unload_vlm(self) -> None:
        """Drop VLM and reclaim its VRAM."""
        bt.logging.debug("Unloading VLM from GPU")
        if self.vlm is not None:
            del self.vlm
            self.vlm = None
            self.vlm_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_llm(self) -> None:
        """Drop LLM and reclaim its VRAM."""
        bt.logging.debug("Unloading LLM from GPU")
        if self.llm is not None:
            del self.llm
            self.llm = None
        self._prior_image_prompts.clear()
        self._prior_video_prompts.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_gpu(self) -> None:
        """Drop both models and reclaim VRAM (convenience)."""
        self.unload_vlm()
        self.unload_llm()

    # ------------------------------------------------------------------
    # Two-phase batch (VLM then LLM, never both on GPU simultaneously)
    # ------------------------------------------------------------------

    def generate_scenes_batch(
        self, images: list[Image.Image]
    ) -> list[SceneDescription]:
        """VLM-only: process a batch of images, returning cached scenes.

        Loads VLM if needed. Does NOT load or touch the LLM.
        Caller should call :meth:`unload_vlm` after this to free ~9 GB.
        """
        if not images:
            return []
        if self.vlm is None:
            self.load_vlm()

        scenes: list[SceneDescription] = []
        for i, image in enumerate(images):
            try:
                scene = extract_scene_with_vlm(image, self.vlm, self.vlm_processor)
                scenes.append(scene)
            except Exception as e:
                bt.logging.warning(f"VLM failed on image {i}: {e}")
                scenes.append(None)  # type: ignore[arg-type]
                # CUDA OOM can leave the GPU in a bad state; flush so
                # subsequent VLM calls don't cascade-fail.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return scenes

    def compose_prompts_from_scenes(
        self,
        scenes: list[SceneDescription],
        specs: Optional[list[dict]] = None,
        on_composed: Optional[
            Callable[[int, Optional[SceneDescription], dict, dict[Modality, str]], None]
        ] = None,
    ) -> list[dict[Modality, str]]:
        """LLM-only: compose (image, video) prompts from cached scenes.

        Loads LLM if needed. Does NOT load or touch the VLM.
        Caller should call :meth:`unload_llm` after this to free ~60 GB.

        Args:
            scenes: One SceneDescription (or None) per source image.
            specs: Optional list parallel to `scenes`; each entry is a dict
                mapping Modality -> PromptSpec for that scene. When omitted,
                a spec is sampled per scene per modality (so direct callers
                still get diversified output).
            on_composed: Optional callback invoked as
                ``on_composed(i, scene, scene_specs, result)`` immediately
                after each scene is composed, so callers can persist prompts
                incrementally instead of waiting for the full batch.

        Returns one dict per input scene, mapping Modality -> prompt.
        Positions with ``None`` scenes produce empty dicts.
        """
        if not scenes:
            return []
        if self.llm is None:
            self.load_llm()

        self._prior_image_prompts.clear()
        self._prior_video_prompts.clear()

        results: list[dict[Modality, str]] = []
        for i, scene in enumerate(scenes):
            if scene is None:
                results.append({})
                continue
            scene_specs = specs[i] if specs and i < len(specs) and specs[i] else {
                Modality.IMAGE: sample_spec("image"),
                Modality.VIDEO: sample_spec("video"),
            }
            result = {}
            try:
                result[Modality.IMAGE] = self._compose(
                    scene, kind="image", spec=scene_specs.get(Modality.IMAGE)
                )
            except Exception as e:
                bt.logging.warning(f"LLM composition failed for scene (image): {e}")
            try:
                result[Modality.VIDEO] = self._compose(
                    scene, kind="video", spec=scene_specs.get(Modality.VIDEO)
                )
            except Exception as e:
                bt.logging.warning(f"LLM composition failed for scene (video): {e}")
            self._log_generated_prompts(
                scene,
                result.get(Modality.IMAGE, ""),
                result.get(Modality.VIDEO, ""),
            )
            results.append(result)
            if on_composed is not None:
                try:
                    on_composed(i, scene, scene_specs, result)
                except Exception as e:
                    bt.logging.error(f"[PROMPT-GEN] on_composed callback failed: {e}")
        return results

    # ------------------------------------------------------------------
    # Core API (legacy — still used externally; delegates to batch)
    # ------------------------------------------------------------------

    def generate_scene_from_image(self, image: Image.Image) -> SceneDescription:
        """Single VLM forward pass that returns a structured SceneDescription."""
        if self.vlm is None or self.vlm_processor is None:
            self.load_vlm()
        return extract_scene_with_vlm(image, self.vlm, self.vlm_processor)

    def generate_prompts_from_image(
        self, image: Image.Image, modalities: set = None
    ) -> Dict[Modality, str]:
        """Produce one canonical prompt per modality from one image.

        Two-stage: VLM grounds perception, LLM composes prose. The VLM
        runs once per image; the LLM runs once per requested modality.
        Total per-image latency is ~6s on an 80GB card with both modalities.

        Args:
            image: Source image to analyze.
            modalities: Set of Modality values to generate prompts for.
                        Defaults to {Modality.IMAGE, Modality.VIDEO}.
        """
        if modalities is None:
            modalities = {Modality.IMAGE, Modality.VIDEO}

        scene = self.generate_scene_from_image(image)

        if self.llm is None:
            self.load_llm()

        result = {}
        if Modality.IMAGE in modalities:
            result[Modality.IMAGE] = self._compose(scene, kind="image")
        if Modality.VIDEO in modalities:
            result[Modality.VIDEO] = self._compose(scene, kind="video")

        image_prompt = result.get(Modality.IMAGE, "")
        video_prompt = result.get(Modality.VIDEO, "")
        self._log_generated_prompts(scene, image_prompt, video_prompt)
        return result

    @staticmethod
    def _log_generated_prompts(
        scene: SceneDescription, image_prompt: str, video_prompt: str
    ) -> None:
        """Emit a single human-scannable log block per source image.

        Logs at INFO so prompts surface in default pm2 output for QC.
        Format is intentionally compact (one labelled block, full prompts
        verbatim) so you can grep by `[PROMPT-GEN]` and read each
        generation top-to-bottom.
        """
        caption = (scene.caption or "").strip().replace("\n", " ")
        if len(caption) > 280:
            caption = caption[:277] + "..."

        bt.logging.info(
            "\n"
            "[PROMPT-GEN] ─────────────────────────────────────────────────\n"
            f"  scene_kind : {scene.scene_kind}\n"
            f"  subject    : {scene.subject}\n"
            f"  setting    : {scene.setting}\n"
            f"  caption    : {caption}\n"
            f"  IMAGE ({len(image_prompt)} chars):\n"
            f"    {image_prompt}\n"
            f"  VIDEO ({len(video_prompt)} chars):\n"
            f"    {video_prompt}\n"
            "[PROMPT-GEN] ─────────────────────────────────────────────────"
        )

    # ------------------------------------------------------------------
    # LLM composition
    # ------------------------------------------------------------------

    def _compose(
        self,
        scene: SceneDescription,
        *,
        kind: str,
        spec: Optional[PromptSpec] = None,
    ) -> str:
        """Single LLM forward pass that writes the canonical prompt for a modality.

        When `spec` is provided, length budget and max_new_tokens follow the
        sampled length band instead of the legacy fixed ranges, and the
        committed spec block is injected into the user message.

        Returns the LLM output verbatim (with light whitespace cleanup).
        Raises on failure; the caller is the worker loop, which logs and
        skips the item, so we don't want to silently swallow errors here.
        """
        if kind == "video":
            system = _VIDEO_SYSTEM
            max_new_tokens = 400
            temperature = 1.0
            prior = self._prior_video_prompts
            length_spec = "100-180 words"
        elif kind == "image":
            system = _IMAGE_SYSTEM
            max_new_tokens = 260
            temperature = 0.95
            prior = self._prior_image_prompts
            length_spec = "60-120 words"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if spec is not None:
            # ~1.5 tokens/word headroom above the band ceiling.
            # The spec-derived budget replaces the legacy cap (which was
            # sized for a single 60-120 image / 100-180 video band and
            # would truncate long-band compositions).
            max_new_tokens = int(spec.length_words[1] * 1.5) + 40

        user = self._build_user_message(scene, prior, length_spec, spec=spec)

        text = self._generate_once(system, user, max_new_tokens, temperature)
        ok, reason = qc_validate(text, spec)
        if not ok:
            bt.logging.debug(f"Prompt QC reject ({kind}): {reason}; retrying once")
            retry_user = (
                user
                + f"\n\nPrevious attempt rejected: {reason}. "
                "Fix exactly that problem and output the paragraph only."
            )
            text = self._generate_once(system, retry_user, max_new_tokens, temperature)
            ok, reason = qc_validate(text, spec)
            if not ok:
                raise ValueError(f"Prompt failed QC twice ({kind}): {reason}")

        prior.append(text)
        return text

    def _generate_once(
        self, system: str, user: str, max_new_tokens: int, temperature: float
    ) -> str:
        """One LLM call + cleanup. Split out so the QC retry path can reuse it."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        out = self.llm(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            return_full_text=False,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        return self._clean_output(out[0]["generated_text"])

    @staticmethod
    def _build_user_message(
        scene: SceneDescription,
        prior_prompts: Deque[str],
        length_spec: str,
        spec: Optional["PromptSpec"] = None,
    ) -> str:
        """Build the user turn: committed spec + scene facts + dense caption + prior-prompt feedback.

        `spec` (when provided) carries the sampled per-prompt axes —
        register, camera behavior, length band, event budget, style
        strictness — as COMMITTED facts the LLM must render. Sampling
        these outside the LLM is what breaks the single-register,
        single-voice monoculture: the LLM is a strong renderer but a
        weak diversity source.

        `prior_prompts` is the recent per-modality composition history
        for the current batch (sliding window, capped at `_PRIOR_WINDOW`
        most recent, verbatim). Re-injecting it lets the LLM actively
        diversify against its own past output along every dimension
        (shot type, camera move, lens, lighting, palette, opener,
        vocabulary, sentence structure) without us hardcoding any
        option lists.

        `length_spec` restates the system-prompt word range here in the
        user turn because LLMs weight closer-to-output instructions more
        heavily, and the verbose prior-prompt history was inflating
        outputs above the system-prompt cap. When `spec` is provided it
        supersedes `length_spec`.
        """
        scene_dict = asdict(scene)
        # Drop the dense caption from the JSON since we surface it
        # explicitly below; keeps the JSON focused on structured facts.
        caption = scene_dict.pop("caption", "")
        scene_json = json.dumps(scene_dict, indent=2, ensure_ascii=False)

        parts: List[str] = []

        if spec is not None:
            lo, hi = spec.length_words
            spec_lines = [
                "COMMITTED SHOT SPEC (authoritative — render exactly this, "
                "do not negotiate):",
                f"- register: {spec.register} — {spec.register_directives}",
            ]
            if spec.modality == "video":
                spec_lines.append(f"- camera: {spec.camera_motion}")
                if spec.event_count > 0:
                    event_line = (
                        f"- events: exactly {spec.event_count} discrete "
                        "event(s) must occur mid-clip (someone/something "
                        "enters, exits, passes, falls, or changes state — "
                        "pick what fits the scene)"
                    )
                    if scene.plausible_events:
                        event_line += (
                            "; draw from: "
                            + "; ".join(scene.plausible_events)
                        )
                    spec_lines.append(event_line)
                else:
                    spec_lines.append(
                        "- events: none — ambient motion only, nothing "
                        "discrete happens"
                    )
            spec_lines.append(f"- length: {lo}-{hi} words")
            if spec.style_strictness == "plain":
                banned = ", ".join(f'"{p}"' for p in spec.banned_phrases)
                spec_lines.append(
                    "- style: plain, factual, non-literary. Forbidden in "
                    f"this register: similes, negation litanies, closing "
                    f"aphorisms, and the phrases: {banned}"
                )
            else:
                spec_lines.append(
                    "- style: free — match the register's natural voice"
                )
            parts += ["\n".join(spec_lines), ""]

        parts += [
            "Reference image, structured scene facts (from VLM):",
            scene_json,
            "",
            "Dense caption of the same image (from VLM):",
            caption,
        ]

        if prior_prompts:
            history = "\n\n".join(
                f"[{i}] {p}" for i, p in enumerate(prior_prompts, start=1)
            )
            parts += [
                "",
                "Your earlier compositions in this batch are below. Make THIS "
                "composition demonstrably different from them across shot "
                "framing, camera movement, focal length, lighting, color "
                "palette, mood, opening sentence structure, and vocabulary. "
                "Do not echo their phrasings. Diverse output across the "
                "batch is required.",
                "",
                history,
            ]

        if spec is not None:
            lo, hi = spec.length_words
            parts += [
                "",
                f"Strict length: {lo}-{hi} words. Do not exceed.",
                "Compose the prompt now. Output the paragraph only.",
            ]
        else:
            parts += [
                "",
                f"Strict length: {length_spec}. Do not exceed.",
                "Compose the prompt now. Output the paragraph only.",
            ]
        return "\n".join(parts)

    @staticmethod
    def _clean_output(text: str) -> str:
        """Light whitespace + quote cleanup; preserve LLM word choice."""
        text = text.strip()
        # Strip wrapping quotes (single, double, smart) if the model added them.
        text = text.strip("\"'\u201c\u201d ")
        # Collapse internal newlines (we asked for one paragraph).
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return text
