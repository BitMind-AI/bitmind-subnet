"""
Runway ML text-to-video via the official runwayml Python SDK.

Uses RUNWAYML_API_SECRET (see https://docs.dev.runwayml.com/api/).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, cast

import bittensor as bt
import requests

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask

# Default matches Runway docs / common preset (veo3.1)
DEFAULT_MODEL = "veo3.1"
DEFAULT_RATIO = "1280:720"
DEFAULT_DURATION_VEO = 8

# Runway POST /v1/text_to_video — accepted ``model`` values (API contract).
TEXT_TO_VIDEO_MODELS: frozenset[str] = frozenset(
    {"gen4.5", "veo3.1", "veo3.1_fast", "veo3"}
)

VEO31_RATIOS = frozenset({"1280:720", "720:1280", "1080:1920", "1920:1080"})
GEN45_RATIOS = frozenset({"1280:720", "720:1280"})
VEO_DURATIONS = (4, 6, 8)


def canonical_text_to_video_model(model: Any) -> str:
    """
    Resolve ``parameters.model`` to a supported Runway model id.

    Matching is case-insensitive (e.g. ``VEO3.1`` → ``veo3.1``). Empty / missing
    uses ``DEFAULT_MODEL``. Unknown values raise ``ValueError`` with the accepted list.
    """
    if model is None or (isinstance(model, str) and not model.strip()):
        return DEFAULT_MODEL
    raw = str(model).strip()
    if raw in TEXT_TO_VIDEO_MODELS:
        return raw
    lower = raw.lower()
    for mid in TEXT_TO_VIDEO_MODELS:
        if mid.lower() == lower:
            return mid
    accepted = ", ".join(sorted(TEXT_TO_VIDEO_MODELS))
    raise ValueError(
        f"Unsupported Runway text-to-video model {raw!r}. Accepted values: {accepted}"
    )


class RunwayService(BaseGenerationService):
    """
    Text-to-video generation using Runway's `/v1/text_to_video` API.

    Request parameters (via task.parameters) align with the API / SDK:
      - model: required string; must be one of TEXT_TO_VIDEO_MODELS —
        ``gen4.5``, ``veo3.1``, ``veo3.1_fast``, ``veo3`` (case-insensitive).
      - ratio: aspect string (model-specific; default 1280:720)
      - duration: seconds (constraints depend on model)
      - audio: bool (veo3.1 / veo3.1_fast)
      - seed: optional (gen4.5)

    Polling uses the SDK's ``wait_for_task_output()`` (see runwayml.lib.polling).
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv("RUNWAYML_API_SECRET")
        self._client = None
        self.download_timeout = 600

        if self.api_key and self.api_key.strip():
            try:
                from runwayml import RunwayML

                self._client = RunwayML()
                bt.logging.info("RunwayService initialized (RUNWAYML_API_SECRET set)")
            except ImportError:
                bt.logging.warning(
                    "runwayml package not installed. Install with: pip install runwayml"
                )
            except Exception as e:
                bt.logging.error(f"Failed to initialize RunwayML client: {e}")
        else:
            bt.logging.warning("RUNWAYML_API_SECRET not set; RunwayService unavailable")

    def is_available(self) -> bool:
        return (
            self.api_key is not None
            and self.api_key.strip() != ""
            and self._client is not None
        )

    def supports_modality(self, modality: str) -> bool:
        return modality == "video"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": [],
            "video": ["text_to_video"],
        }

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["text_to_video_models"] = sorted(TEXT_TO_VIDEO_MODELS)
        return info

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {
            "RUNWAYML_API_SECRET": "Runway API key for text-to-video (RUNWAY Gen / Veo, etc.)"
        }

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality != "video":
            raise ValueError(
                f"RunwayService supports video only, got modality={task.modality}"
            )
        return self._text_to_video(task)

    def _text_to_video(self, task: GenerationTask) -> Dict[str, Any]:
        from runwayml.lib.polling import TaskFailedError, TaskTimeoutError

        if self._client is None:
            raise RuntimeError("RunwayML client not initialized")

        params = task.parameters or {}
        poll_timeout = float(params.get("runway_poll_timeout", params.get("timeout", 900.0)))
        if poll_timeout <= 0:
            poll_timeout = None  # SDK: wait indefinitely

        model = canonical_text_to_video_model(params.get("model"))
        bt.logging.info(f"Runway text-to-video: model={model}")

        create_kwargs = self._build_create_kwargs(task.prompt, params, model)
        start = time.time()

        try:
            created = self._client.text_to_video.create(**create_kwargs)
            task_result = created.wait_for_task_output(timeout=poll_timeout)
        except TaskFailedError as e:
            detail = getattr(e.task_details, "failure", str(e))
            code = getattr(e.task_details, "failure_code", None)
            bt.logging.error(f"Runway task failed: {detail} (code={code})")
            raise RuntimeError(f"Runway video generation failed: {detail}") from e
        except TaskTimeoutError as e:
            bt.logging.error("Runway task polling timed out")
            raise TimeoutError("Runway video generation timed out") from e

        output_urls = cast(List[str], getattr(task_result, "output", None) or [])
        if not output_urls:
            raise RuntimeError("Runway succeeded but returned no output URLs")

        url = output_urls[0]
        bt.logging.info(f"Downloading Runway output ({len(output_urls)} asset(s)), first URL ...")

        resp = requests.get(url, timeout=self.download_timeout)
        resp.raise_for_status()
        video_bytes = resp.content

        elapsed = time.time() - start
        bt.logging.success(f"Runway video downloaded: {len(video_bytes)} bytes in {elapsed:.1f}s")

        return {
            "data": video_bytes,
            "metadata": {
                "model": model,
                "provider": "runwayml",
                "mime_type": "video/mp4",
                "runway_task_id": getattr(task_result, "id", None),
                "source_url_used": url[:80] + "..." if len(url) > 80 else url,
                "generation_time": elapsed,
            },
        }

    def _build_create_kwargs(
        self, prompt: str, params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """Map miner parameters to runwayml ``text_to_video.create`` keyword arguments."""

        # model is already canonical and in TEXT_TO_VIDEO_MODELS
        if model in ("veo3.1", "veo3.1_fast"):
            ratio = self._pick_ratio(
                params.get("ratio", DEFAULT_RATIO), VEO31_RATIOS, DEFAULT_RATIO
            )
            dur_raw = params.get("duration", DEFAULT_DURATION_VEO)
            duration = self._nearest_veo_duration(int(dur_raw))
            kwargs: Dict[str, Any] = {
                "model": cast(Any, model),
                "prompt_text": prompt,
                "ratio": ratio,
                "duration": cast(Any, duration),
            }
            if "audio" in params:
                kwargs["audio"] = bool(params["audio"])
            return kwargs

        if model == "gen4.5":
            ratio = self._pick_ratio(
                params.get("ratio", DEFAULT_RATIO), GEN45_RATIOS, "1280:720"
            )
            dur = int(params.get("duration", 5))
            dur = max(2, min(10, dur))
            kwargs = {
                "model": "gen4.5",
                "prompt_text": prompt,
                "ratio": cast(Any, ratio),
                "duration": dur,
            }
            if "seed" in params:
                kwargs["seed"] = int(params["seed"])
            return kwargs

        if model == "veo3":
            ratio = self._pick_ratio(
                params.get("ratio", DEFAULT_RATIO), VEO31_RATIOS, DEFAULT_RATIO
            )
            return {
                "model": "veo3",
                "prompt_text": prompt,
                "ratio": ratio,
                "duration": 8,
            }

        # Exhaustive if TEXT_TO_VIDEO_MODELS stays in sync with API
        raise AssertionError(f"Unhandled canonical model {model!r}")

    @staticmethod
    def _nearest_veo_duration(seconds: int) -> int:
        return min(VEO_DURATIONS, key=lambda x: abs(x - int(seconds)))

    @staticmethod
    def _pick_ratio(value: Any, allowed: frozenset, default: str) -> str:
        s = str(value).strip()
        if s in allowed:
            return s
        bt.logging.warning(f"Ratio {value!r} not valid for this model; using {default}")
        return default
