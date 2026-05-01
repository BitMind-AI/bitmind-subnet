"""Public surface for the gas.generation package.

Internal layout:
    prompts/   - canonical prompt production (VLM + LLM) and per-local-model
                 prompt adaptation
    media/     - local diffusers runtime, model registry, SoTA API clients
    util/      - shared helpers (image, model loading, prompt truncation)

Re-exports below preserve the historical flat import paths so external
callers can keep using `from gas.generation import PromptGenerator` etc.
"""

from .media.generation_pipeline import GenerationPipeline
from .media.models import initialize_model_registry
from .prompts.model_prompt_styles import AdaptedPrompt, adapt_for_local_model
from .prompts.prompt_generator import PromptGenerator
from .prompts.scene import SceneDescription, extract_scene_with_vlm
