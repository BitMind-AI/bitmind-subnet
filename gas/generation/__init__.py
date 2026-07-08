"""Public surface for the gas.generation package.

Internal layout:
    prompts/   - canonical prompt production (VLM + LLM) and per-local-model
                 prompt adaptation
    media/     - local diffusers runtime, model registry, SoTA API clients
    util/      - shared helpers (image, model loading, prompt truncation)

Re-exports below preserve the historical flat import paths so external
callers can keep using `from gas.generation import PromptGenerator` etc.
They are resolved lazily (PEP 562) so that importing a pure-python
submodule (e.g. gas.generation.prompts.prompt_qc) does not drag in the
torch/diffusers/janus runtime stack.
"""

_LAZY_EXPORTS = {
    "GenerationPipeline": "gas.generation.media.generation_pipeline",
    "initialize_model_registry": "gas.generation.media.models",
    "AdaptedPrompt": "gas.generation.prompts.model_prompt_styles",
    "adapt_for_local_model": "gas.generation.prompts.model_prompt_styles",
    "PromptGenerator": "gas.generation.prompts.prompt_generator",
    "SceneDescription": "gas.generation.prompts.scene",
    "extract_scene_with_vlm": "gas.generation.prompts.scene",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
