import torch
from diffusers import MotionAdapter, HunyuanVideoTransformer3DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_hunyuanvideo_transformer(
      model_id: str = "tencent/HunyuanVideo",
      subfolder: str = "transformer",
      torch_dtype: torch.dtype = torch.bfloat16, 
      revision: str = 'refs/pr/18'
):
    return HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder=subfolder, torch_dtype=torch_dtype, revision=revision
    )


def load_annimatediff_motion_adapter(
    step: int = 4
) -> MotionAdapter:
    """
    Load a motion adapter model for AnimateDiff.

    Args:
        step: The step size for the motion adapter. Options: [1, 2, 4, 8].
        repo: The HuggingFace repository to download the motion adapter from.
        ckpt: The checkpoint filename
    Returns:
        A loaded MotionAdapter model.

    Raises:
        ValueError: If step is not one of [1, 2, 4, 8].
    """
    if step not in [1, 2, 4, 8]:
        raise ValueError("Step must be one of [1, 2, 4, 8]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = MotionAdapter().to(device, torch.float16)

    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    adapter.load_state_dict(
        load_file(
            hf_hub_download(repo, ckpt),
            device=device
        )
    )
    return adapter
