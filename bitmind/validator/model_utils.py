import torch
import numpy as np
from diffusers import MotionAdapter, HunyuanVideoTransformer3DModel, DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
import PIL.Image


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


class JanusWrapper(DiffusionPipeline):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.register_modules(
            model=model,
            processor=processor,
            tokenizer=self.processor.tokenizer
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 4,
        cfg_weight: float = 5.0,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        **kwargs
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.processor.image_start_tag

        input_ids = self.processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(self.device)
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(self.device)
        outputs = None

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state
            
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        images = []
        for i in range(parallel_size):
            images.append(PIL.Image.fromarray(dec[i].astype(np.uint8)))
            
        # Return object with images attribute
        class Output:
            def __init__(self, images):
                self.images = images

        return Output(images)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model, processor = load_janus_model(model_path, **kwargs)
        return cls(model=model, processor=processor)
        
    def to(self, device):
        self.model = self.model.to(device)
        return self


def load_janus_model(model_path: str, **kwargs):
    processor = VLChatProcessor.from_pretrained(model_path)
    
    # Filter kwargs to only include what Janus expects
    janus_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': kwargs.get('torch_dtype', torch.bfloat16)
    }
    
    # Let device placement be handled by diffusers like other models
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        **janus_kwargs
    ).eval()
    
    return model, processor
