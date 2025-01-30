import torch
import numpy as np
from diffusers import Pipeline
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import PIL.Image


class JanusPipeline(Pipeline):
    def __init__(self, model: MultiModalityCausalLM, processor: VLChatProcessor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 16,
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
        input_ids = torch.LongTensor(input_ids)

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
            
        return type('GenerationOutput', (), {'images': images})()

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        processor = VLChatProcessor.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=kwargs.get('torch_dtype', torch.bfloat16)
        ).cuda().eval()
        
        return cls(model=model, processor=processor) 