import torch
from pathlib import Path

import bittensor as bt
from base_miner.registry import DETECTOR_REGISTRY
from base_miner.DFB.config.constants import CONFIGS_DIR, WEIGHTS_DIR
from base_miner.DFB.detectors import DETECTOR, TALLDetector
from base_miner.deepfake_detectors import DeepfakeDetector
from bitmind.utils.video_utils import pad_frames


@DETECTOR_REGISTRY.register_module(module_name="TALL")
class TALLVideoDetector(DeepfakeDetector):
    def __init__(
        self,
        model_name: str = "TALL",
        config_name: str = "tall.yaml",
        device: str = "cpu",
    ):
        super().__init__(model_name, config_name, device)

        total_params = sum(p.numel() for p in self.tall.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.tall.model.parameters() if p.requires_grad
        )
        bt.logging.info('device:', self.device)
        bt.logging.info(total_params, "parameters")
        bt.logging.info(trainable_params, "trainable parameters")

    def load_model(self):
        # download weights from hf if not available locally
        self.ensure_weights_are_available(WEIGHTS_DIR, self.weights)
        bt.logging.info(f"Loaded config: {self.config}")
        self.tall = TALLDetector(self.config, self.device)

        # load weights
        checkpoint_path = Path(WEIGHTS_DIR) / self.weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.tall.load_state_dict(checkpoint, strict=True)
        self.tall.model.eval()

    def preprocess(self, frames_tensor):
        """ Prepare input data dict for TALLDetector """ 
        frames_tensor = pad_frames(frames_tensor, 4)
        return {'image': frames_tensor}

    def __call__(self, frames_tensor):
        input_data = self.preprocess(frames_tensor)
        with torch.no_grad():
            output_data = self.tall.forward(input_data, inference=True)
        return output_data['prob'][0]
