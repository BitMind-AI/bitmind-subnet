import torch
from pathlib import Path

import bittensor as bt
from base_miner import DETECTOR_REGISTRY
from base_miner.DFB.config.constants import CONFIGS_DIR, WEIGHTS_DIR
from base_miner.DFB.detectors import DETECTOR, TALLDetector
from base_miner.deepfake_detectors import DeepfakeDetector


@DETECTOR_REGISTRY.register_module(module_name="TALL")
class TALLVideoDetector(DeepfakeDetector):
    def __init__(
        self,
        model_name: str = "TALL",
        config: str = "tall.yaml",
        device: str = "cpu",
    ):
        super().__init__(model_name, config, device)

        total_params = sum(p.numel() for p in self.tall.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.tall.model.parameters() if p.requires_grad
        )
        bt.logging.info('device:', self.device)
        bt.logging.info(total_params, "parameters")
        bt.logging.info(trainable_params, "trainable parameters")

    def load_model(self):
        self.ensure_weights_are_available(WEIGHTS_DIR, self.weights)
        bt.logging.info(f"Loaded config from training run: {self.config}")
        self.tall = TALLDetector(self.config, self.device)
        checkpoint_path = Path(WEIGHTS_DIR) / self.weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.tall.load_state_dict(checkpoint, strict=True)
        self.tall.model.eval()

    def __call__(self, data_dict):
        with torch.no_grad():
            return self.tall.forward(data_dict, inference=True)["prob"]
