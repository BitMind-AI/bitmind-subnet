import torch
from PIL import Image

# Paths for pre-trained UCF checkpoints
from base_miner.UCF.config.constants import (
    BM_FACE_CKPT,
    BM_18K_CKPT
)

from base_miner import DETECTOR_REGISTRY, GATE_REGISTRY
from base_miner.deepfake_detectors import DeepfakeDetector

@DETECTOR_REGISTRY.register_module(module_name='CAMO')
class CAMODetector(DeepfakeDetector):
    def __init__(self, model_name: str = 'CAMO', detector_configs: dict = None):
        """
        Initialize the CAMODetector with dynamic model selection based on detector_configs.
        
        Args:
            model_name (str): The name of the CAMO model.
            detector_configs (Dict): A dictionary where keys represent content types 
                                     (e.g., 'face', 'general') and values represent 
                                     the detector names to use from the registry.
        """
        self.model_name = model_name
        self.gate = GATE_REGISTRY["GATING_MECHANISM"]()
        self.detectors = {}
        self.detector_configs = detector_configs or {
            'face': {'model': 'UCF',
                     'weights': BM_FACE_CKPT,
                     'config_name': 'pretrained_config.yaml'},  # Default model for 'face'
            'general': {'model': 'UCF',
                        'weights': BM_18K_CKPT,
                        'config_name': 'pretrained_config.yaml'}  # Default model for 'general'
        }
        super().__init__(model_name)

    def load_model(self):
        """
        Load detectors dynamically based on the provided configuration and registry.
        """
        for content_type, detector_info in self.detector_configs.items():
            model_name = detector_info['model']
            weight_path = detector_info['weights']
            config_name = detector_info['config_name']
            if model_name in DETECTOR_REGISTRY:
                print(f"Loading {model_name} model for {content_type} detection with config {config_name}.")
                self.detectors[content_type] = DETECTOR_REGISTRY[model_name](
                    model_name=f'{model_name}_{content_type.capitalize()}',
                    ucf_checkpoint_name=weight_path,
                    config_name=config_name
                )
            else:
                raise ValueError(f"Detector {model_name} not found in the registry for {content_type}.")

    def __call__(
            self, image
    ) -> float:
        try:
            # Determine image content type.
            content_type, content_data = self.gate(image, use_object_detection=False)
            print(f"content_type: {content_type}")
            print(f"content_data: {content_data}")

            pred = self.detectors[content_type](content_data)

        except Exception as e:
            print("Error performing inference")
            print(e)
        return pred