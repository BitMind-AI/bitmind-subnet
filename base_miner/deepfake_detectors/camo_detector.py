from pathlib import Path
import yaml
import torch
from PIL import Image
from base_miner import DETECTOR_REGISTRY, GATE_REGISTRY
from base_miner.deepfake_detectors import DeepfakeDetector


@DETECTOR_REGISTRY.register_module(module_name='CAMO')
class CAMODetector(DeepfakeDetector):
    def __init__(self, model_name: str = 'CAMO', config: str = 'camo.yaml', cuda: bool = True):
        """
        Initialize the CAMODetector with dynamic model selection based on config.
        """
        self.model_name = model_name
        self.gate = GATE_REGISTRY["GATING_MECHANISM"]()
        self.detectors = {}
        super().__init__(model_name, config, cuda)
    
    def load_model(self):
        """
        Load detectors dynamically based on the provided configuration and registry.
        """
        for content_type, detector_info in self.content_type.items():
            model_name = detector_info['model']
            detector_config = detector_info['detector_config']
            
            if model_name in DETECTOR_REGISTRY:
                print(f"Loading {model_name} model for {content_type} detection with config {detector_config}.")
                self.detectors[content_type] = DETECTOR_REGISTRY[model_name](
                    model_name=f'{model_name}_{content_type.capitalize()}',
                    config=detector_config,
                    cuda=self.device
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