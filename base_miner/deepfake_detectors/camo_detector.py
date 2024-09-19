from pathlib import Path
import yaml
import torch
from PIL import Image
from base_miner import DETECTOR_REGISTRY, GATE_REGISTRY
from base_miner.deepfake_detectors import DeepfakeDetector


@DETECTOR_REGISTRY.register_module(module_name='CAMO')
class CAMODetector(DeepfakeDetector):
    """
    This DeepfakeDetector subclass implements Content-Aware Model Orchestration
    (CAMO), a mixture-of-experts approach to the binary classification of
    real and fake images, breaking the classification problem into content-specific
    subproblems.
    
    The subproblems are solved by using a GatingMechanism to route image
    content to appropriate DeepfakeDetector subclass instance(s) that
    initialize models pretrained to handle the content type.
    
    Attributes:
        model_name (str): Name of the detector instance.
        config (str): Name of the YAML file in deepfake_detectors/config/ to load
                      attributes from.
        cuda (bool): Whether to enable cuda (GPU).
    """
    
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
            pred = self.detectors[content_type](content_data)
        except Exception as e:
            print(f"Error performing inference: {e}")
        return pred