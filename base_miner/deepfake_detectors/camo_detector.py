import torch
from PIL import Image
from detector_registry import DETECTOR_REGISTRY
from ultralytics import YOLO

# Paths for pre-trained UCF checkpoints
from base_miner.UCF.config.constants import (
    BM_FACE_CKPT,
    BM_18K_CKPT
)
from deepfake_detector import DeepfakeDetector
from ucf_detector import UCFDetector
from npr_detector import NPRDetector

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
        self.detectors = {}
        self.detector_configs = detector_configs or {
            'face': {'UCF': BM_FACE_CKPT},   # Default model for 'face'
            'general': {'UCF': BM_18K_CKPT}  # Default model for 'general'
        }
        super().__init__(model_name)

    def load_model(self):
        """
        Load detectors dynamically based on the provided configuration and registry.
        """
        print(f"DETECTOR_REGISTRY: ", DETECTOR_REGISTRY.data.keys())
        for content_type, detector_info in self.detector_configs.items():
            for detector_name, weight_path in detector_info.items():
                if detector_name in DETECTOR_REGISTRY:
                    print(f"Loading {detector_name} model for {content_type} detection.")
                    detector_class = DETECTOR_REGISTRY[detector_name]
                    self.detectors[content_type] = detector_class(
                        model_name=f'{detector_name}_{content_type.capitalize()}',
                        ucf_checkpoint_name=weight_path  # Use the weight path from config
                    )
                else:
                    raise ValueError(f"Detector {detector_name} not found in the registry for {content_type}.")

    def classify_image(self, image, use_object_detection=True):
        """
        Classify the image to determine its content type.

        Args:
            image (PIL.Image): The image to analyze.

        Returns:
            str: 'face' if the image contains at least one face, otherwise 'general'.
            list: List of detected face data, or None if no faces are detected
        """

        faces, num_faces = self.detectors['face'].detect_faces(image)

        if use_object_detection:
            try:
                results = self.object_detector(image)
            except Exception as e:
                print(f"Error in object detection: {e}")
                return 'general', None

            detected_classes = []
            try:
                for result in results:
                    for box in result.boxes:
                        try:
                            if box.conf.item() is not None and box.conf.item() > 0.5:
                                detected_classes.append(result.names[box.cls.item()])
                        except Exception as e:
                            print(f"Error processing object detection box: {e}")
                            continue
            except Exception as e:
                print(f"Error processing object detection results: {e}")
                return 'general', None

            try:
                if 'person' in detected_classes and num_faces:
                    return 'face', faces
                return 'general', None
            except Exception as e:
                print(f"Error checking detected classes: {e}")
                return 'general', None

        else:
            try:
                if num_faces:
                    return 'face', faces
                return 'general', None
            except Exception as e:
                print(f"Error in non-object detection branch: {e}")
                return 'general', None

    def __call__(
            self, image
    ) -> float:
        try:
            # Determine image content type.
            image_type, faces = self.classify_image(image, use_object_detection=False)
            image_tensor = \
            self.detectors[image_type].preprocess(image, faces=faces if image_type=="face" else None)
            pred = self.detectors[image_type].infer(image_tensor)
        except Exception as e:
            print("Error performing inference")
            print(e)
        return pred