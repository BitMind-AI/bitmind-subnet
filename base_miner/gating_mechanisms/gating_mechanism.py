import torch
from PIL import Image
from ultralytics import YOLO
from base_miner.gating_mechanisms import Gate, FaceGate
from base_miner import GATE_REGISTRY


@GATE_REGISTRY.register_module(module_name='GATING_MECHANISM')
class GatingMechanism(Gate):
    """
    This Gate subclass orchestrates multi-gate content detection
    and content-specific preprocessing.

    This is useful for routing images to appropriate detectors
    trained to handle different content types in a mixture-of-experts
    framework such as Content-Aware Model Orchestration (CAMO).

    Attributes:
        gate_name (str): The name of the gate.
        gates_dict (dict): Dictionary of gate subclasses.
    """
    
    def __init__(
        self, 
        gate_name: str = 'GatingMechanism', 
        gates_dict = GATE_REGISTRY.data.copy(), 
        object_detection: bool = False
    ):
        self.gates_dict = gates_dict
        self.gates_dict.pop('GATING_MECHANISM', None)
        self.gates = {}
        self.object_detector = YOLO("yolov8x.pt") if object_detection else None
        self.load_gates()
        super().__init__(gate_name, "all")

    def load_gates(self):
        """
        Load detectors dynamically based on the provided configuration and registry.
        """
        for gate_name in self.gates_dict.keys():
            try:
                self.gates[gate_name] = self.gates_dict[gate_name]()
            except Exception as e:
                print(f"Gate {gate_name} not found in the registry: {e}")
    
    def classify_image(self, image):
        """
        Classify the image to determine its content type.

        Args:
            image (PIL.Image): The image to analyze.

        Returns:
            str: Content name, e.g. 'face' if the image contains at least one face, otherwise 'general'.
            list: List of detected data or None if no content detected
        """
        gate_results = {}
        for gate_name in self.gates.keys():
            gate_results[gate_name] = {"content": self.gates[gate_name].detect_content_type(image),
                                        "processed_image": self.gates[gate_name](image)}

        if self.object_detector:
            try:
                results = self.object_detector(image)
            except Exception as e:
                print(f"Error in object detection: {e}")
                return 'general', None

            detected_content = []
            try:
                for result in results:
                    for box in result.boxes:
                        try:
                            if box.conf.item() is not None and box.conf.item() > 0.5:
                                detected_content.append(result.names[box.cls.item()])
                        except Exception as e:
                            print(f"Error processing object detection box: {e}")
                            continue
            except Exception as e:
                print(f"Error processing object detection results: {e}")
                return 'general', None

            try:
                if 'person' in detected_content and gate_results["FACE"]["content"]:
                    return self.gates["FACE"].content_type, gate_results["FACE"]["processed_image"]
                return 'general', None
            except Exception as e:
                print(f"Error checking detected content: {e}")
                return 'general', None

        else:
            try:
                if len(gate_results["FACE"]["content"]):
                    return self.gates["FACE"].content_type, gate_results["FACE"]["processed_image"]
                return 'general', None
            except Exception as e:
                print(f"Error in non-object detection branch: {e}")
                return 'general', None

    def detect_content_type(self, image: Image):
        """Detect the content type of the image."""
        return self.classify_image(image)[0]

    def preprocess(self, image: Image):
        """Preprocess the image based on its content type."""
        return self.classify_image(image)[1]

    def __call__(self, image: Image):
        return self.classify_image(image)