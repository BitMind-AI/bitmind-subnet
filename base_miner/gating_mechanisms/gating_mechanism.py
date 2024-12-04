from PIL import Image
from base_miner.registry import GATE_REGISTRY


class GatingMechanism:
    """
    This class orchestrates multi-gate content detection and content-specific
    preprocessing to facilitate use by downstream models

    This is useful for routing images to appropriate detectors
    trained to handle different content types in a mixture-of-experts
    framework such as Content-Aware Model Orchestration (CAMO).
    """
    def __init__(self, gate_names: list):
        self.gates = {
            gate: GATE_REGISTRY[gate.upper()]()
            for gate in gate_names
        }

    def __call__(self, image: Image):
        gate_results = {}
        for gate_name, gate in self.gates.items():
            gate_output_image, gate_activated = gate(image)
            if gate_activated:
                gate_results[gate_name] = gate_output_image

        return gate_results
