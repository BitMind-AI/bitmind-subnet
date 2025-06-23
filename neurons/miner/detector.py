import json
import torch
import bittensor as bt
from torchvision import models


class Detector:
    """Handler for image and video detection models."""
    
    def __init__(self, config):
        self.config = config
        self.image_detector = None
        self.video_detector = None
        self.device = (
            self.config.device
            if hasattr(self.config, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.load_model()

    def load_model(self, modality=None):
        """Load the appropriate detection model based on modality.

        MINER TODO:
            This class has placeholder models to demonstrate the required outputs
            for validator requests. They have not been trained and will perform
            poorly. Your task is to train performant video and image detection
            models and load them here. Happy mining!

        Args:
            modality (str): Type of detection model to load ('image' or 'video')
        """
        bt.logging.info(f"Loading {modality} detection model...")
        if modality in ("image", None):
            ### REPLACE WITH YOUR OWN MODEL
            self.image_detector = models.resnet50(pretrained=True)
            num_ftrs = self.image_detector.fc.in_features
            self.image_detector.fc = torch.nn.Linear(num_ftrs, 3)
            self.image_detector = self.image_detector.to(self.device)
            self.image_detector.eval()

        if modality in ("video", None):
            ### REPLACE WITH YOUR OWN MODEL
            self.video_detector = models.video.r3d_18(pretrained=True)
            num_ftrs = self.video_detector.fc.in_features
            self.video_detector.fc = torch.nn.Linear(num_ftrs, 3)
            self.video_detector = self.video_detector.to(self.device)
            self.video_detector.eval()

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def preprocess(self, media_tensor, modality):
        bt.logging.debug(
            json.dumps(
                {
                    "modality": modality,
                    "shape": tuple(media_tensor.shape),
                    "dtype": str(media_tensor.dtype),
                    "min": torch.min(media_tensor).item(),
                    "max": torch.max(media_tensor).item(),
                },
                indent=2,
            )
        )

        if modality == "image":
            media_tensor = media_tensor.unsqueeze(0).float().to(self.device)
        elif modality == "video":
            media_tensor = media_tensor.unsqueeze(0).float().to(self.device)
        return media_tensor

    def detect(self, media_tensor, modality):
        """Perform inference with either self.video_detector or self.image_detector

        MINER TODO: Update detection logic as necessary for your own model

        Args:
            media_tensor (torch.tensor): Input media tensor
            modality (str): Type of detection to perform ('image' or 'video')

        Returns:
            torch.Tensor: Probability vector containing 3 class probabilities
                [p_real, p_synthetic, p_semisynthetic]
        """
        media_tensor = self.preprocess(media_tensor, modality)

        if modality == "image":
            if self.image_detector is None:
                self.load_model("image")

            bt.logging.debug(
                f"Running image detection on array shape {media_tensor.shape}"
            )

            # MINER TODO update detection logic as necessary
            with torch.no_grad():
                outputs = self.image_detector(media_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

        elif modality == "video":
            if self.video_detector is None:
                self.load_model("video")

            bt.logging.debug(
                f"Running video detection on array shape {media_tensor.shape}"
            )

            # MINER TODO update detection logic as necessary
            with torch.no_grad():
                outputs = self.video_detector(media_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        bt.logging.success(f"Detection prediction: {probs}")
        return probs