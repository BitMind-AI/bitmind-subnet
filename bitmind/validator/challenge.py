import random
from typing import Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image

import numpy as np
import pandas as pd
import wandb
import bittensor as bt

from bitmind.protocol import prepare_synapse
from bitmind.utils.image_transforms import apply_augmentation_by_level
from bitmind.utils.uids import get_random_uids
from bitmind.validator.reward import get_rewards
from bitmind.validator.config import (
    TARGET_IMAGE_SIZE, 
    MIN_FRAMES,
    MAX_FRAMES,
    P_STITCH,
    LABELS,
    LABEL_TO_TYPE,
    LABEL_PROBS,
    Modality,
    MODALITY_PROBS,
)


@dataclass
class ChallengeConfig:
    """Configuration parameters for challenge generation."""
    target_image_size: Tuple[int] = TARGET_IMAGE_SIZE
    modality_options: Tuple[int] = field(default_factory=lambda: [m.value for m in Modality])
    modality_probs: List[float] = MODALITY_PROBS
    label_options: Tuple[int] = LABELS
    label_probs: Tuple[float] = LABEL_PROBS
    label_to_type: Dict[int, str] = field(default_factory=lambda: LABEL_TO_TYPE)
    min_frames: int = MIN_FRAMES
    max_frames: int = MAX_FRAMES
    min_fps: int = 8
    max_fps: int = 30
    stitch_prob: float = P_STITCH


@dataclass
class Challenge:
    """
    Container for challenge data and metadata.

    A challenge consists of either an image or video that needs to be classified
    as real, synthetic, or semisynthetic. The class manages the challenge lifecycle
    including creation, data processing, and metadata handling.

    Attributes:
        label (int): Label value (0=real, 1=synthetic, 2=semisynthetic)
        media_type (str): Type of media ('real', 'synthetic', 'semisynthetic')
        modality (str): Media modality ('image' or 'video')
        original_media (Union[Image.Image, List[Image.Image], None]): The actual image or
            video frames
        original_media (Union[Image.Image, List[Image.Image], None]): The actual image or
            video frames with transformations and augmentations applied
        metadata (Dict[str, Any]): Additional information about the challenge
        config (ChallengeConfig): Configuration parameters for challenge generation
    """
    label: int = -1
    media_type: str = ""
    modality: str = ""
    original_media: Union[Image.Image, List[Image.Image], None] = None
    augmented_media: Union[Image.Image, List[Image.Image], None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    config: ChallengeConfig = field(default_factory=ChallengeConfig)

    @classmethod
    def create(cls, media_cache):
        """Factory method to create and initialize a challenge."""
        challenge = cls()

        challenge.label = np.random.choice(
            challenge.config.label_options, 
            p=challenge.config.label_probs
        )
        challenge.media_type = challenge.config.label_to_type[challenge.label]
        challenge.modality = np.random.choice(
            challenge.config.modality_options, 
            p=challenge.config.modality_probs
        )

        bt.logging.info(f"Sampling data from {challenge.modality} cache")
        cache = media_cache[challenge.modality][challenge.media_type]

        if challenge.modality == 'video':
            sample = challenge.sample_video_frames(cache)
        elif challenge.modality == 'image':
            sample = cache.sample()

        if sample is None:
            bt.logging.warning(f"Waiting for {challenge.media_type} cache to populate. Challenge skipped.")
            return None

        challenge.original_media = sample[challenge.modality]
        try:
            challenge.augmented_media, aug_level, aug_params = apply_augmentation_by_level(
                challenge.original_media, 
                challenge.config.target_image_size, 
                sample.get('mask_center', None))
        except Exception as e:
            bt.logging.error(f"Unable to apply augmentations: {e}\nChallenge generation failed.")
            return None

        sample.update({'aug_params': aug_params, 'aug_level': aug_level})
        if not challenge.process_metadata(sample):
            bt.logging.warning(f"Failed to process metadata. Challenge skipped.")
            return None

        return challenge

    def sample_video_frames(self, video_cache):
        """Sample frames from the video cache, either as a single clip or two combined clips."""
        min_frames = self.config.min_frames
        max_frames = self.config.max_frames
        min_fps = self.config.min_fps
        max_fps = self.config.max_fps

        if np.random.rand() > self.config.stitch_prob:
            num_frames = random.randint(min_frames, max_frames)
            sample = video_cache.sample(num_frames, min_fps=min_fps, max_fps=max_fps)
        else:
            num_frames_A = random.randint(min_frames, max_frames - 1)
            sample_A = video_cache.sample(num_frames_A, min_fps=min_fps, max_fps=max_fps)
            if sample_A is None:
                return None
            num_frames_B = random.randint(min_frames, max(max_frames - num_frames_A, min_frames + 1))
            sample_B = video_cache.sample(num_frames_B, fps=sample_A['fps'])
            sample = {k + '_A': v for k, v in sample_A.items()}
            sample.update({k + '_B': v for k, v in sample_B.items()})
            sample['video'] = sample_A['video'] + sample_B['video']

        return sample
      
    def process_metadata(self, sample) -> bool:
        """Prepare challenge metadata and media for logging to Weights & Biases """
        self.metadata = {
            'label': int(self.label),
            'media_type': str(self.media_type),
            'modality': str(self.modality)
        }
        self.metadata.update({
            k: v for k, v in sample.items() 
            if self.modality not in k
        })
        try:
            if self.modality == 'video':
                def create_wandb_video(video_frames, fps):
                    frames = [np.array(img) for img in video_frames]
                    frames_arr = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
                    return wandb.Video(frames_arr, format="mp4", fps=fps)

                if 'video_A' in sample:
                    self.metadata['video_A'] = create_wandb_video(sample['video_A'], sample['fps_A'])
                    self.metadata['video_B'] = create_wandb_video(sample['video_B'], sample['fps_B'])
                else:
                    self.metadata['video'] = create_wandb_video(self.original_media, self.metadata.get('fps', 30))

                self.metadata['augmented_video'] = create_wandb_video(
                    self.augmented_media, self.metadata.get('fps', 30))

            elif self.modality == 'image':
                self.metadata['image'] = wandb.Image(self.original_media)
                self.metadata['augmented_image'] = wandb.Image(self.augmented_media)

            return True

        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(f"{self.modality} is truncated or corrupt. Challenge skipped.")
            return False
