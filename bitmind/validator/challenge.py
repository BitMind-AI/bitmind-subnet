import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

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
    MODALITIES,
    MODALITY_PROBS,
)


@dataclass
class Challenge:
    """Container for challenge data, metadata, and configuration."""

    # Challenge state
    label: int = -1       # 0='real', 1='synthetic', 2='semisynthetic'
    media_type: str = ""  # 'real', 'synthetic', 'semisynthetic'
    modality: str = ""    # 'image', 'video'
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Config
    target_image_size: Tuple[int] = TARGET_IMAGE_SIZE
    modality_options: Tuple[int] = MODALITIES
    modality_probs: List[float] = MODALITY_PROBS
    label_options: Tuple[int] = LABELS
    label_probs: Tuple[float] = LABEL_PROBS
    label_to_type: Dict[int, str] = field(default_factory=lambda: LABEL_TO_TYPE)
    min_frames: int = MIN_FRAMES
    max_frames: int = MAX_FRAMES
    stitch_prob: float = P_STITCH

    @classmethod
    def create(cls, media_cache):
        """Factory method to create and initialize a challenge."""
        challenge = cls()

        # randomly initialize challenge parameters
        challenge.modality = np.random.choice(cls.modality_options, p=cls.modality_probs)
        challenge.label = np.random.choice(cls.label_options, p=cls.label_probs)
        challenge.media_type = challenge.label_to_type[challenge.label]

        # initialize metadata
        challenge.metadata = {
            'label': challenge.label,
            'media_type': challenge.media_type,
            'modality': challenge.modality
        }

        # sample data from cache
        bt.logging.info(f"Sampling data from {challenge.media_type} {challenge.modality} cache")
        cache = media_cache[challenge.modality][challenge.media_type]
        if challenge.modality == 'video':
            challenge.data = challenge.sample_video_frames(
                cache, challenge.min_frames, challenge.max_frames)
        elif challenge.modality == 'image':
            challenge.data = cache.sample()

        if challenge.data is None:
            bt.logging.warning(f"Challenge skipped -- waiting for {challenge.media_type} cache to populate")
            return None

        # apply augmentation
        original_media = challenge.data.get(challenge.modality, None)
        try:
            augmented_data, aug_level, aug_params = apply_augmentation_by_level(
                original_media, 
                cls.target_image_size, 
                challenge.data.get('mask_center', None))

            challenge.data[f'{challenge.modality}_augmented'] = augmented_data
            challenge.metadata['data_aug_params'] = aug_params
            challenge.metadata['data_aug_level'] = aug_level

        except Exception as e:
            bt.logging.error(f"Unable to apply augmentations: {e}")

        return challenge

    @staticmethod
    def sample_video_frames(video_cache, min_frames, max_frames, min_fps=8, max_fps=30):
        """Sample frames from the video cache, either as a single clip or two combined clips."""
        if np.random.rand() > P_STITCH:
            num_frames = random.randint(min_frames, max_frames)
            challenge = video_cache.sample(num_frames, min_fps=min_fps, max_fps=max_fps)
        else:
            num_frames_A = random.randint(min_frames, max_frames - 1)
            sample_A = video_cache.sample(num_frames_A, min_fps=min_fps, max_fps=max_fps)
            if sample_A is None:
                return None
            num_frames_B = random.randint(min_frames, max(max_frames - num_frames_A, min_frames + 1))
            sample_B = video_cache.sample(num_frames_B, fps=sample_A['fps'])
            challenge = {
                'videos': [sample_A['video'], sample_B['video']],  # for wandb logging to handle different shapes
                'video': sample_A['video'] + sample_B['video'],
                'num_frames': sample_A['num_frames'] + sample_B['num_frames'],
                'fps': sample_A['fps']
            }
        return challenge
                
    def get_media(self):
        """Extract the input data (image or video)"""
        return self.data.get(self.modality, None)
    
    def get_augmented_media(self):
        """Extract the augmented input data (image or video)"""
        media = self.data.get(f'{self.modality}_augmented', None)
        if media is None:
            bt.logging.warning(f"No augmented media found. Returning original media.")
            return self.get_media()
        return media
      
    def process_metadata(self) -> bool:
        """Prepare challenge metadata and media for logging.
        Note that for challenges with two videos stitched together, we log the original videos separately
        as video_0 and video_1, as they are not necessarily the same dimensions and cannot be stacked until
        after transformations are applied. 
        """
        try:
            if self.modality == 'video':
                self.metadata['fps'] = self.data['fps']
                self.metadata['num_frames'] = self.data['num_frames']

                # log individual videos before augmentation if multiple videos were combined
                if 'videos' in self.data:
                    for i, video in enumerate(self.data['videos']):
                        video_frames = [np.array(img) for img in video]
                        self.metadata[f'video_{i}'] = wandb.Video(
                            np.stack(video_frames, axis=0).transpose(0, 3, 1, 2), 
                            format="mp4", 
                            fps=self.data.fps) 
                else:
                    video_frames = [np.array(img) for img in self.data['video']]
                    self.metadata['video'] = wandb.Video(
                        np.stack(video_frames, axis=0).transpose(0, 3, 1, 2), 
                        format="mp4", 
                        fps=self.data.fps) 
                    
                video_frames = [np.array(img) for img in self.data['video_augmented']]
                self.metadata['video_augmented'] = wandb.Video(
                    np.stack(video_frames, axis=0).transpose(0, 3, 1, 2), 
                    format="mp4", 
                    fps=self.data.fps) 

            elif self.modality == 'image':
                self.metadata['image'] = wandb.Image(self.data['image'])
                self.metadata['image_augmented'] = wandb.Image(self.data['image_augmented'])
 
            # Update metadata with everything except image/video data
            self.metadata.update({
                k: v for k, v in self.data.items() 
                if 'image' not in k and 'video' not in k
            })
            return True

        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(f"{self.modality} is truncated or corrupt. Challenge skipped.")
            return False
