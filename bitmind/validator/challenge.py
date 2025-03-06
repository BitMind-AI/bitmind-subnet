import random
import re
from typing import Dict, List, Any
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
    CHALLENGE_TYPE, 
    TARGET_IMAGE_SIZE, 
    CHALLENGE_DISTRIBUTION,
    MIN_FRAMES,
    MAX_FRAMES,
    P_STITCH
)


@dataclass
class Challenge:
    """Container for challenge data, metadata, and configuration."""

    # Challenge state
    media_type: str = ""  # 'real', 'synthetic', 'semisynthetic'
    modality: str = ""    # 'image', 'video'
    label: int = -1       # 0='real', 1='synthetic', 2='semisynthetic'
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Class variables
    target_image_size: tuple = TARGET_IMAGE_SIZE
    challenge_distribution: List[float] = CHALLENGE_DISTRIBUTION
    challenge_types: Dict[int, str] = CHALLENGE_TYPE
    min_frames: int = MIN_FRAMES
    max_frames: int = MAX_FRAMES
    stitch_prob: float = P_STITCH

    @classmethod
    def create(cls, media_cache):
        """Factory method to create and initialize a challenge."""
        challenge = cls()

        # randomly initialize challenge parameters
        challenge.label = np.random.choice([0, 1, 2], p=challenge.challenge_distribution)
        challenge.media_type = challenge.challenge_types[challenge.label]
        challenge.modality = 'video' if np.random.rand() > 0.5 else 'image'

        # initialize metadata
        challenge.metadata = {
            'label': challenge.label,
            'media_type': challenge.media_type,
            'modality': challenge.modality
        }

        # sample data from cache
        bt.logging.info(f"Sampling data from {challenge.modality} cache")
        cache = media_cache[challenge.media_type][challenge.modality]
        if challenge.modality == 'video':
            challenge.data = challenge.sample_video_frames(
                cache, challenge.clip_frames_min, challenge.clip_frames_max)
        elif challenge.modality == 'image':
            challenge.data = cache.sample()

        # apply augmentation
        original_media = challenge.get_media()
        try:
            augmented_data, aug_level, aug_params = apply_augmentation_by_level(
                original_media, 
                cls.target_image_size, 
                cls.data.get('mask_center', None))
        except Exception as e:
            augmented_data = original_media
            aug_level = -1
            aug_params = {}
            bt.logging.error(f"Unable to apply augmentations: {e}")

        # store augmented data and augmentation metadata
        challenge.data[f'{challenge.modality}_augmented'] = augmented_data
        challenge.metadata['data_aug_params'] = aug_params
        challenge.metadata['data_aug_level'] = aug_level
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
        """Extract the input data (image or video) from the challenge data."""
        return self.data[self.modality]
    
    def get_augmented_media(self):
        """Extract the input data (image or video) from the challenge data."""
        return self.data[f'{self.modality}_augmented']
      
    def process_metadata(self) -> bool:
        """Process and enrich metadata for logging."""
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
                if re.match(r'^(?!image$|video$|videos$|video_\d+$).+', k)
            })
            return True

        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(f"{self.modality} is truncated or corrupt. Challenge skipped.")
            return False
