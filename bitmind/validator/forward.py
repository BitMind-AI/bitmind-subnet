# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 Bitmind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Union, Optional
from datetime import datetime
from io import BytesIO
from PIL import Image
import bittensor as bt
import pandas as pd
import numpy as np
import random
import wandb
import time
import os
import cv2

from bitmind.validator.config import SYNTH_IMAGE_CACHE_DIR, SYNTH_VIDEO_CACHE_DIR
from bitmind.utils.uids import get_random_uids
from bitmind.utils.data import sample_dataset_index_name
from bitmind.protocol import prepare_synapse
from bitmind.validator.reward import get_rewards
from bitmind.utils.image_transforms import apply_augmentation_by_level


def video_to_pil(video_path: str | Path) -> list[Image.Image]:
   """
   Load video as a list of PIL images.
   
   Args:
       video_path: Path to video file
       
   Returns:
       List of PIL Image objects
   """
   clip = VideoFileClip(str(video_path))
   frames = [Image.fromarray(np.array(frame)) for frame in clip.iter_frames()]
   clip.close()
   return frames


def sample_random_files(
    directory: str | Path,
    extensions: list[str] | None = None
) -> list[Path]:
    """
    Sample n random files from a directory, optionally filtering by extension.

    Args:
        directory: Path to directory
        n: Number of files to sample
        extensions: List of extensions to filter by (e.g. ['.jpg', '.png'])

    Returns:
        List of random file paths (may be fewer than n if not enough files)
    """
    files = Path(directory).iterdir()
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    files = list(files)
    return random.sample(files, 1)[0] if files else None


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Steps are:
    1. Sample miner UIDs
    2. Get an image. 50/50 chance of:
        A. REAL (label = 0): Randomly sample a real image from self.real_image_datasets
        B. FAKE (label = 1): Generate a synthetic image with self.random_image_generator
    3. Apply random data augmentation to the image
    4. Base64 encode the image and prepare an ImageSynapse
    5. Query miner axons
    6. Log results, including image and miner responses (soon to be W&B)
    7. Compute rewards and update scores

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    while True:
        challenge_data = {}
        modality = 'video' if np.random.rand() > 0.0 else 'image'
        challenge_data['modality'] = modality

        miner_uids = get_random_uids(self, k=self.metagraph.n) # self.config.neuron.sample_size)
        if np.random.rand() > self._fake_prob:
            label = 0
            if modality == 'video':
                clip_length = random.randint(
                    self.config.neuron.clip_length_min, 
                    self.config.neuron.clip_length_max)
                sample = self.video_cache.sample(clip_length)
                challenge_data['clip_length_s'] = clip_length
                challenge_data.update({k: v for k, v in sample.items() if k not in ('video')})
                bt.logging.info(f"sampled {clip_length}s of video from {challenge_data['path']}")

            elif modality == 'image':
                sample = self.image_cache.sample()
                challenge_data[modality] = sample[modality]
                challenge_data['dataset'] = sample['dataset']
                challenge_data['image_index'] = sample['index']

        else:
            label = 1
            if modality == 'image':
                file = sample_random_files(
                    SYNTH_IMAGE_CACHE_DIR, extensions=['.png', '.jpg', '.jpeg'])
                image = Image.open(file)
                if file is None:
                    bt.logging.warning("No synthetic images available")
                    continue
                challenge_data['image'] = wandb.Image(image)

            elif modality == 'video':
                file = sample_random_files(
                    SYNTH_VIDEO_CACHE_DIR, extensions=['.mp4'])
                if file is None:
                    bt.logging.warning("No synthetic videos available")
                    continue
                video = video_to_pil(file)
                print(f'{len(video)} frames')

                #np_video = np.stack([np.array(img) for img in gen_output], axis=0)
                #challenge_data['video'] = wandb.Video(np_video) # TODO format video for w&b

            # TODO get prompt and other metadata for synth sample
        break

    input_data = sample[modality]  # extract video or image
    try:
        image, level, data_aug_params = apply_augmentation_by_level(input_data)
    except Exception as e:
        bt.logging.error(f"Unable to applay augmentations: {e}")

    bt.logging.info(f"Querying {len(miner_uids)} miners...")
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=prepare_synapse(input_data, modality=modality),
        deserialize=True,
        timeout=9
    )
    bt.logging.info(f"Responses received in {time.time() - start}s")

    rewards, metrics = get_rewards(
        label=label,
        responses=responses,
        uids=miner_uids,
        axons=axons,
        performance_tracker=self.performance_tracker)

    # Logging image source (model for synthetic, dataset for real) and verification details
    source_name = challenge_data['model'] if 'model' in challenge_data else challenge_data['dataset']
    bt.logging.info(f'{"real" if label == 0 else "fake"} image | source: {source_name}')

    # Logging responses and rewards
    bt.logging.info(f"Received responses: {responses}")
    bt.logging.info(f"Scored responses: {rewards}")

    # Update the scores based on the rewards.
    self.update_scores(rewards, miner_uids)

    # update logging data
    challenge_data['data_aug_params'] = data_aug_params
    challenge_data['data_aug_level'] = level
    challenge_data['label'] = label
    challenge_data['miner_uids'] = list(miner_uids)
    challenge_data['miner_hotkeys'] = list([axon.hotkey for axon in axons])
    challenge_data['predictions'] = responses
    challenge_data['correct'] = [
        np.round(y_hat) == y
        for y_hat, y in zip(responses, [label] * len(responses))
    ]
    challenge_data['rewards'] = list(rewards)
    challenge_data['scores'] = list(self.scores)

    metric_names = list(metrics[0].keys())
    for metric_name in metric_names:
        challenge_data[f'miner_{metric_name}'] = [m[metric_name] for m in metrics]

    # W&B logging if enabled
    if not self.config.wandb.off:
        wandb.log(challenge_data)

    # ensure state is saved after each challenge
    self.save_miner_history()

    # Track miners who have responded
    self.last_responding_miner_uids = []
    for i, pred in enumerate(responses):
        # Logging specific prediction details
        if pred != -1:
            bt.logging.info(f'Miner uid: {miner_uids[i]} | prediction: {pred} | correct: {np.round(pred) == label} | reward: {rewards[i]}')
            self.last_responding_miner_uids.append(miner_uids[i])
