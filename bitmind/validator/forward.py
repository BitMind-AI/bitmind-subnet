# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 BitMind

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

import random
import time
import re

import numpy as np
import pandas as pd
import wandb
import bittensor as bt

from bitmind.protocol import prepare_synapse
from bitmind.utils.image_transforms import apply_augmentation_by_level
from bitmind.utils.uids import get_random_uids
from bitmind.validator.config import CHALLENGE_TYPE, MAINNET_UID, TARGET_IMAGE_SIZE
from bitmind.validator.reward import get_rewards


def determine_challenge_type(media_cache, fake_prob=0.5):
    modality = 'video' if np.random.rand() > 0.5 else 'image'
    label = 0 if np.random.rand() > fake_prob else 1
    cache = media_cache[CHALLENGE_TYPE[label]][modality]
    task = None
    if label == 1:
        if modality == 'video':
            task = 't2v'
        elif modality == 'image':
            # 20% chance to use i2i (in-painting)
            task = 'i2i' if np.random.rand() < 0.2 else 't2i'
            label = 2
        cache = cache[task]
    return label, modality, task, cache


def sample_video_frames(video_cache, min_frames, max_frames, min_fps=8, max_fps=30):
    if np.random.rand() > 0.2:
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


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.

    Steps are:
    1. Sample miner UIDs
    2. Sample synthetic/real image/video (50/50 chance for each choice)
    3. Apply random data augmentation to the image
    4. Encode data and prepare Synapse
    5. Query miner axons
    6. Compute rewards and update scores

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    challenge_metadata = {}  # for bookkeeping
    challenge = {}           # for querying miners

    label, modality, source_model_task, cache = determine_challenge_type(self.media_cache)
    challenge_metadata['label'] = label
    challenge_metadata['modality'] = modality
    challenge_metadata['source_model_task'] = source_model_task

    bt.logging.info(f"Sampling data from {modality} cache")

    if modality == 'video':
        challenge = sample_video_frames(
            cache, self.config.neuron.clip_frames_min, self.config.neuron.clip_frames_max)
    elif modality == 'image':
        challenge = cache.sample()

    if challenge is None:
        bt.logging.warning("Waiting for cache to populate. Challenge skipped.")
        return

    # prepare metadata for logging
    try:
        if modality == 'video':
            if 'videos' in challenge:
                for i, video in enumerate(challenge['videos']):
                    video_arr = np.stack([np.array(img) for img in video], axis=0)
                    video_arr = video_arr.transpose(0, 3, 1, 2)
                    challenge_metadata[f'video_{i}'] = wandb.Video(video_arr, fps=1) 
            else:
                video_arr = np.stack([np.array(img) for img in challenge['video']], axis=0)
                video_arr = video_arr.transpose(0, 3, 1, 2)
                challenge_metadata['video'] = wandb.Video(video_arr, fps=1)
            challenge_metadata['fps'] = challenge['fps']
            challenge_metadata['num_frames'] = challenge['num_frames']
        elif modality == 'image':
            challenge_metadata['image'] = wandb.Image(challenge['image'])
    except Exception as e:
        bt.logging.error(e)
        bt.logging.error(f"{modality} is truncated or corrupt. Challenge skipped.")
        return

    # update logging dict with everything except image/video data
    challenge_metadata.update({
        k: v for k, v in challenge.items() 
        if re.match(r'^(?!image$|video$|videos$|video_\d+$).+', k)
    })
    input_data = challenge[modality]  # extract video or image

    # apply data augmentation pipeline
    try:
        input_data, level, data_aug_params = apply_augmentation_by_level(
            input_data, TARGET_IMAGE_SIZE, challenge.get('mask_center', None))
    except Exception as e:
        level, data_aug_params = -1, {}
        bt.logging.error(f"Unable to apply augmentations: {e}")

    challenge_metadata['data_aug_params'] = data_aug_params
    challenge_metadata['data_aug_level'] = level

    # sample miner uids for challenge
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    challenge_metadata['miner_uids'] = list(miner_uids)
    challenge_metadata['miner_hotkeys'] = list([axon.hotkey for axon in axons])

    # prepare synapse
    synapse = prepare_synapse(input_data, modality=modality)
    if self.metagraph.netuid != MAINNET_UID:
        synapse.testnet_label = label

    bt.logging.info(f"Sending {modality} challenge to {len(miner_uids)} miners")
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=True,
        timeout=9
    )
    bt.logging.info(f"Responses received in {time.time() - start}s")
    bt.logging.success(f"{CHALLENGE_TYPE[label]} {modality} challenge complete!")
    bt.logging.info({k: v for k, v in challenge_metadata.items() if k not in ('miner_uids', 'miner_hotkeys')})

    bt.logging.info(f"Scoring responses")
    rewards, metrics = get_rewards(
        label=label,
        responses=responses,
        uids=miner_uids,
        axons=axons,
        challenge_modality=modality,
        performance_trackers=self.performance_trackers)

    self.update_scores(rewards, miner_uids)

    responding_miner_uids = []
    for uid, pred, reward in zip(miner_uids, responses, rewards):
        if -1 not in pred:
            bt.logging.success(f"UID: {uid} | Prediction: {pred} | Reward: {reward}")
            responding_miner_uids.append(uid)

    if len(responding_miner_uids) > 0:
        self.last_responding_miner_uids = responding_miner_uids 
    
    for modality in ['image', 'video']:
        for metric_name in list(metrics[0][modality].keys()):
            challenge_metadata[f'miner_{modality}_{metric_name}'] = [m[modality][metric_name] for m in metrics]

    challenge_metadata['predictions'] = responses
    challenge_metadata['rewards'] = rewards
    challenge_metadata['scores'] = list(self.scores)

    # W&B logging if enabled
    if not self.config.wandb.off:
        wandb.log(challenge_metadata)

    # ensure state is saved after each challenge
    self.save_miner_history()
    if label == 1:
        cache._prune_extracted_cache()
