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

from PIL import Image
from io import BytesIO
from datetime import datetime
import bittensor as bt
import pandas as pd
import numpy as np
import os
import wandb

from bitmind.utils.uids import get_random_uids
from bitmind.utils.data import sample_dataset_index_name
from bitmind.protocol import prepare_image_synapse
from bitmind.validator.reward import get_rewards
from bitmind.image_transforms import random_aug_transforms


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
    wandb_data = {}

    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    if np.random.rand() > self._fake_prob:
        bt.logging.info('sampling real image')

        label = 0
        real_dataset_index, source_dataset = sample_dataset_index_name(self.real_image_datasets)
        real_dataset = self.real_image_datasets[real_dataset_index]
        samples, idx = real_dataset.sample(k=1)  # {'image': PIL Image ,'id': int}
        sample = samples[0]

        wandb_data['dataset'] = source_dataset
        wandb_data['image_index'] = idx[0]
        wandb_data['image_name'] = sample['id']

    else:
        label = 1

        if self.config.neuron.prompt_type == 'annotation':
            bt.logging.info('generating fake image from annotation of real image')

            retries = 10
            while retries > 0:
                retries -= 1

                # sample image(s) from real dataset for captioning
                real_dataset_index, source_dataset = sample_dataset_index_name(self.real_image_datasets)
                real_dataset = self.real_image_datasets[real_dataset_index]
                images_to_caption, image_indexes = real_dataset.sample(k=1)  # [{'image': PIL Image ,'id': int}, ...]

                # generate captions for the real images, then synthetic images from these captions
                sample = self.synthetic_image_generator.generate(
                    k=1, real_images=images_to_caption)[0]  # {'prompt': str, 'image': PIL Image ,'id': int}

                wandb_data['model'] = self.synthetic_image_generator.diffuser_name
                wandb_data['source_dataset'] = source_dataset
                wandb_data['source_image_name'] = images_to_caption[0]['id']
                wandb_data['source_image_index'] = image_indexes[0]
                wandb_data['image'] = wandb.Image(sample['image'])
                wandb_data['prompt'] = sample['prompt']
                if not np.any(np.isnan(sample['image'])):
                    break

                bt.logging.warning("NaN encountered in prompt/image generation, retrying...")

        elif self.config.neuron.prompt_type == 'random':
            bt.logging.info('generating fake image using prompt_generator')
            sample = self.synthetic_image_generator.generate(k=1)[0]

            wandb_data['model'] = self.synthetic_image_generator.diffuser_name
            wandb_data['image'] = wandb.Image(sample['image'])
            wandb_data['prompt'] = sample['prompt']

        else:
            bt.logging.error(f'unsupported neuron.prompt_type: {self.config.neuron.prompt_type}')
            raise NotImplementedError

    image = random_aug_transforms(sample['image'])
    data_aug_params = random_aug_transforms.params

    bt.logging.info(f"Querying {len(miner_uids)} miners...")
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    responses = await self.dendrite(
        axons=axons,
        synapse=prepare_image_synapse(image=image),
        deserialize=True,
        timeout=9
    )

    # update logging data
    wandb_data['data_aug_params'] = data_aug_params
    wandb_data['label'] = label
    wandb_data['miner_uids'] = list(miner_uids)
    wandb_data['miner_hotkeys'] = list([axon.hotkey for axon in axons])
    wandb_data['predictions'] = responses
    wandb_data['correct'] = [
        np.round(y_hat) == y
        for y_hat, y in zip(responses, [label] * len(responses))
    ]

    rewards, metrics = get_rewards(
        label=label,
        responses=responses,
        uids=miner_uids,
        axons=axons,
        performance_tracker=self.performance_tracker)
    
    # Logging image source and verification details
    source_name = wandb_data['model'] if 'model' in wandb_data else wandb_data['dataset']
    bt.logging.info(f'{"real" if label == 0 else "fake"} image | source: {source_name}: {sample["id"]}')
    
    # Logging responses and rewards
    bt.logging.info(f"Received responses: {responses}")
    bt.logging.info(f"Scored responses: {rewards}")
    
    # Update the scores based on the rewards.
    self.update_scores(rewards, miner_uids)

    metric_names = list(metrics[0].keys())
    for metric_name in metric_names:
        wandb_data[f'miner_{metric_name}'] = [m[metric_name] for m in metrics]

    # W&B logging if enabled
    if not self.config.wandb.off:
        wandb.log(wandb_data)

    # Track miners who have responded
    self.last_responding_miner_uids = []
    for i, pred in enumerate(responses):
        # Logging specific prediction details
        if pred != -1:
            bt.logging.info(f'Miner uid: {miner_uids[i]} | prediction: {pred} | correct: {np.round(pred) == label} | reward: {rewards[i]}')
            self.last_responding_miner_uids.append(miner_uids[i])
