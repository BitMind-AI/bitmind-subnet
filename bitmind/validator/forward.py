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

from datetime import datetime
from io import BytesIO
from PIL import Image
import bittensor as bt
import pandas as pd
import numpy as np
import wandb
import time
import os

from bitmind.utils.uids import get_random_uids
from bitmind.utils.data import sample_dataset_index_name
from bitmind.protocol import prepare_synapse
from bitmind.validator.reward import get_rewards
from bitmind.image_transforms import random_aug_transforms, base_transforms


def sample_random_real_image(datasets, total_images, retries=10):
    random_idx = np.random.randint(0, total_images)
    source, idx = sample_real_image(datasets, random_idx)
    if source[idx]['image'] is None:
        if retries:
            return sample_random_real_image(datasets, total_images, retries-1)
        return None, None
    return source, idx


def sample_real_image(datasets, index):
    cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
    source_index = np.searchsorted(cumulative_sizes, index % (cumulative_sizes[-1]))
    source = datasets[source_index]
    valid_index = index - (cumulative_sizes[source_index - 1] if source_index > 0 else 0)
    return source, valid_index


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
    challenge_data = {}

    modality = 'video' if np.random.rand() > 0.0 else 'image'

    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    if np.random.rand() > 1.:#self._fake_prob:
        if modality == 'video':
            bt.logging.warning('TODO')
            return 

        elif modality == 'image':
            bt.logging.info('sampling real image')
            label = 0
            dataset, local_index = sample_random_real_image(self.real_image_datasets, self.total_real_images)
            sample = dataset[local_index]

            challenge_data[modality] = sample[modality]
            challenge_data['dataset'] = dataset.huggingface_dataset_name
            challenge_data['image_index'] = local_index

    else:
        label = 1

        if self.config.neuron.prompt_type == 'annotation':
            bt.logging.info('generating fake image from annotation of real image')

            retries = 10
            while retries > 0:
                retries -= 1

                # sample real data from which to generate prompt
                prompt_dataset, local_index = sample_random_real_image(self.real_image_datasets, self.total_real_images)
                prompt_sample = prompt_dataset[local_index]
                prompt_image = prompt_sample['image']
                if prompt_image is None:
                    bt.logging.warning(f"Missing image encountered at {prompt_sample['id']}, resampling...")
                    continue

                # generate captions for the real images, then synthetic images from these captions
                sample = self.synthetic_data_generator.generate(
                    k=1, real_images=[prompt_sample], modality=modality)[0]  # {'prompt': str, 'image': PIL Image ,'id': int}

                challenge_data['model'] = self.synthetic_data_generator.t2vis_model_name
                challenge_data['prompt_dataset'] = prompt_dataset.huggingface_dataset_name
                challenge_data['prompt_image_index'] = local_index
                challenge_data['prompt'] = sample['prompt']
                if modality == 'image':
                    gen_output = sample['gen_output'].images[0]
                    sample['image'] = gen_output
                    challenge_data['image'] = wandb.Image(gen_output)
                elif modality == 'video':
                    gen_output = sample['gen_output'].frames[0]
                    sample['video'] = gen_output
                    np_video = np.stack([np.array(img) for img in gen_output], axis=0)
                    challenge_data['video'] = wandb.Video(np_video)
    
                if not np.any(np.isnan(gen_output)):
                    break

                bt.logging.warning("NaN encountered in prompt/image generation, retrying...")

        else:
            bt.logging.error(f'unsupported neuron.prompt_type: {self.config.neuron.prompt_type}')
            raise NotImplementedError

    input_data = sample[modality]  # extract video or image
    if np.random.rand() > 0.25:
        input_data = random_aug_transforms(input_data)
        data_aug_params = random_aug_transforms.params
    else:
        input_data = base_transforms(input_data)
        data_aug_params = {}

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
    bt.logging.info(f'{"real" if label == 0 else "fake"} image | source: {source_name}: {sample["id"]}')
    
    # Logging responses and rewards
    bt.logging.info(f"Received responses: {responses}")
    bt.logging.info(f"Scored responses: {rewards}")
    
    # Update the scores based on the rewards.
    self.update_scores(rewards, miner_uids)

    # update logging data
    challenge_data['data_aug_params'] = data_aug_params
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
