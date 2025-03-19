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

import time
import wandb
import bittensor as bt

from bitmind.protocol import prepare_synapse
from bitmind.utils.uids import get_random_uids
from bitmind.validator.reward import get_rewards
from bitmind.validator.config import MAINNET_UID
from bitmind.validator.challenge import Challenge


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    
    This implementation uses a Challenge class to encapsulate challenge data and configuration,
    with execution logic directly in the forward function.
    """

    # create challenge
    challenge = Challenge.create(self.media_cache)
    if challenge is None:
        return

    # sample miners
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    challenge.metadata['miner_uids'] = list(miner_uids)
    challenge.metadata['miner_hotkeys'] = list([axon.hotkey for axon in axons])
    
    # prepare synapse
    synapse = prepare_synapse(
        challenge.augmented_media,
        modality=challenge.modality)

    # on testnet, add label for eyeballing correctness
    if self.metagraph.netuid != MAINNET_UID:
        synapse.testnet_label = challenge.label

    # query miners
    bt.logging.info(f"Sending {challenge.modality} challenge to {len(miner_uids)} miners")
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=True,
        run_async=True,
        timeout=9
    )
    bt.logging.info(f"Responses received in {time.time() - start}s")
    bt.logging.success(f"{challenge.media_type} {challenge.modality} challenge complete!")
    bt.logging.info({
        k: v for k, v in challenge.metadata.items() 
        if k not in ('miner_uids', 'miner_hotkeys')
    })

    # compute miner rewards and update score vector
    bt.logging.info(f"Scoring responses")
    rewards, metrics = get_rewards(
        label=challenge.label,
        responses=responses,
        uids=miner_uids,
        axons=axons,
        challenge_modality=challenge.modality,
        performance_trackers=self.performance_trackers)

    self.update_scores(rewards, miner_uids)

    # log results, track responding miners for serving organics
    responding_miner_uids = []
    unresponsive_miner_uids = []
    for uid, pred, reward, perf in zip(miner_uids, responses, rewards, metrics):
        if -1 in pred:
            unresponsive_miner_uids.append(uid)
            continue
        metric_str = ' | '.join([f"{modality} {m}: {perf[modality][m]:.4f}" for modality in perf for m in perf[modality]])
        bt.logging.success(f"UID: {uid} | {pred} | Reward: {reward:.4f} | " + metric_str)
        responding_miner_uids.append(uid)

    if len(unresponsive_miner_uids) > 0:
        bt.logging.warning(f"Failed to get responses from {len(unresponsive_miner_uids)} miners:")
        for uid in unresponsive_miner_uids:
            bt.logging.warning(f'UID {uid} ({self.metagraph.axons[uid]})')

    if responding_miner_uids:
        self.last_responding_miner_uids = responding_miner_uids

    # add predictions, rewards, scores and metrics to logging data
    challenge.metadata['predictions'] = responses
    challenge.metadata['rewards'] = rewards
    challenge.metadata['scores'] = list(self.scores)
    for modality in ['image', 'video']:
        if metrics and modality in metrics[0]:
            for metric_name in list(metrics[0][modality].keys()):
                challenge.metadata[f'miner_{modality}_{metric_name}'] = [
                    m[modality][metric_name] for m in metrics
                ]

    if not self.config.wandb.off:
        wandb.log(challenge.metadata)

    self.save_miner_history()
    self.media_cache[challenge.modality][challenge.media_type].prune_cache('extracted')
