import asyncio
import sys
import json
import traceback
from time import sleep

import numpy as np
from dotenv import load_dotenv
import bittensor as bt
from threading import Thread

from gas import __spec_version__ as spec_version
from gas.protocol.validator_requests import get_benchmark_results
from gas.utils.autoupdater import autoupdate
from gas.cache import ContentManager
from gas.utils.metagraph import create_set_weights
from gas.types import (
    NeuronType,
    MinerType,
)

from gas.utils import (
    on_block_interval,
    print_info,
)
from gas.utils.wandb_utils import WandbLogger
from neurons.base import BaseNeuron
from gas.evaluation import (
    GenerativeChallengeManager,
    MinerTypeTracker,
    get_generator_base_rewards,
    get_generator_reward_multipliers,
)

try:
    load_dotenv(".env.validator")
except Exception:
    pass


MAINNET_UID = 34


class Validator(BaseNeuron):
    neuron_type = NeuronType.VALIDATOR

    def __init__(self, config=None):
        super().__init__(config=config)

        self.initialization_complete = False

        ## CHECK IF REGG'D
        if (
            not self.metagraph.validator_permit[self.uid]
            and self.config.netuid == MAINNET_UID
        ):
            bt.logging.error("Validator does not have vpermit")
            sys.exit(1)

        self.init()

    def init(self):
        self.heartbeat_thread: Thread = None
        self.step = 0

        self._state_lock = asyncio.Lock()

        self.content_manager = ContentManager(self.config.cache.base_dir)

        ## Typesafety
        self.set_weights_fn = create_set_weights(spec_version, self.config.netuid)
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        bt.logging.info(f"Initialized scores vector for {len(self.scores)} miners")

        if not self.config.wandb_off:
            self.wandb_logger = WandbLogger(self.config, self.uid, self.wallet.hotkey)

        bt.logging.info(self.config)
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.start_new_wanbd_run,
                self.issue_generator_challenge,
                self.set_weights,
            ]
        )

        self.reward_config = {
            "window": self.config.scoring.window,
            "image_score_weight": self.config.scoring.image_weight,
            "video_score_weight": self.config.scoring.video_weight,
            "binary_score_weight": self.config.scoring.binary_weight,
            "multiclass_score_weight": self.config.scoring.multiclass_weight,
        }

        # SETUP HEARTBEAT THREAD
        if self.config.neuron.heartbeat:
            self.heartbeat_thread = Thread(name="heartbeat", target=self.heartbeat)
            self.heartbeat_thread.start()


    async def run(self):

        bt.logging.info(
            f"Running validator {self.uid} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        self.miner_type_tracker = MinerTypeTracker(
            self.config, self.wallet, self.metagraph, self.subtensor
        )

        self.generative_challenge_manager = GenerativeChallengeManager(
            self.config,
            self.wallet,
            self.metagraph,
            self.subtensor,
            self.miner_type_tracker,
        )

        dataset_counts = self.content_manager.get_dataset_media_counts()
        total_dataset_media = sum(dataset_counts.values())
        if total_dataset_media == 0:
            bt.logging.warning("No dataset media found.")
        else:
            bt.logging.info(
                f"Found {total_dataset_media} dataset media entries: {dataset_counts}"
            )

        await self.start_substrate_subscription()

        self.initialization_complete = True
        bt.logging.success(
            "\N{GRINNING FACE WITH SMILING EYES}",
            f"Initialization Complete. Validator starting at block: {self.subtensor.block}",
        )

        await self.set_weights(0)

        while not self.exit_context.isExiting:
            self.step += 1
            if self.config.autoupdate and (self.step == 0 or not self.step % 300):
                bt.logging.debug("Checking autoupdate")
                autoupdate(branch="main", install_deps=True)

            self.check_substrate_connection()

            await asyncio.sleep(1)

        await self.shutdown()

    @on_block_interval("generator_challenge_interval")
    async def issue_generator_challenge(self, block):
        """Generator challenges coming soon!"""
        await self.generative_challenge_manager.issue_generative_challenge()

    @on_block_interval("epoch_length")
    async def set_weights(self, block):
        """
        Query orchestrator for results, computes rewards, updates scores, set weights
        """
        bt.logging.info(f"Updating scores at block {block}")
        generator_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)
        await self.update_scores()

        async with self._state_lock:
            bt.logging.debug("set_weights() acquired state lock")
            try:
                bt.logging.info(f"Setting weights at block {block}")
                self.subtensor = bt.subtensor(
                    config=self.config, network=self.config.subtensor.chain_endpoint
                )
                uids = list(range(self.metagraph.n))

                if np.isnan(self.scores).any():
                    bt.logging.warning(
                        "Scores contain NaN values. This may be due to a lack of "
                        "responses from miners, or a bug in your reward functions."
                    )

                norm = np.ones_like(self.scores)
                norm[generator_uids] = np.linalg.norm(self.scores[generator_uids], ord=1)

                if np.any(norm == 0) or np.isnan(norm).any():
                    norm = np.ones_like(norm)  # Avoid division by zero or NaN

                normed_weights = self.scores / norm

                # discriminator rewards distributed only upon performance improvements on benchmark exam
                discriminator_reward_hotkey = "5HjBSeeoz52CLfvDWDkzupqrYLHz1oToDPHjdmJjc4TF68LQ"
                discriminator_reward_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=discriminator_reward_hotkey, netuid=self.config.netuid, block=block)

                # .7 to discriminators, .3 to generators for now
                normed_weights[discriminator_reward_uid] = .7
                normed_weights = np.array([v * 0.3 for v in normed_weights])

                self.set_weights_fn(
                    self.wallet, self.metagraph, self.subtensor, (uids, normed_weights)
                )

            except Exception as e:
                bt.logging.error(f"Error in set_weights_on_interval: {e}")
                bt.logging.error(traceback.format_exc())
                return False

        return True

    async def update_scores(self):
        """
        Update self.scores with exponential moving average of rewards.
        """
        verification_stats = self.content_manager.get_unrewarded_verification_stats()
        generator_base_rewards, media_ids = get_generator_base_rewards(verification_stats)
        generator_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)

        generator_results, discriminator_results = await get_benchmark_results(self.metagraph)
        reward_multipliers = get_generator_reward_multipliers(generator_results, self.metagraph)
        rewards = {
            generator_base_rewards.get(uid, 0) * reward_multipliers.get(uid, 0)
            for uid in generator_uids 
        }

        if len(rewards) == 0:
            bt.logging.trace("No rewards available for score update")
            return

        bt.logging.info(f"Rewards:\n{json.dumps(rewards, indent=2)}")

        extend_scores = max(list(rewards.keys())) - len(self.scores) + 1
        if extend_scores > 0:
            self.scores = np.append(self.scores, np.zeros(extend_scores))

        reward_arr = np.array([rewards.get(i, 0) for i in range(len(self.scores))])

        alpha = 0.1
        self.scores = alpha * reward_arr + (1 - alpha) * self.scores

        bt.logging.info(
            f"Updated scores for {len(rewards)} miners with EMA (alpha={alpha})"
        )

        if media_ids:
            success = self.content_manager.mark_media_rewarded(media_ids)
            if success:
                bt.logging.info(f"Marked {len(media_ids)} media entries as rewarded")
            else:
                bt.logging.warning("Failed to mark media as rewarded")


    @on_block_interval("wandb_restart_interval")
    async def start_new_wanbd_run(self, block):
        try:
            self.wandb_logger.start_new_run()
        except Exception as e:
            bt.logging.error(f"Not able to start new W&B run: {e}")

    async def log_on_block(self, block):
        """
        Log information about validator state at regular intervals.
        """
        try:
            blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
            bt.logging.info(
                f"Forward Block: {self.subtensor.block} | Blocks till Set Weights: {blocks_till}"
            )
            print_info(
                self.metagraph,
                self.wallet.hotkey.ss58_address,
                block,
            )
        
        except Exception as e:
            bt.logging.warning(f"Error in log_on_block: {e}")


    async def shutdown(self):
        """Shutdown the validator and clean up resources."""
        await self.shutdown_substrate()
        if self.generative_challenge_manager:
           await self.generative_challenge_manager.shutdown()

    def heartbeat(self):
        bt.logging.info("Starting Heartbeat")
        last_step = self.step
        stuck_count = 0
        while True:
            sleep(self.config.neuron.heartbeat_interval_seconds)
            if last_step == self.step:
                stuck_count += 1
            if last_step != self.step:
                stuck_count = 0
            if stuck_count >= self.config.neuron.max_stuck_count:
                bt.logging.error(
                    "Heartbeat detecting main process hang, attempting restart"
                )
                autoupdate(force=True, install_deps=True)
                sys.exit(0)
            last_step = self.step
            bt.logging.info("Heartbeat")


if __name__ == "__main__":
    try:
        validator = Validator()
        asyncio.run(validator.run())
    except KeyboardInterrupt:
        bt.logging.info("Validator interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
