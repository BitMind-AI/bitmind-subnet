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
from gas.utils.state_manager import load_validator_state, save_validator_state
from gas.utils.wandb_utils import init_wandb, clean_wandb_cache
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
SS58_ADDRESSES = {
    "burn": "5HjBSeeoz52CLfvDWDkzupqrYLHz1oToDPHjdmJjc4TF68LQ",
    "video_escrow": "5G6BJ1Z6LeDptRn5GTw74QSDmG1FP3eqVque5JhUb5zeEyQa",
    "image_escrow": "5EUJFyH4ZSSiD3C8sM698nsVE26Tq98LoBwkmopmWZqaZqCA",
}


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
            wandb_dir = self.config.neuron.full_path
            clean_wandb_cache(wandb_dir)
            self.wandb_run = init_wandb(
                self.config,
                "validator",
                self.uid,self.wallet.hotkey,
                wandb_dir
            )

        bt.logging.info(self.config)
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
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
            save_state_callback=self.save_state,
        )

        await self.load_state()

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
        """Issue generative challenges and save state to preserve active tasks"""
        await self.generative_challenge_manager.issue_generative_challenge()
        await self.save_state()

    @on_block_interval("epoch_length")
    async def set_weights(self, block):
        """
        Query orchestrator for results, computes rewards, updates scores, set weights
        """
        bt.logging.info(f"Updating scores at block {block}")
        generator_uids = await self.update_scores()

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
                burn_pct = 0.7
                burn_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=SS58_ADDRESSES["burn"],
                    netuid=self.config.netuid, 
                    block=block
                )

                d_pct = .8
                video_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=SS58_ADDRESSES["video_escrow"],
                     netuid=self.config.netuid, 
                     block=block)
                image_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=SS58_ADDRESSES["image_escrow"],
                     netuid=self.config.netuid, 
                     block=block)

                # .8 to discriminators, .2 to generators for now
                g_pct = (1. - d_pct)
                normed_weights = np.array([
                    (1 - burn_pct) * g_pct  * v
                    for v in normed_weights
                ])
                
                normed_weights[burn_uid] = burn_pct
                normed_weights[image_escrow_uid] = (1 - burn_pct) * d_pct / 2
                normed_weights[video_escrow_uid] = (1 - burn_pct) * d_pct / 2
                bt.logging.info(f"Image discriminator escrow UID: {image_escrow_uid}")
                bt.logging.info(f"Video discriminator escrow UID: {video_escrow_uid}")

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
        # Get verification stats for only unrewarded media to quickly slash for submissions that fail verification
        verification_stats = self.content_manager.get_unrewarded_verification_stats(include_all=True)
        generator_base_rewards, media_ids = get_generator_base_rewards(verification_stats)
        generator_results, discriminator_results = await get_benchmark_results(
            self.wallet.hotkey, self.metagraph, base_url=self.config.benchmark.api_url
        )
        #bt.logging.debug(f"discriminator_results: {json.dumps(discriminator_results, indent=2)}")
        #bt.logging.debug(f"generator_results: {json.dumps(generator_results, indent=2)}")

        reward_multipliers = get_generator_reward_multipliers(generator_results, self.metagraph)
        all_generator_uids = set(generator_base_rewards.keys()) | set(reward_multipliers.keys())
        rewards = {
            uid: generator_base_rewards.get(uid, 1e-4) * reward_multipliers.get(uid, .01)
            for uid in all_generator_uids 
        }

        if len(rewards) == 0:
            bt.logging.trace("No rewards available for score update")
            return

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

        return list(all_generator_uids)

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

    async def save_state(self):
        """
        Atomically save validator state (scores + challenge tasks)
        Maintains the current state and one backup.
        """
        async with self._state_lock:
            bt.logging.debug("save_state() acquired state lock")
            try:
                state_data = {"scores.npy": self.scores}
                state_objects = [
                    (self.generative_challenge_manager, "challenge_tasks.pkl")
                ]

                success = save_validator_state(
                    base_dir=self.config.neuron.full_path,
                    state_data=state_data,
                    state_objects=state_objects,
                    max_backup_age_hours=24,
                )
                if success:
                    bt.logging.success("Saved validator state")
                else:
                    bt.logging.error("Failed to save validator state")
            except Exception as e:
                bt.logging.error(f"Error during state save: {str(e)}")
                bt.logging.error(traceback.format_exc())
            finally:
                bt.logging.debug("save_state() releasing state lock")

    async def load_state(self):
        """
        Load validator state, falling back to backup if needed.
        """
        try:
            state_data_keys = ["scores.npy"]
            state_objects = [
                (self.generative_challenge_manager, "challenge_tasks.pkl")
            ]

            loaded_state = load_validator_state(
                base_dir=self.config.neuron.full_path,
                state_data_keys=state_data_keys,
                state_objects=state_objects,
                max_backup_age_hours=24,
            )

            if loaded_state is not None and "scores.npy" in loaded_state:
                self.scores = loaded_state["scores.npy"]
                bt.logging.info(f"Loaded scores vector for {len(self.scores)} miners")
                return True
            else:
                bt.logging.warning("No valid state found")
                return False

        except Exception as e:
            bt.logging.error(f"Error during state load: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False

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
