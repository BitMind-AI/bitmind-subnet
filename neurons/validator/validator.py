import asyncio
import sys
import json
import traceback
from time import sleep

import numpy as np
from dotenv import load_dotenv
import aiohttp
import bittensor as bt
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from substrateinterface import SubstrateInterface
from threading import Thread

from gas import __spec_version__ as spec_version
from gas.protocol import query_orchestrator, media_to_bytes
from gas.utils.autoupdater import autoupdate
from gas.cache import ContentManager
from gas.utils.metagraph import (
    create_set_weights,
    run_block_callback_thread,
)
from gas.types import NeuronType, MediaType, MinerType, Modality, DiscriminatorType
from gas.utils import (
    on_block_interval,
    print_info,
    save_validator_state,
    load_validator_state,
    apply_random_augmentations,
)
from gas.utils.wandb_utils import WandbLogger
from neurons.base import BaseNeuron
from gas.evaluation import (
    GenerativeChallengeManager,
    MinerTypeTracker,
    DiscriminatorTracker,
    get_discriminator_rewards,
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
        self.lock_waiting = False
        self.lock_halt = False
        self.step = 0

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
                # self.issue_generator_challenge,
                self.issue_discriminator_challenge,
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

        #self.generative_challenge_manager = GenerativeChallengeManager(
        #    self.config,
        #    self.wallet,
        #    self.metagraph,
        #    self.subtensor,
        #    self.miner_type_tracker,
        #)

        self.discriminator_tracker = DiscriminatorTracker(store_last_n=200)

        # self.generator_tracker = GeneratorTracker()

        await self.load_state()

        dataset_counts = self.content_manager.get_dataset_media_counts()
        total_dataset_media = sum(dataset_counts.values())
        if total_dataset_media == 0:
            bt.logging.warning("No dataset media found.")
        else:
            bt.logging.info(
                f"Found {total_dataset_media} dataset media entries: {dataset_counts}"
            )

        self.initialization_complete = True
        bt.logging.success(
            "\N{GRINNING FACE WITH SMILING EYES}",
            f"Initialization Complete. Validator starting at block: {self.subtensor.block}",
        )

        while not self.exit_context.isExiting:
            self.step += 1
            if self.config.autoupdate and (self.step == 0 or not self.step % 30):
                bt.logging.debug("Checking autoupdate")
                autoupdate(branch="main")

            # Make sure our substrate thread is alive
            if not self.substrate_thread.is_alive():
                bt.logging.info("Restarting substrate interface due to killed node")
                self.substrate = SubstrateInterface(
                    ss58_format=SS58_FORMAT,
                    use_remote_preset=True,
                    url=self.config.subtensor.chain_endpoint,
                    type_registry=TYPE_REGISTRY,
                )
                self.substrate_thread = run_block_callback_thread(
                    self.substrate, self.run_callbacks
                )

            if self.lock_halt:
                self.lock_waiting = True
                while self.lock_halt:
                    bt.logging.info("Waiting for lock to release")
                    sleep(self.config.neuron.lock_sleep_seconds)
                self.lock_waiting = False
            await asyncio.sleep(1)

        await self.shutdown()

    @on_block_interval("generator_challenge_interval")
    async def issue_generator_challenge(self, block):
        """Generator challenges coming soon!"""
        # await self.generative_challenge_manager.issue_generative_challenge()
        return

    @on_block_interval("discriminator_challenge_interval")
    async def issue_discriminator_challenge(self, block, retries=3):

        bt.logging.info("\033[96m🔍 Starting Discriminator Challenge\033[0m")

        # get miners
        await self.miner_type_tracker.update_miner_types()
        miner_uids = self.miner_type_tracker.get_miners_by_type(MinerType.DISCRIMINATOR)
        if len(miner_uids) > self.config.neuron.sample_size:
            miner_uids = np.random.choice(
                miner_uids,
                size=self.config.neuron.sample_size,
                replace=False,
            ).tolist()

        if not miner_uids:
            bt.logging.trace("No dscriminative miners found to challenge.")
            return
        # sample media
        for attempt in range(retries):
            modality, media_type, _ = self.determine_challenge_type()
            bt.logging.debug(
                f"Sampling attempt {attempt + 1}/{retries}: {modality}/{media_type}"
            )
            cache_result = self.content_manager.sample_media_with_content(
                modality, media_type
            )
            if cache_result is not None and cache_result.get("count", 0):
                break

        if cache_result is None or not cache_result.get("count", 0):
            bt.logging.warning(
                f"Failed to sample data after {retries} attempts. Discriminator challenge skipped."
            )
            return

        # extract and augment media
        media_sample = cache_result["items"][0]
        bt.logging.info(json.dumps(media_sample.get("metadata"), indent=2))

        media = media_sample[modality.value]
        augmented_media, _, _, aug_params = apply_random_augmentations(media)

        bt.logging.success(f"Sampled {media_type} {modality} from cache")
        bt.logging.info(
            f"Querying orchestrator with discriminator challenge for {miner_uids}"
        )

        # query orchestrator
        async with aiohttp.ClientSession() as session:
            results = await query_orchestrator(
                session,
                self.wallet.hotkey,
                miner_uids,
                modality,
                media_to_bytes(augmented_media)[0],
                total_timeout=self.config.neuron.miner_total_timeout,
                connect_timeout=self.config.neuron.miner_connect_timeout,
                sock_connect_timeout=self.config.neuron.miner_sock_connect_timeout,
            )

        # process responses
        if isinstance(results, dict) and results.get("status") != 200:
            bt.logging.error(
                f"Orchestrator request failed: {results.get('error', 'Unknown error')}"
            )
            return  # no penalty is applied to any miner if orchestrator call fails altogether

        bt.logging.info(f"Received {len(results)} results from orchestrator")
        predictions = [
            r["result"]["probabilities"] if r.get("result") is not None else None
            for r in results
        ]

        # compute rewards, update scores, save state, & log
        generator_rewards = {}  # placeholder for GAS Phase II
        discriminator_reward_outputs = get_discriminator_rewards(
            label=media_type.int_value,
            predictions=predictions,
            uids=[r.get("uid") for r in results],
            hotkeys=[r.get("hotkey") for r in results],
            challenge_modality=modality,
            discriminator_tracker=self.discriminator_tracker,
            **self.reward_config,
        )
        discriminator_rewards = discriminator_reward_outputs["rewards"]
        discriminator_metrics = discriminator_reward_outputs["metrics"]
        discriminator_correct = discriminator_reward_outputs["correct"]

        self.update_scores(generator_rewards, discriminator_rewards)
        await self.save_state()

        # Log responses and metrics
        for r in results:
            log_fn = (
                bt.logging.success if r.get("status") == 200 else bt.logging.warning
            )

            # update result with challenge media_type, metrics, rewards, scores for logging
            uid = r.get("uid")
            r["result"] = {} if r.get("result") is None else r["result"]
            r["result"]["media_type"] = media_type.value
            r["reward"] = discriminator_rewards.get(uid, {})
            r["correct"] = int(discriminator_correct.get(uid, False))
            r["score"] = self.scores[uid]
            r.update({
                f"{k}_metrics": v 
                for k, v in discriminator_metrics.get(uid, {}).items()
            })

            log_fn(json.dumps(r, indent=2))

        self.wandb_logger.log(results, media_sample, aug_params)

    @on_block_interval("epoch_length")
    async def set_weights(self, block):
        """
        Query orchestrator for results, computes rewards, updates scores, set weights
        """
        generator_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)
        discriminator_uids = self.miner_type_tracker.get_miners_by_type(
            MinerType.DISCRIMINATOR
        )
        if not len(generator_uids + discriminator_uids):
            bt.logging.warning(f"No miners currently being tracked")
            return

        extend_scores = max(discriminator_uids + generator_uids) - len(self.scores) + 1
        if extend_scores > 0:
            self.scores = np.append(self.scores, np.zeros(extend_scores))

        try:
            bt.logging.info(
                f"Waiting to safely set weights at block {block} (epoch length = {self.config.epoch_length})"
            )
            self.lock_halt = True
            while not self.lock_waiting and block != 0:
                sleep(self.config.neuron.lock_sleep_seconds)

            bt.logging.info(f"Setting weights at block {block}")
            self.subtensor = bt.subtensor(
                config=self.config, network=self.config.subtensor.chain_endpoint
            )
            uids = list(range(self.metagraph.n))

            if np.isnan(self.scores).any():
                bt.logging.warning(
                    f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
                )

            norm = np.ones_like(self.scores)
            norm[generator_uids] = np.linalg.norm(self.scores[generator_uids], ord=1)
            norm[discriminator_uids] = np.linalg.norm(
                self.scores[discriminator_uids], ord=1
            )

            if np.any(norm == 0) or np.isnan(norm).any():
                norm = np.ones_like(norm)  # Avoid division by zero or NaN

            normed_weights = self.scores / norm

            self.set_weights_fn(
                self.wallet, self.metagraph, self.subtensor, (uids, normed_weights)
            )
            bt.logging.success("Weights set successfully")

        except Exception as e:
            bt.logging.error(f"Error in set_weights_on_interval: {e}")
            bt.logging.error(traceback.format_exc())
            return False
        finally:
            self.lock_halt = False

        return True

    def update_scores(self, generator_rewards: dict, discriminator_rewards: dict):
        """
        Update self.scores with exponential moving average of rewards.

        Args:
            generator_rewards: Dict mapping generator UID to reward score
            discriminator_rewards: Dict mapping discriminator UID to reward score
        """
        rewards = {}
        rewards.update(generator_rewards)
        rewards.update(discriminator_rewards)

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

    def determine_challenge_type(self):
        """
        Randomly selects a modality (image, video) and media type (real, synthetic, semisynthetic)
        based on configured probabiltiies
        """
        modalities = [Modality.IMAGE.value, Modality.VIDEO.value]
        modality = np.random.choice(
            modalities,
            p=[
                self.config.challenge.image_prob,
                self.config.challenge.video_prob,
            ],
        )

        media_types = [
            MediaType.REAL.value,
            MediaType.SYNTHETIC.value,
            MediaType.SEMISYNTHETIC.value,
        ]
        media_type = np.random.choice(
            media_types,
            p=[
                self.config.challenge.real_prob,
                self.config.challenge.synthetic_prob,
                self.config.challenge.semisynthetic_prob,
            ],
        )

        multi_video = (
            modality == Modality.VIDEO
            and np.random.rand() < self.config.challenge.multi_video_prob
        )

        return Modality(modality), MediaType(media_type), multi_video

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

    async def save_state(self):
        """
        Atomically save validator state (scores + miner history)
        Maintains the current state and one backup.
        """
        self.lock_halt = True
        while not self.lock_waiting:
            sleep(self.config.neuron.lock_sleep_seconds)

        try:
            state_data = {"scores.npy": self.scores}
            state_objects = [(self.discriminator_tracker, "discriminator_history.pkl")]

            success = save_validator_state(
                base_dir=self.config.neuron.full_path,
                state_data=state_data,
                state_objects=state_objects,
                max_backup_age_hours=self.config.neuron.max_state_backup_hours,
            )
            if success:
                bt.logging.success("Saved validator state")
            else:
                bt.logging.error("Failed to save validator state")
        except Exception as e:
            bt.logging.error(f"Error during state save: {str(e)}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.lock_halt = False

    async def load_state(self):
        """
        Load validator state, falling back to backup if needed.
        """
        try:
            state_data_keys = ["scores.npy"]
            state_objects = [(self.discriminator_tracker, "discriminator_history.pkl")]

            loaded_state = load_validator_state(
                base_dir=self.config.neuron.full_path,
                state_data_keys=state_data_keys,
                state_objects=state_objects,
                max_backup_age_hours=self.config.neuron.max_state_backup_hours,
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
        pass
        #if self.generative_challenge_manager:
        #   await  self.generative_challenge_manager.shutdown()

    def heartbeat(self):
        bt.logging.info("Starting Heartbeat")
        last_step = self.step
        stuck_count = 0
        while True:
            while self.lock_halt:
                sleep(self.config.neuron.lock_sleep_seconds)
            sleep(self.config.neuron.heartbeat_interval_seconds)
            if last_step == self.step:
                stuck_count += 1
            if last_step != self.step:
                stuck_count = 0
            if stuck_count >= self.config.neuron.max_stuck_count:
                bt.logging.error(
                    "Heartbeat detecting main process hang, attempting restart"
                )
                autoupdate(force=True)
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
