import asyncio
import json
import os
import shutil
import sys
import threading
import time
import traceback
from threading import Thread
from time import sleep
from typing import Any, Dict, Optional

import aiohttp
import bittensor as bt
import numpy as np
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from substrateinterface import SubstrateInterface

from bitmind import __spec_version__ as spec_version
from bitmind.autoupdater import autoupdate
from bitmind.cache import CacheSystem
from bitmind.config import MAINNET_UID
from bitmind.encoding import media_to_bytes
from bitmind.epistula import query_miner
from bitmind.metagraph import (
    create_set_weights,
    get_miner_uids,
    run_block_callback_thread,
)
from bitmind.scoring import EvalEngine
from bitmind.transforms import apply_random_augmentations
from bitmind.types import (
    MediaType,
    Modality,
    NeuronType,
)
from bitmind.utils import on_block_interval, print_info
from bitmind.wandb_utils import WandbLogger
from neurons.base import BaseNeuron


class Validator(BaseNeuron):
    neuron_type = NeuronType.VALIDATOR
    cache_system: Optional[CacheSystem] = None
    heartbeat_thread: Thread
    lock_waiting = False
    lock_halt = False
    step = 0
    initialization_complete: bool = False

    def __init__(self, config=None, run_init=True):
        super().__init__(config=config)

        ## Typesafety
        self.set_weights = create_set_weights(spec_version, self.config.netuid)

        ## CHECK IF REGG'D
        if (
            not self.metagraph.validator_permit[self.uid]
            and self.config.netuid == MAINNET_UID
        ):
            bt.logging.error("Validator does not have vpermit")
            sys.exit(1)
        if run_init:
            self.init()

    def init(self):
        assert self.config.netuid
        assert self.config.vpermit_tao_limit
        assert self.config.subtensor

        self._validate_challenge_probs()

        if not self.config.wandb_off:
            self.wandb_logger = WandbLogger(self.config, self.uid, self.wallet.hotkey)

        bt.logging.info(self.config)
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        self.eval_engine = EvalEngine(self.metagraph, self.config)

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.set_weights_on_interval,
                self.send_challenge_to_miners_on_interval,
                self.update_compressed_cache_on_interval,
                self.update_media_cache_on_interval,
                self.start_new_wanbd_run_on_interval,
            ]
        )

        # SETUP HEARTBEAT THREAD
        if self.config.neuron.heartbeat:
            self.heartbeat_thread = Thread(name="heartbeat", target=self.heartbeat)
            self.heartbeat_thread.start()

        ## DONE
        bt.logging.info(
            "\N{GRINNING FACE WITH SMILING EYES}", "Successfully Initialized!"
        )

    async def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        assert self.config.vpermit_tao_limit
        bt.logging.info(
            f"Running validator {self.uid} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        await self.load_state()

        self.cache_system = CacheSystem()
        await self.cache_system.initialize(
            self.config.cache.base_dir,
            self.config.cache.max_compressed_gb,
            self.config.cache.max_media_gb,
            self.config.cache.media_files_per_source,
        )

        self.initialization_complete = True
        bt.logging.success(
            f"Initialization Complete. Validator starting at block: {self.subtensor.block}"
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

    @on_block_interval("challenge_interval")
    async def send_challenge_to_miners_on_interval(self, block):
        assert self.config.vpermit_tao_limit

        miner_uids = get_miner_uids(
            self.metagraph, self.uid, self.config.vpermit_tao_limit
        )
        if len(miner_uids) > self.config.neuron.sample_size:
            miner_uids = np.random.choice(
                miner_uids, size=self.config.neuron.sample_size, replace=False
            ).tolist()

        media_sample = await self._sample_media()
        if not media_sample:
            bt.logging.warning("Waiting for cache to populate. Challenge skipped.")
            return

        modality = media_sample["modality"]
        media = media_sample[modality]

        media_bytes, content_type = media_to_bytes(
            media, fps=media_sample.get("fps", None)
        )

        bt.logging.info(f"---------- Starting Challenge at Block {block} ----------")
        bt.logging.info(f"Sampled from {modality} cache")

        challenge_tasks = []
        challenge_results = []
        async with aiohttp.ClientSession() as session:
            for uid in miner_uids:
                axon_info = self.metagraph.axons[uid]
                challenge_tasks.append(
                    query_miner(
                        uid,
                        media_bytes,
                        content_type,
                        modality,
                        axon_info,
                        session,
                        self.wallet.hotkey,
                        self.config.neuron.miner_total_timeout,
                        self.config.neuron.miner_connect_timeout,
                        self.config.neuron.miner_sock_connect_timeout,
                        testnet_metadata=(
                            {k: v for k, v in media_sample.items() if k != modality}
                            if self.config.netuid != MAINNET_UID
                            else {}
                        ),
                    )
                )
            if len(challenge_tasks) != 0:
                responses = await asyncio.gather(*challenge_tasks)
                challenge_results.extend(responses)
                challenge_tasks = []

        valid_responses = [r for r in challenge_results if not r["error"]]
        n_valid = len(valid_responses)
        n_failures = len(challenge_results) - len(valid_responses)
        bt.logging.info(
            f"Received {n_valid} valid miner responses. ({n_failures} others failed.)"
        )

        bt.logging.info(f"Scoring {modality} challenge")
        rewards = self.eval_engine.score_challenge(
            uids=[r["uid"] for r in challenge_results],
            hotkeys=[r["hotkey"] for r in challenge_results],
            predictions=[r["prediction"] for r in challenge_results],
            errors=[r["error"] for r in challenge_results],
            label=media_sample["label"],
            challenge_modality=modality
        )

        self.log_challenge_results(media_sample, challenge_results, rewards)

        await self.save_state()
        bt.logging.success(f"---------- Challenge Complete ----------")

    @on_block_interval("compressed_cache_update_interval")
    async def update_compressed_cache_on_interval(self, block):
        if (
            hasattr(self, "_compressed_cache_thread")
            and self._compressed_cache_thread.is_alive()
        ):
            bt.logging.warning(
                f"Previous compressed cache update still running at block {block}, skipping this update"
            )
            return

        def update_compressed_cache():
            """Thread function to update compressed cache."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.cache_system.update_compressed_caches())
                bt.logging.info(f"Compressed cache update complete")
            except Exception as e:
                bt.logging.error(f"Error updating compressed caches: {e}")
                bt.logging.error(traceback.format_exc())
            finally:
                loop.close()

        bt.logging.info(f"Updating compressed caches at block {block}")
        self._compressed_cache_thread = threading.Thread(
            target=update_compressed_cache, daemon=True
        )
        self._compressed_cache_thread.start()

    @on_block_interval("media_cache_update_interval")
    async def update_media_cache_on_interval(self, block):
        if hasattr(self, "_media_cache_thread") and self._media_cache_thread.is_alive():
            bt.logging.warning(
                f"Previous media cache update still running at block {block}, skipping this update"
            )
            return

        def update_media_cache():
            """Thread function to update media cache."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.cache_system.update_media_caches())
                bt.logging.info(f"Media cache update complete")
            except Exception as e:
                bt.logging.error(f"Error updating media caches: {e}")
                bt.logging.error(traceback.format_exc())
            finally:
                loop.close()

        bt.logging.info(f"Updating media caches at block {block}")
        self._media_cache_thread = threading.Thread(
            target=update_media_cache, daemon=True
        )
        self._media_cache_thread.start()

    @on_block_interval("epoch_length")
    async def set_weights_on_interval(self, block):
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
            weights = self.eval_engine.get_weights()
            uids = list(range(self.metagraph.n))

            self.set_weights(
                self.wallet, self.metagraph, self.subtensor, (uids, weights)
            )
            bt.logging.success("Weights set successfully")

        except Exception as e:
            bt.logging.error(f"Error in set_weights_on_interval: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.lock_halt = False

    @on_block_interval("wandb_restart_interval")
    async def start_new_wanbd_run_on_interval(self, block):
        try:
            self.wandb_logger.start_new_run()
        except Exception as e:
            bt.logging.error(f"Not able to start new W&B run: {e}")

    async def _sample_media(self) -> Optional[Dict[str, Any]]:
        """
        Sample a media item from the cache system.

        Returns:
            Dictionary with media item details or None if sampling fails
        """
        if not self.cache_system:
            return None

        modality, media_type, multi_video = self.determine_challenge_type()

        kwargs = {}
        if modality == Modality.VIDEO:
            kwargs = {
                "min_duration": self.config.challenge.min_clip_duration,
                "max_duration": self.config.challenge.max_clip_duration,
                "max_frames": self.config.challenge.max_frames,
            }

        try:
            sampler_name = f"{media_type}_{modality}_sampler"
            results = await self.cache_system.sample(sampler_name, 1, **kwargs)
        except Exception as e:
            bt.logging.error(f"Error sampling media with {sampler_name}: {e}")
            return None

        if not results or results.get("count", 0) == 0:
            return None

        sample = results["items"][0]

        if multi_video:
            try:
                # for now we stitch up to 2 videos together
                max_duration = (
                    self.config.challenge.max_clip_duration
                    - sample["segment"]["duration"]
                )
                results = await self.cache_system.sample(
                    sampler_name, 1, max_duration=max_duration
                )
            except Exception as e:
                bt.logging.error(f"Error sampling media with {sampler_name}: {e}")
                return None

            if results and results.get("count", 0) > 0:
                sample = {"sample_0": sample, "sample_1": results["items"][0]}
                sample["video"] = (
                    sample["sample_0"]["video"],
                    sample["sample_1"]["video"],
                )
                del sample["sample_0"]["video"]
                del sample["sample_1"]["video"]

        if sample and sample.get(modality) is not None:
            bt.logging.debug("Augmenting Media")
            augmented_media, aug_level, aug_params = apply_random_augmentations(
                sample.get(modality),
                (256, 256),
                sample.get("mask_center", None),
            )
            sample[modality] = augmented_media
            sample.update(
                {
                    "modality": modality,
                    "media_type": media_type,
                    "label": MediaType(media_type).int_value,
                    "metadata": sample.get("metadata", {}),
                    "augmentation_level": aug_level,
                    "augmentation_params": aug_params
                }
            )
            return sample

        return None

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

        return modality, media_type, multi_video

    async def log_on_block(self, block):
        """
        Log information about validator state at regular intervals.

        Args:
            block: Current block number
        """
        blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
        bt.logging.info(
            f"Forward Block: {self.subtensor.block} | Blocks till Set Weights: {blocks_till}"
        )
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

        if self.cache_system and block % 5 == 0:
            try:
                for name, sampler in self.cache_system.samplers.items():
                    count = sampler.get_available_count()
                    bt.logging.info(f"Cache status: {name} has {count} available items")

                compressed_blocks = self.config.compressed_cache_update_interval - (
                    block % self.config.compressed_cache_update_interval
                )
                media_blocks = self.config.media_cache_update_interval - (
                    block % self.config.media_cache_update_interval
                )
                bt.logging.info(
                    f"Next compressed cache update in {compressed_blocks} blocks"
                )
                bt.logging.info(f"Next media cache update in {media_blocks} blocks")
            except Exception as e:
                bt.logging.error(f"Error logging cache status: {e}")

    def log_challenge_results(self, media_sample, challenge_results, rewards):
        uids = [d["uid"] for d in challenge_results]
        results = {
            "miner_uids": uids,
            "miner_hotkeys": [d["hotkey"] for d in challenge_results],
            "response_statuses": [d["status"] for d in challenge_results],
            "response_errors": [d["error"] for d in challenge_results],
            "predictions": [d["prediction"] for d in challenge_results],
            "challenge_metadata": {
                k: v for k, v in media_sample.items() if k != media_sample["modality"]
            },
        }

        results["rewards"] = [rewards.get(uid, 0) for uid in uids]
        results["scores"] = [self.eval_engine.scores[uid] for uid in uids]
        results["metrics"] = [self.eval_engine.get_miner_metrics(uid) for uid in uids]

        valid_indices = [
            i for i in range(len(uids)) if not results["response_errors"][i]
        ]
        invalid_indices = [i for i in range(len(uids)) if results["response_errors"][i]]

        if self.config.netuid == MAINNET_UID:
            for i in invalid_indices:
                bt.logging.warning(
                    f"UID: {results['miner_uids'][i]} | "
                    f"HOTKEY: {results['miner_hotkeys'][i]} | "
                    f"STATUS: {results['response_statuses'][i]} | "
                    f"ERROR: {results['response_errors'][i]}"
                )

        for i in valid_indices:
            bt.logging.success(
                f"UID: {results['miner_uids'][i]} | "
                f"HOTKEY: {results['miner_hotkeys'][i]} | "
                f"PRED: {results['predictions'][i]}"
            )
            video_metrics = {
                "video_" + k: f"{v:.4f}"
                for k, v in results["metrics"][i]["video"].items()
            }
            video_metrics = [
                f"{k.upper()}: {float(v)}" for k, v in video_metrics.items()
            ]
            image_metrics = {
                "image_" + k: f"{v:.4f}"
                for k, v in results["metrics"][i]["image"].items()
            }
            image_metrics = [
                f"{k.upper()}: {float(v)}" for k, v in image_metrics.items()
            ]
            bt.logging.success(
                f"{' | '.join(video_metrics)} | "
                f"{' | '.join(image_metrics)} | "
                f"REWARD: {results['rewards'][i]} | "
                f"SCORE: {results['scores'][i]}"
            )

        bt.logging.info(json.dumps(results["challenge_metadata"], indent=2))

        if not self.config.wandb_off:
            self.wandb_logger.log(
                media_sample=media_sample,
                challenge_results=results,
            )

    def _validate_challenge_probs(self):
        """
        Validates that the challenge probabilities in config sum to 1.0.
        """
        total_modality = (
            self.config.challenge.image_prob + self.config.challenge.video_prob
        )
        total_media = (
            self.config.challenge.real_prob
            + self.config.challenge.synthetic_prob
            + self.config.challenge.semisynthetic_prob
        )

        if abs(total_modality - 1.0) > 1e-6:
            raise ValueError(
                f"Modality probabilities must sum to 1.0, got {total_modality} "
                f"(image_prob={self.config.challenge.image_prob}, "
                f"video_prob={self.config.challenge.video_prob})"
            )

        if abs(total_media - 1.0) > 1e-6:
            raise ValueError(
                f"Media type probabilities must sum to 1.0, got {total_media} "
                f"(real_prob={self.config.challenge.real_prob}, "
                f"synthetic_prob={self.config.challenge.synthetic_prob}, "
                f"semisynthetic_prob={self.config.challenge.semisynthetic_prob})"
            )

    async def save_state(self):
        """
        Atomically save validator state (scores + miner history)
        Maintains the current state and one backup.
        """
        self.lock_halt = True
        while not self.lock_waiting:
            sleep(self.config.neuron.lock_sleep_seconds)

        try:
            base_dir = self.config.neuron.full_path
            os.makedirs(base_dir, exist_ok=True)

            current_dir = os.path.join(base_dir, "state_current")
            backup_dir = os.path.join(base_dir, "state_backup")
            temp_dir = os.path.join(base_dir, "state_temp")

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            os.makedirs(temp_dir)

            # save to temp dir
            self.eval_engine.save_state(temp_dir)
            with open(os.path.join(temp_dir, "complete"), "w") as f:
                f.write("1")

            # backup current state
            if os.path.exists(current_dir):
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                os.rename(current_dir, backup_dir)

            # move temp to current
            os.rename(temp_dir, current_dir)

            bt.logging.success("Saved validator state")

        except Exception as e:
            bt.logging.error(f"Error during state save: {str(e)}")
            bt.logging.error(traceback.format_exc())
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        finally:
            self.lock_halt = False

    async def load_state(self):
        """
        Load validator state, falling back to backup if needed.
        """
        base_dir = self.config.neuron.full_path
        current_dir = os.path.join(base_dir, "state_current")
        backup_dir = os.path.join(base_dir, "state_backup")

        try:
            if os.path.exists(current_dir) and os.path.exists(
                os.path.join(current_dir, "complete")
            ):
                bt.logging.trace(
                    f"Attempting to load current validator state {current_dir}"
                )
                success = self.eval_engine.load_state(current_dir)
                if success:
                    bt.logging.info("Successfully loaded current validator state")
                    return True
                else:
                    bt.logging.warning("Failed to load current state, trying backup")
            else:
                bt.logging.warning(
                    "Current state not found or incomplete, trying backup"
                )

            # fall back to backup if needed
            if os.path.exists(backup_dir) and os.path.exists(
                os.path.join(backup_dir, "complete")
            ):
                current_time = time.time()
                complete_marker = os.path.join(backup_dir, "complete")
                marker_mod_time = os.path.getmtime(complete_marker)
                backup_age_hours = (current_time - marker_mod_time) / 3600

                max_age_hours = self.config.neuron.max_state_backup_hours
                if backup_age_hours > max_age_hours:
                    bt.logging.warning(
                        f"Backup is {backup_age_hours:.2f} hours old (> {max_age_hours} hours), skipping load"
                    )
                    return False

                bt.logging.trace(
                    f"Attempting to load backup validator state {backup_dir} (age: {backup_age_hours:.2f} hours)"
                )
                success = self.eval_engine.load_state(backup_dir)
                if success:
                    bt.logging.info(
                        f"Successfully loaded backup validator state (age: {backup_age_hours:.2f} hours)"
                    )
                    return True
                else:
                    bt.logging.error("Failed to load backup state")
                    return False
            else:
                bt.logging.warning("No valid state found")
                return False
        except Exception as e:
            bt.logging.error(f"Error during state load: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False

    async def shutdown(self):
        """Shutdown the validator and clean up resources."""
        bt.logging.info("Shutting down validator")


if __name__ == "__main__":
    try:
        validator = Validator()
        asyncio.run(validator.run())
    except KeyboardInterrupt:
        bt.logging.info("Validator interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
