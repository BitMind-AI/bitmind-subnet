import asyncio
import sys
import json
import traceback
from time import sleep
from threading import Thread

import numpy as np
from dotenv import load_dotenv
import bittensor as bt
import torch

from gas import __spec_version__ as spec_version
from gas.protocol.validator_requests import get_benchmark_results
from gas.protocol.validator_requests import get_escrow_addresses  # noqa: F401  HOTFIX: temporarily unused
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
    ArtifactTaskManager,
    GenerativeChallengeManager,
    MinerTypeTracker,
    artifact_stats_with_uids,
    get_captioner_rewards,
    get_encoder_rewards,
    get_generator_base_rewards,
    get_generator_reward_multipliers,
    normalize_rewards_to_weight_budget,
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
    "audio_escrow": "5F9Qo4jqurfx3qHsC2kQtvge7Si5aW1BfYKwpxnnpVxouPyF",
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
        self.encoder_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.captioner_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        bt.logging.info(
            f"Initialized score vectors for {len(self.scores)} miners"
        )

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
                self.issue_artifact_tasks,
                self.validate_artifact_outputs,
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
        self.artifact_task_manager = ArtifactTaskManager(
            self.config,
            self.wallet,
            self.metagraph,
            self.subtensor,
            self.miner_type_tracker,
        )

        await self.load_state()
        await self.artifact_task_manager.publish_input_metadata()

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

    @on_block_interval("dps_artifact_task_interval")
    async def issue_artifact_tasks(self, block):
        """Assign encoder/captioner miners R2 artifact data to pull."""
        if not hasattr(self, "artifact_task_manager"):
            return
        await self.artifact_task_manager.issue_artifact_tasks(block)

    @on_block_interval("dps_artifact_task_interval")
    async def validate_artifact_outputs(self, block):
        """Read miner artifact commitments and refresh mechanism-1 reward stats."""
        if not hasattr(self, "artifact_task_manager"):
            return
        await self.artifact_task_manager.validate_miner_outputs()

    @on_block_interval("epoch_length")
    async def set_weights(self, block):
        """
        Query orchestrator for results, computes rewards, updates scores, set weights
        """
        bt.logging.info(f"Updating scores at block {block}")
        active_role_uids = await self.update_scores()
        generator_uids = active_role_uids.get(MinerType.GENERATOR, [])
        encoder_uids = active_role_uids.get(MinerType.ENCODER, [])
        captioner_uids = active_role_uids.get(MinerType.CAPTIONER, [])
        
        if generator_uids is None:
            generator_uids = []
            bt.logging.warning("No generator rewards available; using empty generator_uids")

        # HOTFIX: API unstable; always use hardcoded escrow addresses.
        bt.logging.info("HOTFIX: using hardcoded default escrow addresses")
        active_ss58_addresses = SS58_ADDRESSES
        
        async with self._state_lock:
            bt.logging.debug("set_weights() acquired state lock")
            try:
                bt.logging.info(f"Setting weights at block {block}")
                self.subtensor = bt.subtensor(
                    config=self.config, network=self.config.subtensor.chain_endpoint
                )
                uids = list(range(self.metagraph.n))
                self._extend_score_vectors(len(uids) - 1)

                if np.isnan(self.scores).any():
                    bt.logging.warning(
                        "Scores contain NaN values. This may be due to a lack of "
                        "responses from miners, or a bug in your reward functions."
                    )

                # Mechanism 0: existing SN34 generator/discriminator economics.
                burn_pct      = .6
                video_pct     = .2
                image_pct     = .0
                audio_pct     = .0
                generator_pct = .2

                burn_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["burn"],
                    netuid=self.config.netuid,
                    block=block
                )
                video_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["video_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )
                image_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["image_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )
                audio_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["audio_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )

                special_uids = {burn_uid, image_escrow_uid, video_escrow_uid, audio_escrow_uid}

                generator_weights, generator_unallocated = normalize_rewards_to_weight_budget(
                    self.scores,
                    generator_uids,
                    special_uids,
                    generator_pct,
                )
                normed_weights = generator_weights
                burn_pct += generator_unallocated

                normed_weights[burn_uid]         = burn_pct
                normed_weights[video_escrow_uid] = video_pct
                normed_weights[image_escrow_uid] = image_pct
                normed_weights[audio_escrow_uid] = audio_pct

                # Verify allocations
                total_weight = np.sum(normed_weights)
                actual_burn_rate = normed_weights[burn_uid] / total_weight if total_weight > 0 else 0
                bt.logging.info(
                    f"Total weight sum: {total_weight:.4f}, "
                    f"Actual burn rate: {actual_burn_rate:.4f} (target: {burn_pct}), "
                    f"generator budget: {generator_pct - generator_unallocated:.4f}"
                )

                self.set_weights_fn(
                    self.wallet, self.metagraph, self.subtensor, (uids, normed_weights), mechid=0
                )

                if getattr(self.config, "enable_dps_artifact_mechanism", False):
                    artifact_weights = self._build_artifact_mechanism_weights(
                        encoder_uids=encoder_uids,
                        captioner_uids=captioner_uids,
                        burn_uid=burn_uid,
                        special_uids=special_uids,
                    )
                    artifact_total = np.sum(artifact_weights)
                    bt.logging.info(
                        f"Setting DPS artifact mechanism weights with total sum "
                        f"{artifact_total:.4f}"
                    )
                    self.set_weights_fn(
                        self.wallet,
                        self.metagraph,
                        self.subtensor,
                        (uids, artifact_weights),
                        mechid=getattr(self.config, "dps_artifact_mechanism_id", 1),
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
        # Verification stats from last 4h (all verified, rewarded or not) for base rewards.
        verification_stats = self.content_manager.get_verification_stats_last_n_hours(
            lookback_hours=4.0
        )
        generator_base_rewards, media_ids = get_generator_base_rewards(verification_stats)

        generator_results = await get_benchmark_results(
            self.wallet.hotkey, self.metagraph, base_url=self.config.benchmark_api_url
        )

        # Get generator liveness data for filtering inactive generators
        generator_liveness = None
        max_inactive_hours = 24  # Hours of inactivity before generator rewards are zeroed
        if hasattr(self, 'generative_challenge_manager') and self.generative_challenge_manager:
            generator_liveness = self.generative_challenge_manager.get_all_generator_last_seen()
            if generator_liveness:
                bt.logging.debug(f"Using liveness data for {len(generator_liveness)} generators")
        
        reward_multipliers = get_generator_reward_multipliers(
            generator_results, 
            self.metagraph,
            generator_liveness=generator_liveness,
            max_inactive_hours=max_inactive_hours,
        )
        # HOTFIX: default fool-rate fallback lowered from 0.01 to 0.005.
        DEFAULT_FOOL_RATE_FALLBACK = 0.005
        all_generator_uids = set(generator_base_rewards.keys()) | set(reward_multipliers.keys())
        if not reward_multipliers:
            bt.logging.warning(
                f"No benchmark multipliers; applying default fool rate "
                f"{DEFAULT_FOOL_RATE_FALLBACK} to {len(all_generator_uids)} generators"
            )
        rewards = {
            uid: generator_base_rewards.get(uid, 0)
            * reward_multipliers.get(uid, DEFAULT_FOOL_RATE_FALLBACK)
            for uid in all_generator_uids
        }

        # Alpha for role score EMAs - higher = faster decay, less reward persistence.
        alpha = 0.5
        if hasattr(self, "miner_type_tracker") and self.miner_type_tracker:
            await self.miner_type_tracker.update_miner_types()

        if len(rewards) == 0:
            if not generator_base_rewards:
                bt.logging.info(
                    "No generator rewards: no verified submissions on this validator in the last 4h."
                )
            else:
                bt.logging.trace(
                    "No generator rewards: no base rewards or multipliers available."
                )
            artifact_rewards = self._update_artifact_scores(alpha)
            return {
                MinerType.GENERATOR: [],
                MinerType.ENCODER: self._active_encoder_uids(
                    artifact_rewards[MinerType.ENCODER]
                ),
                MinerType.CAPTIONER: self._active_captioner_uids(
                    artifact_rewards[MinerType.CAPTIONER]
                ),
            }

        max_reward_uid = max(list(rewards.keys()) + [-1])
        self._extend_score_vectors(max(max_reward_uid, self._metagraph_uid_count() - 1))

        reward_arr = np.array([rewards.get(i, 0) for i in range(len(self.scores))])

        # 0.5 = 50% new rewards, 50% historical (aggressive decay for inactive miners)
        self.scores = alpha * reward_arr + (1 - alpha) * self.scores

        # Hard cutoff: zero out scores for generators not active within liveness window
        # This prevents inactive miners from retaining any accumulated score via EMA decay
        if generator_liveness:
            inactive_count = 0
            for uid in range(len(self.scores)):
                if uid < len(self.metagraph.hotkeys):
                    hotkey = self.metagraph.hotkeys[uid]
                    if hotkey not in generator_liveness and self.scores[uid] > 0:
                        self.scores[uid] = 0
                        inactive_count += 1
            if inactive_count > 0:
                bt.logging.info(f"Zeroed scores for {inactive_count} inactive generators (not seen in {max_inactive_hours}h)")

        bt.logging.info(
            f"Updated scores for {len(rewards)} miners with EMA (alpha={alpha})"
        )

        if media_ids:
            success = self.content_manager.mark_media_rewarded(media_ids)
            if success:
                bt.logging.info(f"Marked {len(media_ids)} media entries as rewarded")
            else:
                bt.logging.warning("Failed to mark media as rewarded")

        artifact_rewards = self._update_artifact_scores(alpha)
        encoder_rewards = artifact_rewards[MinerType.ENCODER]
        captioner_rewards = artifact_rewards[MinerType.CAPTIONER]

        return {
            MinerType.GENERATOR: list(all_generator_uids),
            MinerType.ENCODER: self._active_encoder_uids(encoder_rewards),
            MinerType.CAPTIONER: self._active_captioner_uids(captioner_rewards),
        }

    def _update_artifact_scores(self, alpha: float):
        artifact_stats = self._load_artifact_reward_stats()
        hotkeys = getattr(self.metagraph, "hotkeys", [])
        encoder_stats = artifact_stats_with_uids(
            self._select_artifact_role_stats(artifact_stats, "encoder"),
            hotkeys,
        )
        captioner_stats = artifact_stats_with_uids(
            self._select_artifact_role_stats(artifact_stats, "captioner"),
            hotkeys,
        )
        encoder_rewards = get_encoder_rewards(
            encoder_stats
        )
        captioner_rewards = get_captioner_rewards(
            captioner_stats
        )

        if not encoder_rewards:
            encoder_rewards = {}
        if not captioner_rewards:
            captioner_rewards = {}

        max_uid = max(
            list(encoder_rewards.keys()) + list(captioner_rewards.keys()) + [-1]
        )
        self._extend_score_vectors(max(max_uid, self._metagraph_uid_count() - 1))

        if encoder_rewards:
            encoder_reward_arr = np.array(
                [encoder_rewards.get(i, 0) for i in range(len(self.encoder_scores))]
            )
            self.encoder_scores = alpha * encoder_reward_arr + (1 - alpha) * self.encoder_scores
            bt.logging.info(
                f"Updated encoder scores for {len(encoder_rewards)} miners with EMA (alpha={alpha})"
            )

        if captioner_rewards:
            captioner_reward_arr = np.array(
                [captioner_rewards.get(i, 0) for i in range(len(self.captioner_scores))]
            )
            self.captioner_scores = alpha * captioner_reward_arr + (1 - alpha) * self.captioner_scores
            bt.logging.info(
                f"Updated captioner scores for {len(captioner_rewards)} miners with EMA (alpha={alpha})"
            )

        return {
            MinerType.ENCODER: encoder_rewards,
            MinerType.CAPTIONER: captioner_rewards,
        }

    def _extend_score_vectors(self, max_uid: int):
        target_len = max_uid + 1
        if target_len <= 0:
            return
        self.scores = self._extend_score_vector(self.scores, target_len)
        self.encoder_scores = self._extend_score_vector(
            self.encoder_scores, target_len
        )
        self.captioner_scores = self._extend_score_vector(
            self.captioner_scores, target_len
        )

    def _extend_score_vector(self, scores, target_len: int):
        extend_scores = target_len - len(scores)
        if extend_scores <= 0:
            return scores
        return np.append(scores, np.zeros(extend_scores))

    def _metagraph_uid_count(self) -> int:
        n = getattr(self.metagraph, "n", 0)
        if hasattr(n, "item"):
            n = n.item()
        return int(n)

    def _load_artifact_reward_stats(self):
        rewards_path = getattr(self.config, "dps_artifact_rewards_path", None)
        if not rewards_path:
            if hasattr(self, "artifact_task_manager"):
                return getattr(self.artifact_task_manager, "latest_reward_stats", {})
            return {}

        try:
            with open(rewards_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload
        except FileNotFoundError:
            bt.logging.warning(
                f"DPS artifact rewards file not found: {rewards_path}"
            )
        except Exception as e:
            bt.logging.error(f"Error loading DPS artifact rewards from {rewards_path}: {e}")
            bt.logging.error(traceback.format_exc())
        return {}

    def _select_artifact_role_stats(self, artifact_stats, role: str):
        if not isinstance(artifact_stats, dict):
            return artifact_stats if role == "encoder" else {}

        role_keys = (
            f"{role}_stats",
            f"{role}_rewards",
            role,
        )
        for key in role_keys:
            if key in artifact_stats:
                return artifact_stats[key]

        # Backward compatibility with the first encoder-only handoff format.
        if role == "encoder" and any(
            key in artifact_stats for key in ("uid", "total_verified", "pass_rate")
        ):
            return [artifact_stats]
        if role == "encoder":
            return artifact_stats
        return {}

    def _active_encoder_uids(self, rewards=None):
        return self._active_artifact_uids(MinerType.ENCODER, rewards)

    def _active_captioner_uids(self, rewards=None):
        return self._active_artifact_uids(MinerType.CAPTIONER, rewards)

    def _active_artifact_uids(self, miner_type: MinerType, rewards=None):
        tracker_uids = set()
        if hasattr(self, "miner_type_tracker") and self.miner_type_tracker:
            tracker_uids = set(
                self.miner_type_tracker.get_miners_by_type(miner_type)
            )
        reward_uids = set(rewards.keys()) if rewards else set()
        return sorted(tracker_uids | reward_uids)

    def _build_artifact_mechanism_weights(
        self,
        encoder_uids,
        captioner_uids,
        burn_uid: int,
        special_uids,
    ):
        artifact_config = getattr(self.config, "dps_artifact", None)
        encoder_budget = float(getattr(artifact_config, "encoder_weight", 0.8))
        captioner_budget = float(getattr(artifact_config, "captioner_weight", 0.2))
        budget_total = encoder_budget + captioner_budget
        if budget_total <= 0:
            encoder_budget = 1.0
            captioner_budget = 0.0
            budget_total = 1.0
        encoder_budget /= budget_total
        captioner_budget /= budget_total

        encoder_weights, encoder_unallocated = normalize_rewards_to_weight_budget(
            self.encoder_scores,
            encoder_uids,
            special_uids,
            encoder_budget,
        )
        captioner_weights, captioner_unallocated = normalize_rewards_to_weight_budget(
            self.captioner_scores,
            captioner_uids,
            special_uids,
            captioner_budget,
        )
        artifact_weights = encoder_weights + captioner_weights
        artifact_weights[burn_uid] = encoder_unallocated + captioner_unallocated
        bt.logging.info(
            "DPS artifact mechanism budget: "
            f"encoder={encoder_budget - encoder_unallocated:.4f}, "
            f"captioner={captioner_budget - captioner_unallocated:.4f}, "
            f"burn fallback={artifact_weights[burn_uid]:.4f}"
        )
        return artifact_weights

    async def log_on_block(self, block):
        """
        Log information about validator state at regular intervals.
        """
        try:
            blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
            bt.logging.info(
                f"Forward Block: {block} | Blocks till Set Weights: {blocks_till}"
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
                state_data = {
                    "scores.npy": self.scores,
                    "encoder_scores.npy": self.encoder_scores,
                    "captioner_scores.npy": self.captioner_scores,
                }
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
            state_data_keys = [
                "scores.npy",
                "encoder_scores.npy",
                "captioner_scores.npy",
            ]
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
                if "encoder_scores.npy" in loaded_state:
                    self.encoder_scores = loaded_state["encoder_scores.npy"]
                else:
                    self.encoder_scores = np.zeros_like(self.scores)
                if "captioner_scores.npy" in loaded_state:
                    self.captioner_scores = loaded_state["captioner_scores.npy"]
                else:
                    self.captioner_scores = np.zeros_like(self.scores)
                bt.logging.info(f"Loaded scores vector for {len(self.scores)} miners")
                bt.logging.info(
                    f"Loaded encoder scores vector for {len(self.encoder_scores)} miners"
                )
                bt.logging.info(
                    f"Loaded captioner scores vector for {len(self.captioner_scores)} miners"
                )
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
