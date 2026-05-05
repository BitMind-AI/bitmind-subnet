import argparse
import asyncio
import inspect
import sys
import os
import tempfile
import time
import traceback
from threading import Event, Lock, Thread
from typing import Optional
from pathlib import Path

import bittensor as bt
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY

from gas.cache import ContentManager
from gas.config import add_args, add_data_service_args
from gas.datasets import load_all_datasets
from gas.datasets.download import download_and_extract
from gas.types import MediaType, Modality, SourceType
from gas.utils import on_block_interval
from gas.utils.metagraph import SubstrateConnectionManager
from gas.protocol.gas_api_validator import post_generator_verification_upload

# Same window as generator base rewards in neurons/validator/validator.py
_VERIFICATION_STATS_LOOKBACK_HOURS = 4.0


class DataService:
    """
    Data service that drives dataset downloads, periodic HuggingFace uploads,
    and disk cleanup on a substrate block schedule. Threading is managed
    internally.
    """

    def __init__(self, config):
        self.config = config

        if hasattr(self.config, "cache") and hasattr(self.config.cache, "base_dir"):
            self.config.cache.base_dir = str(
                Path(self.config.cache.base_dir).expanduser()
            )

        # Ensure temporary files live under the cache directory to avoid /tmp OOM
        try:
            tmp_base = Path(self.config.cache.base_dir) / "tmp"
            tmp_base.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TMPDIR", str(tmp_base))
            os.environ.setdefault("TEMP", str(tmp_base))
            os.environ.setdefault("TMP", str(tmp_base))
            tempfile.tempdir = str(tmp_base)
            bt.logging.info(f"[DATA-SERVICE] Using temp dir: {tmp_base}")
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Failed to set temp dir under cache: {e}")

        # used for filesystem/db writes
        self.content_manager = ContentManager(
            base_dir=self.config.cache.base_dir,
            max_per_source=self.config.max_per_source,
            enable_source_limits=self.config.enable_source_limits,
            prune_strategy=self.config.prune_strategy,
            remove_on_sample=self.config.remove_on_sample,
            min_source_threshold=self.config.min_source_threshold,
        )

        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
        if not self.hf_token:
            raise RuntimeError(
                "[DATA-SERVICE] HUGGINGFACE_HUB_TOKEN (or deprecated HUGGING_FACE_TOKEN) "
                "environment variable must be set"
            )
        self.hf_dataset_repos = {
            "image": self.config.hf_image_repo,
            "video": self.config.hf_video_repo,
        }

        try:
            self.validator_wallet = bt.wallet(config=self.config)
        except Exception:
            self.validator_wallet = None

        # Validator UID is resolved lazily on first upload to avoid a
        # synchronous metagraph fetch at startup.
        self.validator_uid: Optional[int] = None
        self._validator_uid_resolved = False

        if self.validator_wallet:
            bt.logging.info(
                f"[DATA-SERVICE] Validator hotkey: "
                f"{self.validator_wallet.hotkey.ss58_address[:16]}..."
            )

        self.all_datasets = [d for d in load_all_datasets() if d.modality == Modality.IMAGE]
        self.downloader_thread: Optional[Thread] = None
        self.uploader_thread: Optional[Thread] = None
        self.uploader_thread_start_time: Optional[float] = None
        self.uploader_lock = Lock()
        self.stop_event = Event()

        self.is_running = False
        self.step = 0

        # Required by the @on_block_interval decorator. DataService's init is
        # fully synchronous, so callbacks are safe to run as soon as the
        # substrate subscription starts.
        self.initialization_complete = True

        self.substrate_manager = SubstrateConnectionManager(
            url=self.config.subtensor.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY
        )
        self.substrate_task = None

        self.block_callbacks = [
            self.log_on_block,
            self.start_dataset_download,
            self.start_upload_cycle,
            self.start_cleanup_cycle,
        ]

    def _resolve_validator_uid(self) -> Optional[int]:
        """Look up the validator's UID from the metagraph on first use."""
        if self._validator_uid_resolved:
            return self.validator_uid
        self._validator_uid_resolved = True
        if not self.validator_wallet:
            return None
        try:
            subtensor = bt.subtensor(config=self.config)
            metagraph = subtensor.metagraph(netuid=self.config.netuid)
            hotkey = self.validator_wallet.hotkey.ss58_address
            hotkeys_list = list(metagraph.hotkeys)
            if hotkey in hotkeys_list:
                self.validator_uid = hotkeys_list.index(hotkey)
            else:
                bt.logging.warning(
                    f"[DATA-SERVICE] Validator hotkey {hotkey[:16]}... not found in metagraph"
                )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Could not lookup validator UID: {e}")
        return self.validator_uid

    async def start(self):
        """Start the data service with substrate connection."""
        bt.logging.info("[DATA-SERVICE] Starting data service")

        # Enforce hard caps at startup in case max_per_source was lowered
        try:
            pruned = self.content_manager.enforce_source_caps()
            if pruned:
                bt.logging.info(
                    f"[DATA-SERVICE] Enforced source caps on startup: {pruned}"
                )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Failed to enforce caps on startup: {e}")

        self.is_running = True
        bt.logging.success("[DATA-SERVICE] Data service started successfully")

        await self.start_dataset_download(0)

        # Start substrate subscription
        self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
        bt.logging.info("🚀 Data service substrate subscription started")

        # Main service loop
        while self.is_running:
            self.step += 1

            # Check substrate connection health
            if self.substrate_task is not None and self.substrate_task.done():
                bt.logging.info("Data service substrate connection lost, restarting...")
                try:
                    self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
                    bt.logging.info("✅ Data service substrate connection restarted")
                except Exception as e:
                    bt.logging.error(f"Failed to restart data service substrate task: {e}")

            if self.step % 60 == 0:
                self.log_status()

            await asyncio.sleep(1)

    async def stop(self):
        """Stop the data service."""
        bt.logging.info("[DATA-SERVICE] Stopping data service")
        self.is_running = False

        if self.substrate_manager:
            self.substrate_manager.stop()

        if self.substrate_task and not self.substrate_task.done():
            self.substrate_task.cancel()
            try:
                await self.substrate_task
            except asyncio.CancelledError:
                pass

        self.stop_event.set()

        if self.downloader_thread and self.downloader_thread.is_alive():
            self.downloader_thread.join(timeout=10)

        bt.logging.success("[DATA-SERVICE] Data service stopped")

    async def run_callbacks(self, block):
        for callback in self.block_callbacks:
            try:
                res = callback(block)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                bt.logging.error(
                    f"Failed running callback {callback.__name__}: {str(e)}"
                )
                bt.logging.error(traceback.format_exc())

    @on_block_interval("dataset_interval")
    async def start_dataset_download(self, block):
        """Start dataset download at the specified block interval."""
        try:
            bt.logging.info(
                f"[DATA-SERVICE] Starting dataset download at block {block}"
            )

            # Start downloader thread if not already running
            if not self.downloader_thread or not self.downloader_thread.is_alive():
                self.downloader_thread = Thread(
                    target=self._downloader_worker, daemon=True
                )
                self.downloader_thread.start()
                bt.logging.info("[DATA-SERVICE] Dataset downloader thread started")

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error starting dataset download: {e}")
            bt.logging.error(traceback.format_exc())

    def _downloader_worker(self):
        try:
            datasets_to_download = [
                d for d in self.all_datasets
                if self.content_manager.needs_more_data(SourceType.DATASET, d.path)
            ]

            if not datasets_to_download:
                bt.logging.warning("[DATA-SERVICE] No datasets need downloading")
                return

            # Prioritize datasets with zero items first, then by ascending count
            dataset_with_counts = [
                (d, self.content_manager.get_source_count(SourceType.DATASET, d.path))
                for d in datasets_to_download
            ]
            prioritized_datasets = [
                d for d, _ in sorted(dataset_with_counts, key=lambda t: (t[1] != 0, t[1]))
            ]

            names = [d.path for d in prioritized_datasets]
            bt.logging.info(f"[DATA-SERVICE] Starting download for {len(prioritized_datasets)} datasets (priority): {names}")

            total_saved = 0
            for dataset in prioritized_datasets:
                # Check if we still need data for this dataset
                if not self.content_manager.needs_more_data(SourceType.DATASET, dataset.path):
                    continue

                for media_bytes, metadata in download_and_extract(
                    dataset,
                    images_per_parquet=self.config.dataset_images_per_parquet,
                    parquet_per_dataset=self.config.dataset_parquet_per_dataset,
                    temp_dir=str(Path(self.config.cache.base_dir) / "tmp"),
                ):
                    if self.stop_event.is_set():
                        break

                    # Check again if we still need data
                    if not self.content_manager.needs_more_data(SourceType.DATASET, dataset.path):
                        break

                    if self._write_dataset_media(media_bytes, metadata):
                        total_saved += 1

            bt.logging.success(
                f"[DATA-SERVICE] Dataset download completed. Saved {total_saved} media files."
            )

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error in dataset download worker: {e}")
            bt.logging.error(traceback.format_exc())

    def _write_dataset_media(self, media_content, metadata: dict) -> bool:
        """Write dataset media using ContentManager."""
        try:
            modality_raw = metadata.get("modality")
            media_type_raw = metadata.get("media_type")
            if not modality_raw or not media_type_raw:
                bt.logging.warning(
                    f"[DATA-SERVICE] Skipping dataset item missing modality/media_type: "
                    f"modality={modality_raw!r} media_type={media_type_raw!r}"
                )
                return False

            # `metadata["modality"]` may be a `Modality` enum (which subclasses
            # `str`, so `.lower()` returns the value) or a plain string.
            # `Modality(...)` is idempotent for the enum case.
            dataset_source_file = metadata.get("source_parquet") or metadata.get(
                "source_zip", "unknown"
            )
            dataset_index = metadata.get("original_index") or metadata.get(
                "path_in_zip", "unknown"
            )
            dataset_name = metadata.get("dataset_path", "unknown")

            save_path = self.content_manager.write_dataset_media(
                modality=Modality(modality_raw.lower()),
                media_type=MediaType(media_type_raw),
                media_content=media_content,
                dataset_name=dataset_name,
                dataset_source_file=dataset_source_file,
                dataset_index=str(dataset_index),
                resolution=(
                    tuple(metadata["image_size"]) if "image_size" in metadata else None
                ),
            )

            if save_path:
                bt.logging.trace(f"[DATA-SERVICE] Saved dataset media: {save_path}")
                return True
            bt.logging.warning(
                f"[DATA-SERVICE] Failed to save dataset media from {dataset_name}"
            )
            return False

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error writing dataset media: {e}")
            return False

    @on_block_interval("upload_check_interval")
    async def start_upload_cycle(self, block):
        """Start uploader thread if not already running and threshold met."""
        with self.uploader_lock:
            if self.uploader_thread and self.uploader_thread.is_alive():
                # Treat threads running longer than upload_max_duration_hours as hung
                max_duration = self.config.upload_max_duration_hours * 3600
                elapsed = time.time() - (self.uploader_thread_start_time or 0)
                if elapsed < max_duration:
                    return
                bt.logging.warning(
                    f"[DATA-SERVICE] Uploader thread has been alive for {elapsed / 3600:.1f}h "
                    f"(max {self.config.upload_max_duration_hours:.1f}h) — treating as hung, "
                    "allowing new upload cycle"
                )
                self.uploader_thread = None

            try:
                total = self.content_manager.content_db.count_unuploaded_media(modality="image")
                if total < self.config.upload_image_threshold:
                    bt.logging.debug(
                        f"[DATA-SERVICE] Upload threshold not met: "
                        f"{total} < {self.config.upload_image_threshold}"
                    )
                    return

                batches = min(
                    self.config.upload_max_batches,
                    total // self.config.upload_image_threshold,
                )
                bt.logging.info(
                    f"[DATA-SERVICE] Upload threshold met: {total} >= {self.config.upload_image_threshold}. "
                    f"Will upload {batches} batch(es)"
                )
            except Exception as e:
                bt.logging.warning(f"[DATA-SERVICE] Unable to compute upload threshold: {e}")
                return

            if batches <= 0:
                bt.logging.info("[DATA-SERVICE] Upload skipped (waiting for upload batch minimum)")
                return

            self.uploader_thread = Thread(
                target=self._uploader_worker, args=(batches,), daemon=True
            )
            self.uploader_thread_start_time = time.time()
            self.uploader_thread.start()

    def _uploader_worker(self, batches: int) -> None:
        try:
            bt.logging.info(
                f"[DATA-SERVICE] Uploading {batches} batch(es) of "
                f"{self.config.upload_batch_size} files per modality"
            )
            validator_uid = self._resolve_validator_uid()
            uploaded = self.content_manager.upload_batch_to_huggingface(
                hf_token=self.hf_token,
                hf_dataset_repos=self.hf_dataset_repos,
                upload_batch_size=self.config.upload_batch_size,
                images_per_archive=self.config.images_per_archive,
                videos_per_archive=self.config.videos_per_archive,
                validator_hotkey=(
                    self.validator_wallet.hotkey.ss58_address
                    if self.validator_wallet else None
                ),
                validator_uid=validator_uid,
                num_batches=batches,
            )
            if uploaded and uploaded > 0 and self.validator_wallet:
                stats = self.content_manager.get_verification_stats_last_n_hours(
                    lookback_hours=_VERIFICATION_STATS_LOOKBACK_HOURS,
                )
                post_generator_verification_upload(
                    self.validator_wallet,
                    self.config.benchmark_api_url,
                    _VERIFICATION_STATS_LOOKBACK_HOURS,
                    stats,
                )
        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Uploader error: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            with self.uploader_lock:
                self.uploader_thread = None
                self.uploader_thread_start_time = None

    async def log_on_block(self, block):
        """Log service status on block events."""
        try:
            bt.logging.info(f"[DATA-SERVICE] Block: {block} | Step: {self.step}")
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Error in log_on_block: {e}")

    @on_block_interval("cleanup_interval")
    async def start_cleanup_cycle(self, block):
        """
        Clean up uploaded+rewarded media to free disk space.
        Runs once per cleanup_interval (default ~24 hours).
        """
        try:
            bt.logging.info(f"[DATA-SERVICE] Starting cleanup cycle at block {block}")
            
            min_age_hours = getattr(self.config, 'cleanup_min_age_hours', 48.0)
            batch_size = getattr(self.config, 'cleanup_batch_size', 1000)
            
            stats = self.content_manager.cleanup_uploaded_media(
                min_age_hours=min_age_hours,
                require_rewarded=True,
                batch_size=batch_size,
            )
            
            if stats['media_deleted'] > 0:
                bt.logging.success(
                    f"[DATA-SERVICE] Cleanup complete: {stats['media_deleted']} media, "
                    f"{stats['prompts_deleted']} prompts, {stats['files_deleted']} files removed"
                )
            else:
                bt.logging.info("[DATA-SERVICE] Cleanup complete: no eligible media to clean")
                
        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error in cleanup cycle: {e}")
            bt.logging.error(traceback.format_exc())

    def log_status(self):
        """Log current service status."""
        try:
            downloader_status = (
                "Running"
                if (self.downloader_thread and self.downloader_thread.is_alive())
                else "Stopped"
            )
            uploader_status = (
                "Running"
                if (self.uploader_thread and self.uploader_thread.is_alive())
                else "Stopped"
            )
            bt.logging.info(
                f"[DATA-SERVICE] Status - Downloader: {downloader_status}, "
                f"Uploader: {uploader_status}"
            )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Error getting status: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Validator data service")

    add_args(parser)
    add_data_service_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    config = bt.config(parser)

    bt.logging(config=config, logging_dir=config.neuron.full_path)
    bt.logging.set_info()
    if config.logging.debug:
        bt.logging.set_debug(True)
    if config.logging.trace:
        bt.logging.set_trace(True)

    bt.logging.success(config)

    # Create and run service
    service = DataService(config)

    try:
        await service.start()
    except KeyboardInterrupt:
        bt.logging.info("[DATA-SERVICE] Shutting down data service")
        await service.stop()
    except Exception as e:
        bt.logging.error(f"[DATA-SERVICE] Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
