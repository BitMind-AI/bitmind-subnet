import asyncio
import sys
import io
import time
import signal
import traceback
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
import os
import atexit

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings

for module in ["diffusers", "transformers.tokenization_utils_base"]:
    warnings.filterwarnings("ignore", category=FutureWarning, module=module)

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

import transformers

transformers.logging.set_verbosity_error()

import bittensor as bt
from bitmind.config import add_args, add_data_generator_args
from bitmind.utils import ExitContext, get_metadata
from bitmind.wandb_utils import init_wandb, clean_wandb_cache
from bitmind.types import CacheConfig, MediaType, Modality
from bitmind.cache.sampler import ImageSampler
from bitmind.generation import (
    GenerationPipeline,
    initialize_model_registry,
)


class Generator:
    def __init__(self):
        self.exit_context = ExitContext()
        self.task = None
        self.generation_pipeline = None
        self.image_sampler = None

        self.setup_signal_handlers()
        atexit.register(self.cleanup)

        parser = argparse.ArgumentParser()
        bt.subtensor.add_args(parser)
        bt.wallet.add_args(parser)
        bt.logging.add_args(parser)
        add_data_generator_args(parser)
        add_args(parser)

        self.config = bt.config(parser)

        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_trace()
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        bt.logging.success(self.config)
        wallet_configured = (
            self.config.wallet.name is not None
            and self.config.wallet.hotkey is not None
        )
        if wallet_configured and not self.config.wandb_off:
            try:
                self.wallet = bt.wallet(config=self.config)
                self.uid = (
                    bt.subtensor(
                        config=self.config, network=self.config.subtensor.chain_endpoint
                    )
                    .metagraph(self.config.netuid)
                    .hotkeys.index(self.wallet.hotkey.ss58_address)
                )
                self.wandb_dir = str(Path(__file__).parent.parent)
                clean_wandb_cache(self.wandb_dir)
                self.wandb_run = init_wandb(
                    self.config.copy(),
                    self.config.wandb.process_name,
                    self.uid,
                    self.wallet.hotkey,
                )

            except Exception as e:
                bt.logging.error("Not registered, can't sign W&B run")
                bt.logging.error(e)
                self.config.wandb.off = True

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGQUIT, self.signal_handler)

    def signal_handler(self, sig, frame):
        signal_name = signal.Signals(sig).name
        bt.logging.info(f"Received {signal_name}, initiating shutdown...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self.task and not self.task.done():
            self.task.cancel()

        if self.generation_pipeline:
            try:
                bt.logging.trace("Shutting down generator...")
                self.generation_pipeline.shutdown()
                bt.logging.success("Generator shut down gracefully")
            except Exception as e:
                bt.logging.error(f"Error during generator shutdown: {e}")

        # Force cleanup of any GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                bt.logging.trace("CUDA memory cache cleared")
        except Exception as e:
            pass

    async def wait_for_cache(self, timeout: int = 300):
        """Wait for the cache to be populated with images for prompt generation"""
        start = time.time()
        attempts = 0
        while True:
            if time.time() - start > timeout:
                return False

            available_count = self.image_sampler.get_available_count(use_index=False)
            if available_count > 0:
                return True

            await asyncio.sleep(10)
            if not attempts % 3:
                bt.logging.info("Waiting for images in cache...")
            attempts += 1

    async def sample_images(self, k: int = 1) -> List[Dict[str, Any]]:
        """Sample images from the cache"""
        result = await self.image_sampler.sample(k, remove_from_cache=False)
        if result["count"] == 0:
            raise ValueError("No images available in cache")

        # Convert bytes to PIL images
        for item in result["items"]:
            if isinstance(item["image"], bytes):
                item["image"] = Image.open(io.BytesIO(item["image"]))

        return result["items"]

    async def run(self):
        """Main generator loop"""
        try:
            cache_dir = self.config.cache_dir
            batch_size = self.config.batch_size
            device = self.config.device

            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            self.image_sampler = ImageSampler(
                CacheConfig(
                    modality=Modality.IMAGE.value,
                    media_type=MediaType.REAL.value,
                    base_dir=Path(cache_dir),
                )
            )

            await self.wait_for_cache()
            bt.logging.success("Cache populated. Proceeding to generation.")

            model_registry = initialize_model_registry()
            model_names = model_registry.get_interleaved_model_names(self.config.tasks)
            bt.logging.info(f"Starting generator")
            bt.logging.info(f"Tasks: {self.config.tasks}")
            bt.logging.info(f"Models: {model_names}")

            self.generation_pipeline = GenerationPipeline(
                output_dir=cache_dir,
                device=device,
            )

            gen_count = 0
            batch_count = 0
            while not self.exit_context.isExiting:
                if asyncio.current_task().cancelled():
                    break

                try:
                    image_samples = await self.sample_images(batch_size)
                    bt.logging.info(
                        f"Starting batch generation | Batch Size: {len(image_samples)} | Batch Count: {gen_count}"
                    )

                    start_time = time.time()

                    filepaths = self.generation_pipeline.generate(
                        image_samples, model_names=model_names
                    )
                    await asyncio.sleep(1)

                    duration = time.time() - start_time
                    gen_count += len(filepaths)
                    batch_count += 1
                    bt.logging.info(
                        f"Generated {len(filepaths)} files in batch #{batch_count} in {duration:.2f} seconds"
                    )

                    if not self.config.wandb.off:
                        if batch_count >= self.config.wandb.num_batches_per_run:
                            batch_count = 0
                            self.wandb_run.finish()
                            clean_wandb_cache(self.wandb_dir)
                            self.wandb_run = init_wandb(
                                self.config.copy(),
                                self.config.wandb.process_name,
                                self.uid,
                                self.wallet.hotkey,
                            )

                except asyncio.CancelledError:
                    bt.logging.info("Task cancelled, exiting loop")
                    break
                except Exception as e:
                    bt.logging.error(f"Error in batch processing: {e}")
                    bt.logging.error(traceback.format_exc())
                    await asyncio.sleep(10)
        except Exception as e:
            bt.logging.error(f"Unhandled exception in main task: {e}")
            bt.logging.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()

    def start(self):
        """Start the generator"""
        loop = asyncio.get_event_loop()
        try:
            self.task = asyncio.ensure_future(self.run())
            loop.run_until_complete(self.task)
        except KeyboardInterrupt:
            bt.logging.info("Generator interrupted by KeyboardInterrupt, shutting down")
        except Exception as e:
            bt.logging.error(f"Unhandled exception: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.cleanup()


if __name__ == "__main__":
    generator = Generator()
    generator.start()
    sys.exit(0)
