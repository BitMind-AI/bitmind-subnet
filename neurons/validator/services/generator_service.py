import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Thread, Event, Lock
from typing import Optional

import bittensor as bt
import torch
import gc
import numpy as np
from PIL import Image

from gas.config import add_args, add_generation_service_args
from gas.generation.tps import nano_banana
from gas.generation import (
    GenerationPipeline,
    initialize_model_registry,
    PromptGenerator,
)
from gas.cache.content_manager import ContentManager
from gas.types import Modality, MediaType
from gas.verification import (
    run_verification,
    get_verification_summary,
    clear_clip_models,
)


class GeneratorService:
    """
    Enhanced LocalGenerator that can run as a standalone service.
    Combines core generation functionality with service management.
    """

    def __init__(self, config):
        self.config = config

        if hasattr(self.config, "cache") and hasattr(self.config.cache, "base_dir"):
            self.config.cache.base_dir = str(
                Path(self.config.cache.base_dir).expanduser()
            )

        self._output_dir = Path(self.config.cache.base_dir)

        self.validator_wallet = bt.wallet(config=self.config)

        self._service_running = False
        self._generation_running = False
        self._stop_requested = Event()
        self._lock = Lock()

        self._generation_thread: Optional[Thread] = None
        self.content_manager = ContentManager(
            base_dir=self.config.cache.base_dir,
            max_per_source=self.config.max_per_source,
            enable_source_limits=self.config.enable_source_limits,
            prune_strategy=self.config.prune_strategy,
            remove_on_sample=self.config.remove_on_sample,
            min_source_threshold=self.config.min_source_threshold,
        )
        self.generation_pipeline: Optional[GenerationPipeline] = None
        self.prompt_generator: Optional[PromptGenerator] = None
        self.model_registry = None
        self.model_names = []

        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if self.hf_token:
            bt.logging.info(
                f"[GENERATOR-SERVICE] HuggingFace token loaded: {self.hf_token[:10]}..."
            )
        else:
            bt.logging.warning(
                "[GENERATOR-SERVICE] No HuggingFace token found in environment"
            )

        self.hf_org = "gasstation"
        self.hf_dataset_repos = {
            "image": f"{self.hf_org}/generated-images",
            "video": f"{self.hf_org}/generated-videos",
        }
        self.upload_batch_size = getattr(config, "upload_batch_size", 50)
        self.videos_per_archive = getattr(config, "videos_per_archive", 25)

        # third party generative services
        self.tp_generators = {"nano_banana": nano_banana.generate_image}

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir):
        """Update the output directory. Only works when not running."""
        if self._generation_running:
            raise RuntimeError(
                "Cannot change output directory while generator is running"
            )
        self._output_dir = Path(new_dir)
        bt.logging.info(
            f"[GENERATOR-SERVICE] Generator output directory updated to: {self._output_dir}"
        )

    def start(self):
        """Start the generator service."""
        bt.logging.info("[GENERATOR-SERVICE] Starting generator service")

        try:
            self._service_running = True
            self._service_loop()
            return True

        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Failed to start generator service: {e}"
            )
            bt.logging.error(traceback.format_exc())
            return False

    def stop(self):
        """Stop the generator service."""
        bt.logging.info("[GENERATOR-SERVICE] Stopping generator service")
        self._service_running = False
        self._stop_generation()

    def _stop_generation(self):
        """Stop the generation process."""
        with self._lock:
            if not self._generation_running:
                bt.logging.warning(
                    "[GENERATOR-SERVICE] Generation process is not running"
                )
                return False

            self._stop_requested.set()
            if self._generation_thread:
                self._generation_thread.join(timeout=10)
            self._generation_running = False
            bt.logging.info("[GENERATOR-SERVICE] Generation process stopped")
            return True

    def _start_generation(self):
        """Start the generation process in a separate thread."""
        with self._lock:
            if self._generation_running:
                bt.logging.warning(
                    "[GENERATOR-SERVICE] Generation process is already running"
                )
                return False

            self._stop_requested.clear()
            self._generation_thread = Thread(
                target=self._generation_worker, daemon=True
            )
            self._generation_thread.start()
            self._generation_running = True
            return True

    def _service_loop(self):
        """Main service loop that manages the generation process lifecycle."""
        bt.logging.info("[GENERATOR-SERVICE] Service loop started")

        while self._service_running:
            try:
                # Start generation process
                if not self._start_generation():
                    bt.logging.error(
                        "[GENERATOR-SERVICE] Failed to start generation process"
                    )
                    time.sleep(60)
                    continue

                while self._generation_running and self._service_running:
                    time.sleep(30)
                    bt.logging.info("[GENERATOR-SERVICE] Service heartbeat")

                # If service is still running but generation stopped, restart it
                if self._service_running:
                    bt.logging.info(
                        "[GENERATOR-SERVICE] Generation process stopped, restarting..."
                    )
                    time.sleep(60)  # Wait before restart

            except KeyboardInterrupt:
                bt.logging.info("[GENERATOR-SERVICE] Received interrupt signal")
                break
            except Exception as e:
                bt.logging.error(f"[GENERATOR-SERVICE] Error in service loop: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(60)  # Wait before retry

        bt.logging.info("[GENERATOR-SERVICE] Service loop stopped")

    def _generation_worker(self):
        """Worker thread that performs the actual generation work."""
        try:
            self._initialize_pipelines()
            self._generation_work_loop()
        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Fatal error in generation worker: {e}"
            )
            bt.logging.error(traceback.format_exc())
        finally:
            self._cleanup()

    def _initialize_pipelines(self):
        """Initialize the generation models and components."""
        bt.logging.info("[GENERATOR-SERVICE] Initializing models...")

        self.model_registry = initialize_model_registry()
        self.generation_pipeline = GenerationPipeline(
            model_registry=self.model_registry,
        )
        self.prompt_generator = PromptGenerator()

        self.model_names = self.model_registry.get_interleaved_model_names()
        bt.logging.info(
            f"[GENERATOR-SERVICE] Initialized with models: {self.model_names}"
        )

    def _generation_work_loop(self):
        """Main generation work loop."""
        bt.logging.info("[GENERATOR-SERVICE] Starting generation work loop")

        prompt_batch_size = self.config.prompt_batch_size
        query_batch_size = self.config.query_batch_size
        local_batch_size = self.config.local_batch_size
        tps_batch_size = self.config.tps_batch_size

        while not self._stop_requested.is_set():
            try:
                self._verify_miner_media(clip_batch_size=32)

                # Generate search queries
                start = time.time()
                self._generate_text("search_query", query_batch_size)
                bt.logging.info(
                    f"[GENERATOR-SERVICE] Generated {query_batch_size} queries in {time.time()-start:.2f} seconds"
                )
                if self._stop_requested.is_set():
                    break

                # Generate prompts
                start = time.time()
                self._generate_text("prompt", prompt_batch_size)
                bt.logging.info(
                    f"[GENERATOR-SERVICE] Generated {prompt_batch_size} prompts in {time.time()-start:.2f} seconds"
                )
                if self._stop_requested.is_set():
                    break

                # Clear GPU memory
                self.prompt_generator.clear_gpu()

                # Generate media with third party services
                self._generate_media(use_local=False, k=tps_batch_size)

                # Generate media with local models
                self._generate_media(use_local=True, k=local_batch_size)

                # Upload batch of miner/validator generated media to HuggingFace
                bt.logging.info("Beginning hf batch upload")
                self.content_manager.upload_batch_to_huggingface(
                    hf_token=self.hf_token,
                    hf_dataset_repos=self.hf_dataset_repos,
                    upload_batch_size=self.upload_batch_size,
                    videos_per_archive=self.videos_per_archive,
                    validator_hotkey=self.validator_wallet.hotkey.ss58_address,
                )

            except Exception as e:
                bt.logging.error(
                    f"[GENERATOR-SERVICE] Error in generation work loop: {e}"
                )
                bt.logging.error(traceback.format_exc())
                time.sleep(3)  # Wait before retrying

    def _generate_text(
        self, content_type: str = "search_query", batch_size: int = 10
    ) -> list:
        """
        Generate a batch of content (search queries or prompts) and save to database.

        Args:
            content_type: Either "search_query" or "prompt"
            batch_size: Number of items to generate

        Returns:
            List of generated content items
        """
        generated = 0
        for _ in range(batch_size):
            if self._stop_requested.is_set():
                break

            try:
                if content_type == "search_query":
                    search_query = self.prompt_generator.generate_search_query()
                    self.content_manager.write_prompt(
                        content=search_query,
                        content_type=content_type,
                        source_media_id=None,
                    )
                    generated += 1

                elif content_type == "prompt":
                    cache_result = self.content_manager.sample_media_with_content(
                        Modality.IMAGE, MediaType.REAL
                    )

                    if not cache_result or not cache_result["count"]:
                        bt.logging.warning("No images available for prompt generation")
                        continue

                    item = cache_result["items"][0]
                    for modality in Modality:
                        prompt = self.prompt_generator.generate_prompt_from_image(
                            Image.fromarray(item["image"]), intended_modality=modality
                        )

                        self.content_manager.write_prompt(
                            content=prompt,
                            content_type=content_type,
                            source_media_id=item["id"],
                        )
                        generated += 1

                else:
                    raise ValueError(f"Unknown content_type: {content_type}")

            except Exception as e:
                bt.logging.error(
                    f"[GENERATOR-SERVICE] Error generating {content_type}: {e}"
                )
                continue

            bt.logging.info(
                f"[GENERATOR-SERVICE] Added {generated} {content_type}s to database"
            )

        return generated

    def _generate_media(self, use_local=True, k=1):
        start = time.time()
        entries = self.content_manager.sample_prompts_with_source_media(
            k=k,
            remove=True,
            strategy="least_used",
        )

        if entries:
            for entry in entries:
                prompt = entry["prompt"]
                original_media = entries[0]["media"]
                bt.logging.debug(f"[GENERATOR-SERVICE] Generating media")
                bt.logging.debug(f"- Prompt: {prompt}")
                bt.logging.debug(f"- Models: {self.model_names}")

                if use_local:
                    # send prompt to all local models
                    for gen_output in self.generation_pipeline.generate_media(
                        prompt=prompt.content,
                        model_names=self.model_names,
                        image_sample=original_media,
                    ):
                        if self._stop_requested.is_set():
                            break

                        if gen_output:
                            save_path = self._write_media(gen_output, prompt.id)
                            if save_path:
                                bt.logging.info(
                                    f"[GENERATOR-SERVICE] Generated and saved media file: {save_path} from prompt '{prompt.id}'"
                                )
                else:
                    # send prompt to third party services
                    for service_name, generator_fn in self.tp_generators.items():
                        bt.logging.info(
                            f"[GENERATOR-SERVICE] Generating with third party service: {service_name}'"
                        )
                        try:
                            gen_output = generator_fn(prompt.content)
                        except RuntimeError as e:
                            bt.logging.warning(e)
                            continue

                        if gen_output:
                            save_path = self._write_media(gen_output, prompt.id)
                            time.sleep(10)
                            if save_path:
                                bt.logging.info(
                                    f"[GENERATOR-SERVICE] Generated and saved media file: {save_path} from prompt '{prompt.id}'"
                                )

                bt.logging.info(
                    f"[GENERATOR-SERVICE] Completed media generation for prompt '{prompt.id}' in {time.time()-start:.2f} seconds"
                )
        else:
            bt.logging.warning(
                "[GENERATOR-SERVICE] No prompts available for media generation"
            )

    def _verify_miner_media(self, threshold: float = 0.25, clip_batch_size: int = 32):
        """
        Verify all pending miner-submitted media using batched CLIP processing with consensus voting.

        Args:
            threshold: Threshold for determining pass/fail (default: 0.25, raw CLIP score)
            clip_batch_size: Batch size for CLIP operations (default: 32, adjust based on GPU memory)
        """
        try:
            bt.logging.info(
                f"[GENERATOR-SERVICE] Starting verification of all pending media (clip_batch_size={clip_batch_size})"
            )

            results = run_verification(
                content_manager=self.content_manager,
                threshold=threshold,
                clip_batch_size=clip_batch_size,
            )

            if not results:
                bt.logging.info("[GENERATOR-SERVICE] No media to verify")
                return

            # Generate and log summary with performance metrics
            summary = get_verification_summary(results)
            bt.logging.info(
                f"[GENERATOR-SERVICE] Batched verification complete: "
                f"{summary['successful']}/{summary['total']} processed, "
                f"{summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors "
                f"(pass rate: {summary['pass_rate']:.1%}, "
                f"avg score: {summary['average_score']:.3f})"
            )

            error_results = [r for r in results if r.verification_score is None]
            if error_results:
                bt.logging.warning(
                    f"[GENERATOR-SEdRVICE] {len(error_results)} verification processing errors"
                )

            failed_results = [
                r for r in results if r.verification_score is not None and not r.passed
            ]
            if failed_results:
                bt.logging.debug(
                    f"[GENERATOR-SERVICE] {len(failed_results)} media failed verification threshold"
                )

            successful_results = [
                r for r in results if r.verification_score is not None
            ]
            if successful_results:
                bt.logging.debug("[GENERATOR-SERVICE] Sample verification results:")
                for result in successful_results[:3]:  # Log first 3 successes
                    score = result.verification_score.get("score", 0)
                    bt.logging.debug(
                        f"  Media {result.media_entry.id}: "
                        f"score={score:.3f}, passed={result.passed}"
                    )
        except Exception as e:
            bt.logging.error(f"[GENERATOR-SERVICE] Error in verification pipeline: {e}")
            bt.logging.error(traceback.format_exc())

        finally:
            clear_clip_models()

    def _write_media(self, media_sample, prompt_id: str):
        """
        Args:
            media_sample: Dictionary containing the generated media (image/video)
            prompt_id: ID of the prompt used for generation
        """
        try:
            modality = media_sample["modality"]
            media_type = media_sample["media_type"]
            model_name = media_sample["model_name"]

            # Handle mask if present
            mask_content = None
            if "mask_image" in media_sample:
                mask_content = np.array(media_sample["mask_image"])

            save_path = self.content_manager.write_generated_media(
                modality=modality,
                media_type=media_type,
                model_name=model_name,
                prompt_id=prompt_id,
                media_content=media_sample[modality],
                mask_content=mask_content,
                generation_args=media_sample.get("generation_args"),
            )

            return save_path

        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Error writing media with ContentManager: {e}"
            )
            bt.logging.error(traceback.format_exc())
            return None

    def _cleanup(self):
        """Clean up resources."""
        try:
            if self.generation_pipeline:
                self.generation_pipeline.shutdown()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            bt.logging.error(f"[GENERATOR-SERVICE] Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="Image+video generation service")

    add_args(parser)
    add_generation_service_args(parser)
    bt.subtensor.add_args(parser)
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)

    bt.logging(config=config, logging_dir=config.neuron.full_path)
    bt.logging.set_info()
    if config.logging.debug:
        bt.logging.set_debug(True)
    if config.logging.trace:
        bt.logging.set_trace(True)

    bt.logging.success(config)
    service = GeneratorService(config)

    try:
        service.start()
    except KeyboardInterrupt:
        bt.logging.info("[GENERATOR-SERVICE] Shutting down generator service")
        service.stop()
    except Exception as e:
        bt.logging.error(f"[GENERATOR-SERVICE] Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
