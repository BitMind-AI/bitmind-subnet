import asyncio
import time
import os
import aiohttp
import bittensor as bt
import numpy as np
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import Response
from typing import Dict, Any, Optional
from pathlib import Path

from gas.protocol.epistula import get_verifier, generate_header
from gas.protocol.validator_requests import query_generative_miner
from gas.types import MinerType, Modality, MediaType
from gas.cache.content_db import PromptEntry
from gas.cache.content_manager import ContentManager
import requests


class GenerativeChallengeManager:
    def __init__(
        self,
        config,
        wallet,
        metagraph,
        subtensor,
        miner_type_tracker,
    ):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_type_tracker = miner_type_tracker

        self.content_manager = ContentManager(self.config.cache.base_dir)

        self.challenge_tasks = {}
        self.challenge_lock = asyncio.Lock()
        self.external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        self.generative_callback_url = f"http://{self.external_ip}:{self.config.neuron.callback_port}/generative_callback"

        self.init_fastapi()

    async def issue_generative_challenge(self):
        await self.miner_type_tracker.update_miner_types()
        miner_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)

        if len(miner_uids) > self.config.neuron.sample_size:
            miner_uids = np.random.choice(
                miner_uids,
                size=self.config.neuron.sample_size,
                replace=False,
            ).tolist()

        if not miner_uids:
            bt.logging.trace("No generative miners found to challenge.")
            return

        bt.logging.info(f"Issuing generative challenge to UIDs: {miner_uids}")

        prompt_entry = await self._get_prompt_from_cache()
        if not prompt_entry:
            bt.logging.info(
                "Waiting for prompt cache to be populated. Skipping generative challenge."
            )
            return

        await asyncio.gather(
            *[self.send_generative_request(uid, prompt_entry) for uid in miner_uids]
        )

    async def send_generative_request(self, uid: int, prompt_entry):
        """Scoring is handled by the callback in GeneratorEvaluator"""

        parameters = {"width": 1024, "height": 1024}  # dummy params

        # Randomly select modality for this challenge.
        modality = np.random.choice([Modality.IMAGE.value, Modality.VIDEO.value])
        media_type = np.random.choice(
            [MediaType.SYNTHETIC.value, MediaType.SEMISYNTHETIC.value]
        )

        async with aiohttp.ClientSession() as session:
            response_data = await query_generative_miner(
                uid=uid,
                axon_info=self.metagraph.axons[uid],
                session=session,
                hotkey=self.wallet.hotkey,
                prompt=prompt_entry.content,
                modality=Modality(modality),
                media_type=MediaType(media_type),
                webhook_url=self.generative_callback_url,
                parameters=parameters,
                total_timeout=self.config.neuron.miner_total_timeout,
            )

        if response_data and response_data.get("task_id"):
            miner_task_id = response_data.get("task_id")
            async with self.challenge_lock:
                self.challenge_tasks[miner_task_id] = {
                    "uid": uid,
                    "prompt_id": prompt_entry.id,
                    "modality": modality,
                    "media_type": media_type,
                    "status": "pending",
                    "sent_at": time.time(),
                }
            bt.logging.trace(
                f"Sent generative challenge to UID {uid}, task_id: {miner_task_id}, prompt_id: {prompt_entry.id}"
            )
        else:
            error = response_data.get("error") if response_data else "Unknown error"
            bt.logging.error(f"Failed to send challenge to UID {uid}. Error: {error}")

    async def generative_callback(self, request: Request):
        """Callback endpoint for generative challenges.
        The miner will call this endpoint to provide the results of a generative challenge.
        """
        data = await request.json()
        task_id = data.get("task_id")

        async with self.challenge_lock:
            if task_id not in self.challenge_tasks:
                bt.logging.warning(f"Received callback for unknown task_id: {task_id}")
                return Response(status_code=404, content="Task not found")

            challenge_info = self.challenge_tasks[task_id]
            generator_uid = challenge_info["uid"]
            bt.logging.info(
                f"Received callback for task {task_id}, issued to UID {generator_uid}"
            )

            status = data.get("status")
            if status == "completed":
                download_url = data.get("download_url")
                if download_url:
                    bt.logging.success(
                        f"Task {task_id} completed successfully. Download from: {download_url}"
                    )
                    filepath = await self.download_generator_content(
                        download_url, generator_uid, task_id
                    )
                    if filepath:
                        challenge_info["status"] = "completed"
                        challenge_info["filepath"] = filepath
                    else:
                        challenge_info["status"] = "failed"
                        challenge_info["error_message"] = "Failed to download content"

                else:
                    bt.logging.error(f"Task {task_id} completed but no media provided")
                    challenge_info["status"] = "failed"
                    challenge_info["error_message"] = "No media provided"

            elif status == "failed":
                error_message = data.get("error_message")
                bt.logging.error(f"Task {task_id} failed: {error_message}")
                challenge_info["status"] = "failed"
                challenge_info["error_message"] = error_message

            # Don't delete the task yet - keep it for evaluation tracking
            # del self.challenge_tasks[task_id]

        return Response(status_code=200, content="Callback received")

    async def download_generator_content(
        self, download_url: str, generator_uid: int, task_id: str
    ) -> str:
        """
        Download and store generator content with proper Epistula authentication
        """
        try:
            bt.logging.trace(f"Attempting to download from: {download_url}")

            axon_info = self.metagraph.axons[generator_uid]
            if download_url.startswith("/"):
                download_url = f"http://{axon_info.ip}:{axon_info.port}" + download_url

            # need to sign epistula headers if downloading directly from miner
            headers = {"Content-Type": "application/json"}
            if axon_info.ip in download_url:
                epistula_headers = generate_header(
                    self.wallet.hotkey, b"", axon_info.hotkey
                )
                headers.update(epistula_headers)

            modality = self.challenge_tasks[task_id]["modality"]
            media_type = self.challenge_tasks[task_id]["media_type"]

            cache_path = self.content_manager.get_cache_path(
                Modality(modality), MediaType(media_type), adversarial=True
            )
            output_dir = cache_path / str(generator_uid)
            output_dir.mkdir(exist_ok=True, parents=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, headers=headers) as response:
                    if response.status != 200:
                        bt.logging.error(
                            f"Failed to download content from {download_url}: HTTP {response.status}"
                        )
                        return None

                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type:
                        file_ext = ".png"
                    elif "video" in content_type:
                        file_ext = ".mp4"
                    else:
                        file_ext = ".bin"

                    filename = f"{task_id}{file_ext}"
                    file_path = output_dir / filename
                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

                    try:
                        prompt_id = self.challenge_tasks[task_id]["prompt_id"]
                        combined_metadata = {
                            "source": f"generator_{generator_uid}_{self.metagraph.hotkeys[generator_uid]}",
                            "generator_uid": generator_uid,
                            "task_id": task_id,
                            "task_sent_at": self.challenge_tasks[task_id]["sent_at"],
                            "prompt": self.challenge_tasks[task_id]["prompt"],
                            "download_url": download_url,
                            "content_type": content_type,
                            "file_size": file_path.stat().st_size,
                            "label": MediaType(media_type).int_value,
                        }

                        media_id = self.content_manager.add_existing_media(
                            file_path=str(file_path),
                            modality=modality,
                            media_type=media_type,
                            model_name=f"miner_{generator_uid}",
                            prompt_id=prompt_id,
                            metadata=combined_metadata,
                        )
                        bt.logging.trace(
                            f"Added media entry to database with ID: {media_id} linked to prompt: {prompt_id}"
                        )
                    except Exception as e:
                        bt.logging.warning(
                            f"Failed to add media entry to database: {e}"
                        )

                    bt.logging.success(
                        f"Downloaded and stored generator content: {file_path} (type: {media_type})"
                    )
                    return str(file_path)

        except Exception as e:
            bt.logging.error(f"Error downloading generator content: {e}")
            return None

    async def _get_prompt_from_cache(self, retries=3) -> Optional[PromptEntry]:
        """Get a prompt from generated prompt files"""
        try:
            for _ in range(retries):

                prompts = self.content_manager.sample_prompts(k=1)
                if len(prompts) > 0:
                    return prompts[0]

                # Fallback: try to get prompt from media metadata
                media_entries = self.content_manager.sample_media(
                    k=1,
                    modality="image",
                    media_type="synthetic",
                    remove=False,
                    strategy="random",
                )
                if not media_entries:
                    continue

                metadata = media_entries[0].metadata or {}
                if metadata.get("prompt"):
                    # fallback
                    return PromptEntry(
                        id=f"fallback_{int(time.time())}",
                        content=metadata.get("prompt"),
                        content_type="prompt",
                        created_at=time.time(),
                        used_count=0,
                        last_used=None,
                        metadata={"source": "fallback_metadata"},
                    )

        except Exception as e:
            bt.logging.warning(f"Error getting prompt from cache: {e}")

        bt.logging.warning(f"Failed to retrieve prompt after {retries} retries")

        return None

    def init_fastapi(self):
        """Initialize the FastAPI server for generative challenge callbacks."""
        if self.config.neuron.callback_port is None:
            bt.logging.warning(
                "Callback port not set, generative challenges will not have a callback."
            )
            return

        self.api = FastAPI()
        self.router = APIRouter()

        verifier = get_verifier(
            self.wallet, self.metagraph, no_force_validator_permit=True
        )

        self.router.add_api_route(
            "/generative_callback",
            self.generative_callback,
            dependencies=[Depends(verifier)],
            methods=["POST"],
            include_in_schema=False,
        )
        self.api.include_router(self.router)

        fast_config = uvicorn.Config(
            self.api,
            host="0.0.0.0",
            port=self.config.neuron.callback_port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(
            f"FastAPI server started on port {self.config.neuron.callback_port}"
        )

    async def shutdown(self):
        """Shutdown the webhook server gracefully"""
        if hasattr(self, "fast_api"):
            bt.logging.info("Shutting down webhook server...")
            await self.fast_api.stop()
            bt.logging.info("Webhook server stopped")
