import asyncio
import time
import os
import aiohttp
import bittensor as bt
import numpy as np
import random
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
        try:
            self.external_ip = requests.get("https://checkip.amazonaws.com", timeout=10).text.strip()
        except Exception as e:
            bt.logging.error(f"Failed to get external IP: {e}. Using fallback.")
            self.external_ip = "localhost"
        self.external_ip = "localhost"  # TEMP
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

        miner_uids = [7, 8]

        bt.logging.info(f"Issuing generative challenge to UIDs: {miner_uids}")

        retries = 3
        prompt_entry = None
        for _ in range(retries):
            prompts = self.content_manager.sample_prompts(k=1)
            if len(prompts) > 0:
                prompt_entry = prompts[0]

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

        #parameters = {"width": 1024, "height": 1024}
        modality = random.choice([Modality.IMAGE, Modality.VIDEO])

        async with aiohttp.ClientSession() as session:
            response_data = await query_generative_miner(
                uid=uid,
                axon_info=self.metagraph.axons[uid],
                session=session,
                hotkey=self.wallet.hotkey,
                prompt=prompt_entry.content,
                modality=modality,
                webhook_url=self.generative_callback_url,
                parameters=None,
                total_timeout=self.config.neuron.miner_total_timeout,
            )

        if response_data and response_data.get("task_id"):
            miner_task_id = response_data.get("task_id")
            async with self.challenge_lock:
                self.challenge_tasks[miner_task_id] = {
                    "uid": uid,
                    "prompt_id": prompt_entry.id,
                    "prompt_content": prompt_entry.content,
                    "modality": modality,
                    "media_type": MediaType.SYNTHETIC,
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
        Only accepts direct binary image and video payloads.
        """
        content_type = request.headers.get("content-type", "").lower()
        
        # Only accept image and video content types
        if not (content_type.startswith("image/") or content_type.startswith("video/")):
            bt.logging.error(f"Invalid content type: {content_type}. Only image/* and video/* are supported.")
            return Response(status_code=400, content="Only image and video content types are supported")
        
        # Get task ID from header
        task_id = request.headers.get("task-id")
        if not task_id:
            bt.logging.error("Binary upload missing task-id header")
            return Response(status_code=400, content="Missing task-id header")
            
        # Get binary payload
        binary_data = await request.body()
        if not binary_data:
            bt.logging.error(f"Task {task_id}: Empty binary payload received")
            return Response(status_code=400, content="Empty binary payload")
            
        async with self.challenge_lock:
            if task_id not in self.challenge_tasks:
                bt.logging.warning(f"Received binary upload for unknown task_id: {task_id}")
                return Response(status_code=404, content="Task not found")

            challenge_info = self.challenge_tasks[task_id]
            generator_uid = challenge_info["uid"]
            bt.logging.info(
                f"Received binary upload for task {task_id}, UID {generator_uid}, "
                f"type: {content_type}, size: {len(binary_data)} bytes"
            )

            filepath = await self.store_binary_content(
                binary_data, content_type, generator_uid, task_id
            )
            if filepath:
                challenge_info["status"] = "completed"
                challenge_info["filepath"] = filepath
                bt.logging.success(f"Task {task_id} completed with binary upload: {filepath}")
            else:
                challenge_info["status"] = "failed"
                challenge_info["error_message"] = "Failed to store binary content"

            del self.challenge_tasks[task_id]
                
        return Response(status_code=200, content="Binary content received")

    async def store_binary_content(
        self, binary_data: bytes, content_type: str, generator_uid: int, task_id: str
    ) -> Optional[str]:
        """
        Store binary content directly uploaded by miner using ContentManager
        """
        try:
            bt.logging.trace(f"Storing binary content for task {task_id}, size: {len(binary_data)} bytes")

            # Get task info from challenge tracker
            task_info = self.challenge_tasks.get(task_id)
            if not task_info:
                bt.logging.error(f"Task {task_id} not found in challenge tasks")
                return None

            modality = task_info["modality"]
            media_type = task_info["media_type"] 
            prompt_id = task_info["prompt_id"]

            # Let ContentManager handle all filesystem and database operations
            filepath = self.content_manager.write_miner_media(
                modality=modality,
                media_type=media_type,
                prompt_id=prompt_id,
                uid=generator_uid,
                hotkey=self.metagraph.hotkeys[generator_uid],
                media_content=binary_data,
                content_type=content_type,
                task_id=task_id,
                model_name=None,
            )

            if filepath:
                bt.logging.success(
                    f"Stored binary content: {filepath} (size: {len(binary_data)} bytes)"
                )
                return filepath
            else:
                bt.logging.error("ContentManager failed to store binary content")
                return None

        except Exception as e:
            bt.logging.error(f"Error storing binary content: {e}")
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
            self.fast_api.stop()
            bt.logging.info("Webhook server stopped")
