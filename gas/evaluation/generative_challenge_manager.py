import asyncio
import time
import os
import pickle
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
        save_state_callback=None,
    ):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_type_tracker = miner_type_tracker
        self._save_state_callback = save_state_callback

        self.content_manager = ContentManager(self.config.cache.base_dir)

        self.challenge_tasks = {}
        self.challenge_lock = asyncio.Lock()

        self.external_port = (
            getattr(self.config.neuron, 'external_callback_port', None) or 
            self.config.neuron.callback_port
        )

        try:
            self.external_ip = requests.get("https://checkip.amazonaws.com", timeout=10).text.strip()
        except Exception as e:
            bt.logging.error(f"Failed to get external IP: {e}. Using fallback.")
            self.external_ip = "localhost"
        self.generative_callback_url = f"http://{self.external_ip}:{self.external_port}/generative_callback"

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
            bt.logging.info(
                f"Stored challenge task {miner_task_id} for UID {uid}. Total active tasks: {len(self.challenge_tasks)}"
            )
        else:
            error = response_data.get("error") if response_data else "Unknown error"
            bt.logging.error(f"Failed to send challenge to UID {uid}. Error: {error}")

    async def generative_callback(self, request: Request):
        """Callback endpoint for generative challenges.
        Accepts direct binary image, video, and application/octet-stream payloads.
        """
        content_type = request.headers.get("content-type", "").lower()
        task_id = request.headers.get("task-id")
        client_ip = request.client.host if request.client else "unknown"
        
        uid = "unknown"
        signed_by = request.headers.get("Epistula-Signed-By")
        if task_id and task_id in self.challenge_tasks:
            uid = self.challenge_tasks[task_id].get("uid", "unknown")
        elif signed_by and signed_by in self.metagraph.hotkeys:
            try:
                uid = self.metagraph.hotkeys.index(signed_by)
            except (ValueError, AttributeError):
                pass
        
        bt.logging.debug(f"Generative callback request from UID {uid} (IP: {client_ip}), task_id: {task_id}, content_type: {content_type}")

        # Helper function to format UID with hotkey for better debugging
        def format_uid_info():
            if uid == "unknown" and signed_by:
                return f"UID {uid} (signed-by: {signed_by})"
            return f"UID {uid}"

        if not (
            content_type.startswith("image/") or 
            content_type.startswith("video/") or 
            content_type == "application/octet-stream"
        ):
            bt.logging.error(f"Invalid content type: {content_type} from {format_uid_info()} (IP: {client_ip}). Only image/*, video/*, and application/octet-stream are supported.")
            return Response(status_code=400, content="Only image, video, and application/octet-stream content types are supported")
        
        if not task_id:
            bt.logging.error(f"Binary upload missing task-id header from {format_uid_info()} (IP: {client_ip})")
            return Response(status_code=400, content="Missing task-id header")
            
        binary_data = await request.body()
        if not binary_data:
            bt.logging.error(f"Task {task_id} from {format_uid_info()} (IP: {client_ip}): Empty binary payload received")
            return Response(status_code=400, content="Empty binary payload")
            
        async with self.challenge_lock:
            bt.logging.debug(f"Callback for task {task_id}: Current active tasks: {list(self.challenge_tasks.keys())}")
            if task_id not in self.challenge_tasks:
                # Check if this might be a stale task from a previous session
                bt.logging.debug(f"Received binary upload for unknown task_id: {task_id} from {format_uid_info()} (IP: {client_ip}), content_type: {content_type}, size: {len(binary_data)} bytes")
                # Accept the upload gracefully but don't process it - this reduces 404 spam
                # while still indicating the task wasn't found in our debug logs
                return Response(status_code=200, content="Task not found in current session")

            challenge_info = self.challenge_tasks[task_id]
            generator_uid = challenge_info["uid"]
            
            auth_uid_msg = f" (auth UID: {uid})" if uid != generator_uid and uid != "unknown" else ""
            bt.logging.info(
                f"Received binary upload for task {task_id}, UID {generator_uid}{auth_uid_msg}, "
                f"type: {content_type}, size: {len(binary_data)} bytes (IP: {client_ip})"
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
            
        # Save state after task completion to persist the deletion
        if self._save_state_callback:
            try:
                await self._save_state_callback()
            except Exception as e:
                bt.logging.warning(f"Failed to save state after task completion: {e}")
                
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
        if self.external_port != self.config.neuron.callback_port:
            bt.logging.info(
                f"Advertising external port {self.external_port} to miners"
            )
        bt.logging.info(
            f"Callback URL for miners: {self.generative_callback_url}"
        )

    async def shutdown(self):
        """Shutdown the webhook server gracefully"""
        if hasattr(self, "fast_api"):
            bt.logging.info("Shutting down webhook server...")
            self.fast_api.stop()
            bt.logging.info("Webhook server stopped")

    def save_state(self, save_dir: str, filename: str):
        """Save challenge tasks state to disk"""
        try:
            bt.logging.info(f"Saving challenge tasks state: {len(self.challenge_tasks)} active tasks")
            # Clean up stale tasks before saving (older than 2 hours)
            current_time = time.time()
            stale_tasks = []
            for task_id, task_info in self.challenge_tasks.items():
                if current_time - task_info.get("sent_at", 0) > 7200:  # 2 hours
                    stale_tasks.append(task_id)

            for task_id in stale_tasks:
                del self.challenge_tasks[task_id]
                bt.logging.debug(f"Removed stale task {task_id} during state save")

            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(self.challenge_tasks, f)
            bt.logging.info(f"Successfully saved {len(self.challenge_tasks)} active challenge tasks to {filepath}")
        except Exception as e:
            bt.logging.error(f"Failed to save challenge tasks state: {e}")

    def load_state(self, save_dir: str, filename: str):
        """Load challenge tasks state from disk"""
        try:
            filepath = os.path.join(save_dir, filename)
            if not os.path.exists(filepath):
                bt.logging.debug(f"No challenge tasks state file found at {filepath}")
                return True  # Not an error - just no state to load

            with open(filepath, 'rb') as f:
                loaded_tasks = pickle.load(f)

            current_time = time.time()
            valid_tasks = {}
            for task_id, task_info in loaded_tasks.items():
                if current_time - task_info.get("sent_at", 0) <= 7200:  # 2 hours
                    valid_tasks[task_id] = task_info

            self.challenge_tasks = valid_tasks
            bt.logging.info(f"Loaded {len(self.challenge_tasks)} active challenge tasks from {filepath}")
            return True  # Success
        except Exception as e:
            bt.logging.error(f"Failed to load challenge tasks state: {e}")
            self.challenge_tasks = {}
            return False  # Failure
