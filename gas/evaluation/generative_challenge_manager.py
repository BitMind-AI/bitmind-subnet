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
    ):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_type_tracker = miner_type_tracker

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
                
        return Response(status_code=200, content="Binary content received")

    async def store_binary_content(
        self, binary_data: bytes, content_type: str, generator_uid: int, task_id: str
    ) -> Optional[str]:
        """
        Store binary content directly uploaded by miner using ContentManager.
        
        Performs pre-storage validation and REJECTS:
        - Duplicate content (perceptual hash match)
        - Content without valid C2PA from trusted AI generators
        - Corrupted/unreadable media
        
        Only content that passes all checks is stored and eligible for HuggingFace upload.
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

            # Get modality string for checks
            modality_str = modality.value if hasattr(modality, 'value') else str(modality)
            miner_hotkey = self.metagraph.hotkeys[generator_uid]

            # Step 1: Check for corrupted/unreadable media
            try:
                is_corrupted = self._check_media_corrupted(binary_data, modality_str)
                if is_corrupted:
                    bt.logging.warning(
                        f"REJECTED corrupted media from UID {generator_uid} task {task_id}: "
                        f"media is unreadable or invalid"
                    )
                    return None
            except Exception as e:
                bt.logging.debug(f"Corruption check error (allowing): {e}")

            # Step 2: Compute perceptual hash and check for duplicates within same prompt
            perceptual_hash = None
            try:
                from gas.verification.duplicate_detection import compute_media_hash, DEFAULT_HAMMING_THRESHOLD

                perceptual_hash = compute_media_hash(binary_data, modality=modality_str)
                if perceptual_hash:
                    duplicate_info = self.content_manager.check_duplicate(
                        perceptual_hash, 
                        threshold=DEFAULT_HAMMING_THRESHOLD,
                        prompt_id=prompt_id,
                    )
                    if duplicate_info:
                        dup_media_id, dup_distance = duplicate_info
                        bt.logging.warning(
                            f"REJECTED duplicate from UID {generator_uid} task {task_id}: "
                            f"matches media {dup_media_id} with distance {dup_distance}"
                        )
                        return None  # Reject duplicates
            except Exception as e:
                bt.logging.debug(f"Duplicate detection skipped: {e}")

            # Step 3: Verify C2PA content credentials - REQUIRE trusted issuer
            c2pa_verified = False
            c2pa_issuer = None
            try:
                from gas.verification.c2pa_verification import verify_c2pa, C2PA_AVAILABLE

                if not C2PA_AVAILABLE:
                    bt.logging.error(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"c2pa-python library not installed - required for verification"
                    )
                    return None

                c2pa_result = verify_c2pa(binary_data)
                if c2pa_result.verified and c2pa_result.is_trusted_issuer:
                    c2pa_verified = True
                    c2pa_issuer = c2pa_result.issuer
                    bt.logging.info(
                        f"C2PA verified for UID {generator_uid}: issuer={c2pa_issuer}"
                    )
                else:
                    # Reject if no valid C2PA from trusted source
                    rejection_reason = "no C2PA manifest" if not c2pa_result.verified else "untrusted issuer"
                    bt.logging.warning(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"C2PA check failed ({rejection_reason})"
                    )
                    return None
            except ImportError:
                bt.logging.error(
                    f"REJECTED from UID {generator_uid} task {task_id}: "
                    f"c2pa-python library not installed - required for verification"
                )
                return None
            except Exception as e:
                bt.logging.error(
                    f"REJECTED from UID {generator_uid} task {task_id}: "
                    f"C2PA verification error: {e}"
                )
                return None

            # Step 4: All checks passed - store the media
            filepath = self.content_manager.write_miner_media(
                modality=modality,
                media_type=media_type,
                prompt_id=prompt_id,
                uid=generator_uid,
                hotkey=miner_hotkey,
                media_content=binary_data,
                content_type=content_type,
                task_id=task_id,
                model_name=None,
                perceptual_hash=perceptual_hash,
                c2pa_verified=c2pa_verified,
                c2pa_issuer=c2pa_issuer,
            )

            if filepath:
                bt.logging.success(
                    f"Stored verified content: {filepath} (size: {len(binary_data)} bytes, "
                    f"hash: {perceptual_hash[:16] if perceptual_hash else 'N/A'}..., "
                    f"c2pa_issuer: {c2pa_issuer})"
                )
                return filepath
            else:
                bt.logging.error("ContentManager failed to store binary content")
                return None

        except Exception as e:
            bt.logging.error(f"Error storing binary content: {e}")
            return None

    def _check_media_corrupted(self, binary_data: bytes, modality: str) -> bool:
        """
        Check if media data is corrupted/unreadable.
        
        Returns:
            True if corrupted, False if valid
        """
        import io
        
        try:
            if modality == "image":
                from PIL import Image
                img = Image.open(io.BytesIO(binary_data))
                img.verify()  # Verify it's a valid image
                return False
            elif modality == "video":
                import tempfile
                import cv2
                
                # Write to temp file and try to open with cv2
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(binary_data)
                    tmp_path = tmp.name
                
                try:
                    cap = cv2.VideoCapture(tmp_path)
                    if not cap.isOpened():
                        return True
                    
                    # Try to read at least one frame
                    ret, frame = cap.read()
                    cap.release()
                    
                    if not ret or frame is None:
                        return True
                    
                    return False
                finally:
                    import os
                    os.unlink(tmp_path)
            else:
                return False  # Unknown modality, don't reject
                
        except Exception as e:
            bt.logging.debug(f"Media corruption check failed: {e}")
            return True  # If we can't verify, assume corrupted


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
