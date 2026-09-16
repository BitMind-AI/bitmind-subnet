import asyncio
import hashlib
import io
import json
import os
import pickle
import tempfile
import threading
import time
import uuid

from concurrent.futures import ThreadPoolExecutor
import aiohttp
import bittensor as bt
import cv2
import requests
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import Response
from PIL import Image
from typing import Dict, Optional

from gas.cache.content_manager import ContentManager
from gas.evaluation.challenge_allocation import allocate_challenge_slots
from gas.evaluation.resolution_tiers import sample_challenge_tier
from gas.evaluation.rewards import GeneratorQualification, resolve_generator_qualification
from gas.protocol.epistula import get_verifier
from gas.protocol.validator_requests import query_generative_miner
from gas.types import MediaType, MinerType, Modality
from gas.verification.c2pa_verification import verify_c2pa
from gas.verification.duplicate_detection import (
    compute_media_hash,
    DEFAULT_HAMMING_THRESHOLD,
)

# Allowance for provider/validator clock disagreement when checking that a
# C2PA manifest was signed after its challenge was issued. Providers sign
# with their own infrastructure clocks; ten minutes absorbs realistic skew
# without meaningfully weakening the freshness guarantee (challenges are
# issued ~44 minutes apart).
C2PA_MAX_CLOCK_SKEW_SECONDS = 600


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
        # Use threading.Lock instead of asyncio.Lock because FastAPIThreadedServer
        # runs in a separate thread with its own event loop
        self.challenge_lock = threading.Lock()

        # Track generator liveness: hotkey -> last activity timestamp
        # Updated when a generator successfully responds to a challenge
        self.generator_last_seen: Dict[str, float] = {}

        # Fool-rate qualification from the last successful generator-results fetch.
        # qualification_fresh is False until update_scores writes a live map; a
        # missing/stale map treats every UID as onboarding so sampling does not
        # freeze on the last qualified set.
        self.qualification: Optional[Dict[str, GeneratorQualification]] = None
        self.qualification_fresh: bool = False

        # Keep expensive OpenCV/C2PA work off the callback event loop, but
        # bound concurrency so a burst of video uploads cannot create dozens
        # of memory-heavy decoder/verifier jobs at once.
        self.media_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="callback-media"
        )

        self.callback_staging_dir = os.path.join(self.config.cache.base_dir, "callback_staging")
        os.makedirs(self.callback_staging_dir, exist_ok=True)

        self.external_port = (
            getattr(self.config.neuron, 'external_callback_port', None) or
            self.config.neuron.callback_port
        )

        configured_ip = getattr(self.config.neuron, 'external_ip', None)
        if configured_ip:
            self.external_ip = configured_ip
        else:
            try:
                self.external_ip = requests.get("https://checkip.amazonaws.com", timeout=10).text.strip()
            except Exception as e:
                bt.logging.error(f"Failed to get external IP: {e}. Using fallback.")
                self.external_ip = "localhost"
        self.generative_callback_url = f"http://{self.external_ip}:{self.external_port}/generative_callback"

        self.init_fastapi()

    def set_qualification(
        self,
        qualification: Optional[Dict[str, GeneratorQualification]],
        fresh: bool = True,
    ) -> None:
        """Cache fool-rate qualification by hotkey, never by reusable UID."""
        self.qualification = qualification
        self.qualification_fresh = bool(fresh and qualification is not None)

    async def issue_generative_challenge(self):
        await self.miner_type_tracker.update_miner_types()
        miner_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)

        if not miner_uids:
            bt.logging.trace("No generative miners found to challenge.")
            return

        raw = getattr(self.config, 'prompt_modalities', 'video')
        available = []
        for item in raw.split(','):
            item = item.strip().lower()
            if item == 'image':
                available.append(Modality.IMAGE)
            elif item == 'video':
                available.append(Modality.VIDEO)
        if not available:
            available = [Modality.VIDEO]

        sample_size = int(self.config.neuron.sample_size)
        prompt_pools = {}
        for mod in available:
            entries = self.content_manager.sample_prompts(
                k=sample_size, modality=mod.value, remove=False, strategy="least_used",
            )
            if entries:
                prompt_pools[mod] = entries

        if not prompt_pools:
            bt.logging.info(
                "Waiting for prompt cache to be populated. Skipping generative challenge."
            )
            return

        available_names = [mod.value for mod in prompt_pools]
        # Registrations can change between score updates. Resolve identities
        # for each round so replacements enter onboarding, even with fresh data.
        qualification = (
            resolve_generator_qualification(self.qualification, self.metagraph)
            if self.qualification_fresh and self.qualification is not None else None
        )
        scoring = getattr(self.config, "scoring", None)
        lookback_hours = float(getattr(scoring, "no_answer_lookback_hours", 24.0))
        response_stats = self.content_manager.get_challenge_response_stats(
            lookback_hours=lookback_hours
        )
        assignments, pool_stats = allocate_challenge_slots(
            miner_uids,
            available_names,
            qualification,
            sample_size=sample_size,
            qualified_slots=int(getattr(self.config.neuron, "qualified_slots", 36)),
            onboarding_slots=int(getattr(self.config.neuron, "onboarding_slots", 8)),
            probe_slots=int(getattr(self.config.neuron, "probe_slots", 6)),
            min_fool_samples=int(getattr(scoring, "min_fool_samples", 20)),
            response_stats=response_stats,
            min_no_answer_attempts=int(
                getattr(scoring, "min_no_answer_attempts", 5)
            ),
        )

        if not assignments:
            bt.logging.trace("No generative miners found to challenge.")
            return

        bt.logging.info(
            f"Challenge pools: image_qualified={pool_stats['image_qualified']} "
            f"video_qualified={pool_stats['video_qualified']} "
            f"onboarding={pool_stats['onboarding']} probe={pool_stats['probe']} "
            f"image_unresponsive={pool_stats['image_unresponsive']} "
            f"video_unresponsive={pool_stats['video_unresponsive']} "
            f"rolled_onboarding={pool_stats['rolled_onboarding']}"
        )
        bt.logging.info(f"Issuing generative challenge to UIDs: {[uid for uid, _ in assignments]}")

        modality_for = {Modality.IMAGE.value: Modality.IMAGE, Modality.VIDEO.value: Modality.VIDEO}
        tasks = []
        modality_counters = {name: 0 for name in available_names}
        for uid, mod_name in assignments:
            pool = prompt_pools[modality_for[mod_name]]
            ix = modality_counters[mod_name] % len(pool)
            modality_counters[mod_name] += 1
            tasks.append(
                self.send_generative_request(uid, pool[ix], modality_for[mod_name])
            )

        await asyncio.gather(*tasks)

    async def send_generative_request(self, uid: int, prompt_entry, modality: Modality):
        """Scoring is handled by the callback in GeneratorEvaluator"""

        # Challenges request a resolution tier (video: 480p/720p/1080p,
        # image: 1K/2K/4K); rewards are priced at min(observed, requested) so
        # overshooting never pays more and miners whose model can't reach the
        # tier degrade to a lower (video: shortfall-discounted) price, not
        # failure.
        requested_resolution = sample_challenge_tier(modality.value)
        parameters = {"resolution": requested_resolution}

        async with aiohttp.ClientSession() as session:
            response_data = await query_generative_miner(
                uid=uid,
                axon_info=self.metagraph.axons[uid],
                session=session,
                hotkey=self.wallet.hotkey,
                prompt=prompt_entry.content,
                modality=modality,
                webhook_url=self.generative_callback_url,
                parameters=parameters,
                total_timeout=self.config.neuron.miner_total_timeout,
            )

        if response_data and response_data.get("task_id"):
            miner_task_id = response_data.get("task_id")
            miner_hotkey = self.metagraph.hotkeys[uid]
            with self.challenge_lock:
                self.challenge_tasks[miner_task_id] = {
                    "uid": uid,
                    "hotkey": miner_hotkey,
                    "prompt_id": prompt_entry.id,
                    "prompt_content": prompt_entry.content,
                    "modality": modality,
                    "media_type": MediaType.SYNTHETIC,
                    "status": "pending",
                    "sent_at": time.time(),
                    "requested_resolution": requested_resolution,
                }
            self.content_manager.record_challenge_outcome(
                task_id=miner_task_id,
                uid=uid,
                hotkey=miner_hotkey,
                prompt_id=prompt_entry.id,
                modality=modality.value,
                status="pending",
                requested_resolution=requested_resolution,
            )
            bt.logging.info(
                f"Stored challenge task {miner_task_id} for UID {uid}. Total active tasks: {len(self.challenge_tasks)}"
            )
        else:
            error = response_data.get("error") if response_data else "Unknown error"
            miner_hotkey = self.metagraph.hotkeys[uid]
            self.content_manager.record_challenge_outcome(
                task_id=f"no-answer-{uid}-{uuid.uuid4()}",
                uid=uid,
                hotkey=miner_hotkey,
                prompt_id=prompt_entry.id,
                modality=modality.value,
                status="failed",
                failure_reason="no_answer",
                requested_resolution=requested_resolution,
            )
            bt.logging.error(
                f"Failed to send challenge to UID {uid}. Error: {error} "
                f"(recorded no_answer for {modality.value})"
            )

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

        # Miner-reported failure after task_id was issued (async gen failed). Must run before
        # the media-only content-type gate; otherwise application/json is rejected with 400.
        task_status_hdr = (request.headers.get("task-status") or "").lower()
        if task_status_hdr == "failed":
            if not task_id:
                bt.logging.error(
                    f"Failure callback missing task-id from {format_uid_info()} (IP: {client_ip})"
                )
                return Response(status_code=400, content="Missing task-id header")
            base_ct = content_type.split(";")[0].strip()
            if base_ct not in ("application/json", "application/octet-stream"):
                bt.logging.error(
                    f"Failure callback invalid content-type {content_type!r} from {format_uid_info()} "
                    f"(IP: {client_ip}); use application/json or application/octet-stream"
                )
                return Response(
                    status_code=400,
                    content="Failure callbacks require application/json or application/octet-stream",
                )
            binary_data = await request.body()
            if not binary_data:
                bt.logging.error(
                    f"Failure callback empty body for task {task_id} from {format_uid_info()} (IP: {client_ip})"
                )
                return Response(status_code=400, content="Empty binary payload")
            err_msg = request.headers.get("error-message") or "Unknown error"
            try:
                payload = json.loads(binary_data.decode("utf-8"))
                err_msg = payload.get("error_message") or err_msg
            except json.JSONDecodeError:
                bt.logging.warning(
                    f"Failure callback for task {task_id}: body is not JSON; using header error-message"
                )
            with self.challenge_lock:
                if task_id not in self.challenge_tasks:
                    bt.logging.debug(
                        f"Failure callback for unknown/stale task_id={task_id} (IP: {client_ip})"
                    )
                    return Response(status_code=200, content="Task not found in current session")
                generator_uid = self.challenge_tasks[task_id].get("uid")
                generator_hotkey = self.challenge_tasks[task_id].get("hotkey") or self.metagraph.hotkeys[generator_uid]
                del self.challenge_tasks[task_id]
            self.content_manager.update_challenge_outcome(
                task_id=task_id,
                status="failed",
                failure_reason=f"miner_reported_failure: {err_msg}",
            )
            bt.logging.warning(
                f"Generative task failed (miner-reported): task_id={task_id} uid={generator_uid} "
                f"hotkey={generator_hotkey[:16]}... error={err_msg!r} (IP: {client_ip})"
            )
            return Response(status_code=200, content="Failure recorded")

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

        # Atomically claim the task so retries cannot enqueue duplicate work.
        with self.challenge_lock:
            bt.logging.debug(f"Callback for task {task_id}: Current active tasks: {list(self.challenge_tasks.keys())}")
            if task_id not in self.challenge_tasks:
                bt.logging.debug(f"Received binary upload for unknown task_id: {task_id} from {format_uid_info()} (IP: {client_ip}), content_type: {content_type}, size: {len(binary_data)} bytes")
                return Response(status_code=200, content="Task not found in current session")
            if self.challenge_tasks[task_id].get("status") in ("receiving", "processing"):
                return Response(status_code=202, content="Callback already accepted")
            challenge_info = self.challenge_tasks[task_id].copy()
            generator_uid = challenge_info["uid"]
            self.challenge_tasks[task_id]["status"] = "receiving"

        auth_uid_msg = f" (auth UID: {uid})" if uid != generator_uid and uid != "unknown" else ""
        bt.logging.info(
            f"Received binary upload for task {task_id}, UID {generator_uid}{auth_uid_msg}, "
            f"type: {content_type}, size: {len(binary_data)} bytes (IP: {client_ip})"
        )

        # A 202 means the validator has durably accepted responsibility. Stage
        # and fsync the payload before acknowledging it. The database outcome
        # remains pending (reward-neutral) until real validation completes.
        staging_path = self._callback_staging_path(task_id)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._write_staged_callback,
                staging_path,
                binary_data,
                {"task_id": task_id, "content_type": content_type, "uid": generator_uid},
            )
        except Exception as e:
            with self.challenge_lock:
                if task_id in self.challenge_tasks:
                    self.challenge_tasks[task_id]["status"] = "pending"
            bt.logging.error(f"Failed staging callback for task {task_id}: {e}")
            return Response(status_code=503, content="Validator could not accept callback")

        with self.challenge_lock:
            if task_id not in self.challenge_tasks:
                self._remove_staged_callback(staging_path)
                return Response(status_code=200, content="Task already processed")
            self.challenge_tasks[task_id].update({
                "status": "processing",
                "staging_path": staging_path,
                "content_type": content_type,
                "accepted_at": time.time(),
            })
            miner_hotkey = self.challenge_tasks[task_id]["hotkey"]
            self.generator_last_seen[miner_hotkey] = time.time()

        self.media_executor.submit(
            self._process_staged_callback,
            task_id,
            staging_path,
            content_type,
            generator_uid,
        )
        return Response(status_code=202, content="Callback accepted")


    def _callback_staging_path(self, task_id: str) -> str:
        name = hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".payload"
        return os.path.join(self.callback_staging_dir, name)

    @staticmethod
    def _write_staged_callback(
        staging_path: str, binary_data: bytes, metadata: dict
    ) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix="callback-", dir=os.path.dirname(staging_path)
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(binary_data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, staging_path)
            metadata_path = staging_path + ".json"
            metadata_tmp = metadata_path + ".tmp"
            with open(metadata_tmp, "w", encoding="utf-8") as f:
                json.dump(metadata, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(metadata_tmp, metadata_path)
        except Exception:
            for path in (tmp_path, staging_path, staging_path + ".json.tmp"):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            raise

    @staticmethod
    def _remove_staged_callback(staging_path: str) -> None:
        for path in (staging_path, staging_path + ".json"):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _process_staged_callback(
        self, task_id: str, staging_path: str, content_type: str, generator_uid: int
    ) -> None:
        try:
            with open(staging_path, "rb") as f:
                binary_data = f.read()
            filepath, error_message = self.store_binary_content(
                binary_data, content_type, generator_uid, task_id
            )
        except Exception as e:
            bt.logging.exception(
                f"Infrastructure failure processing callback {task_id}: {e}"
            )
            with self.challenge_lock:
                if task_id in self.challenge_tasks:
                    self.challenge_tasks[task_id]["status"] = "processing_error"
            return

        failure_reason = error_message or "Failed to store binary content"
        infrastructure_failure = bool(
            error_message
            and (
                error_message.startswith("Internal error:")
                or error_message == "Failed to store binary content"
            )
        )
        if infrastructure_failure:
            with self.challenge_lock:
                if task_id in self.challenge_tasks:
                    self.challenge_tasks[task_id]["status"] = "processing_error"
                    self.challenge_tasks[task_id]["processing_error"] = failure_reason
            return

        self._remove_staged_callback(staging_path)
        with self.challenge_lock:
            if task_id not in self.challenge_tasks:
                bt.logging.warning(f"Completed callback work for missing task {task_id}")
                return
            if filepath:
                self.challenge_tasks[task_id]["status"] = "completed"
                self.challenge_tasks[task_id]["filepath"] = filepath
                bt.logging.success(
                    f"Task {task_id} completed with binary upload: {filepath}"
                )
                miner_hotkey = self.metagraph.hotkeys[generator_uid]
                self.generator_last_seen[miner_hotkey] = time.time()
                del self.challenge_tasks[task_id]
                return
            del self.challenge_tasks[task_id]

        # Only explicit content rejection is a miner failure. Queue, disk, and
        # worker failures remain pending and therefore reward-neutral.
        self.content_manager.update_challenge_outcome(
            task_id=task_id, status="failed", failure_reason=failure_reason
        )

    def _recover_staged_callbacks(self) -> int:
        recovered = 0
        for metadata_path in os.scandir(self.callback_staging_dir):
            if not metadata_path.name.endswith(".payload.json"):
                continue
            staging_path = metadata_path.path[:-5]
            try:
                with open(metadata_path.path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                task_id = metadata["task_id"]
                content_type = metadata["content_type"]
                generator_uid = int(metadata["uid"])
            except Exception as e:
                bt.logging.error(f"Invalid staged callback metadata {metadata_path.path}: {e}")
                continue
            with self.challenge_lock:
                task = self.challenge_tasks.get(task_id)
                if task is None or not os.path.exists(staging_path):
                    continue
                task.update({
                    "status": "processing",
                    "staging_path": staging_path,
                    "content_type": content_type,
                })
            self.media_executor.submit(
                self._process_staged_callback,
                task_id,
                staging_path,
                content_type,
                generator_uid,
            )
            recovered += 1
        return recovered

    def store_binary_content(
        self, binary_data: bytes, content_type: str, generator_uid: int, task_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Store binary content directly uploaded by miner using ContentManager.

        Performs pre-storage validation and REJECTS:
        - Duplicate content (perceptual hash match)
        - Content without valid C2PA from trusted AI generators
        - Corrupted/unreadable media

        Only content that passes all checks is stored and eligible for HuggingFace upload.

        Returns:
            Tuple of (filepath, error_message). On success, filepath is set and error is None.
            On failure, filepath is None and error contains the rejection reason.
        """
        try:
            bt.logging.trace(f"Storing binary content for task {task_id}, size: {len(binary_data)} bytes")

            # Helper to optionally persist rejected media for manual inspection
            store_failed = getattr(self.config, 'store_failed_media', False)

            def _reject(reason: str) -> tuple[None, str]:
                if store_failed:
                    self.content_manager.write_failed_media(
                        uid=generator_uid, task_id=task_id,
                        binary_data=binary_data, reason=reason,
                        content_type=content_type,
                    )
                return None, reason

            # Get task info from challenge tracker
            task_info = self.challenge_tasks.get(task_id)
            if not task_info:
                bt.logging.error(f"Task {task_id} not found in challenge tasks")
                return None, "Task not found in challenge tasks"

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
                    return _reject("Corrupted or unreadable media")
            except Exception as e:
                bt.logging.warning(
                    f"REJECTED media from UID {generator_uid} task {task_id}: "
                    f"corruption check error: {e}"
                )
                return _reject(f"Corruption check error: {e}")

            # Step 2: Compute perceptual hash and check prompt/global duplicates
            perceptual_hash = None
            try:
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
                        return _reject("Duplicate content detected")

                    global_duplicate_info = self.content_manager.check_duplicate(
                        perceptual_hash,
                        threshold=getattr(getattr(self.config, 'verification', None), 'global_dup_hamming_threshold', 4),
                        prompt_id=None,
                    )
                    if global_duplicate_info:
                        dup_media_id, dup_distance = global_duplicate_info
                        bt.logging.warning(
                            f"REJECTED global duplicate from UID {generator_uid} task {task_id}: "
                            f"matches media {dup_media_id} with distance {dup_distance}"
                        )
                        return _reject("Global duplicate content detected")
            except Exception as e:
                bt.logging.warning(
                    f"REJECTED media from UID {generator_uid} task {task_id}: "
                    f"duplicate detection error: {e}"
                )
                return _reject(f"Duplicate detection error: {e}")

            # Step 3: Verify C2PA content credentials - REQUIRE trusted issuer
            c2pa_verified = False
            c2pa_issuer = None
            c2pa_model_name = None
            try:
                c2pa_result = verify_c2pa(binary_data)
                if c2pa_result.verified and c2pa_result.is_trusted_issuer:
                    c2pa_verified = True
                    c2pa_issuer = c2pa_result.issuer
                    c2pa_model_name = c2pa_result.model_name
                    bt.logging.info(
                        f"C2PA verified for UID {generator_uid}: issuer={c2pa_issuer}, model_name={c2pa_model_name}"
                    )
                else:
                    rejection_reason = c2pa_result.error or (
                        "untrusted issuer" if c2pa_result.signature_valid else "no valid C2PA manifest"
                    )
                    bt.logging.warning(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"C2PA check failed ({rejection_reason})"
                    )
                    return _reject(f"C2PA verification failed: {rejection_reason}")
            except Exception as e:
                bt.logging.error(
                    f"REJECTED from UID {generator_uid} task {task_id}: "
                    f"C2PA verification error: {e}"
                )
                return _reject(f"C2PA verification error: {e}")

            # Step 3b: Freshness - the manifest must have been signed after
            # this challenge was issued. This proves the media was generated
            # for this challenge, closing off replay of hoarded or scraped
            # C2PA-signed content regardless of how far back dedup can see.
            sent_at = task_info.get("sent_at")
            if sent_at:
                signing_time = c2pa_result.signing_time
                if signing_time is None:
                    bt.logging.warning(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"C2PA manifest has no signing timestamp (issuer: {c2pa_issuer})"
                    )
                    return _reject("C2PA manifest has no signing timestamp")
                signed_ts = signing_time.timestamp()
                now = time.time()
                if signed_ts < sent_at - C2PA_MAX_CLOCK_SKEW_SECONDS:
                    age_min = (sent_at - signed_ts) / 60
                    bt.logging.warning(
                        f"REJECTED stale media from UID {generator_uid} task {task_id}: "
                        f"C2PA signed {age_min:.1f} min before challenge was issued "
                        f"(signed {signing_time.isoformat()}, issuer: {c2pa_issuer})"
                    )
                    return _reject("C2PA manifest signed before challenge was issued")
                if signed_ts > now + C2PA_MAX_CLOCK_SKEW_SECONDS:
                    bt.logging.warning(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"C2PA signing timestamp is in the future "
                        f"(signed {signing_time.isoformat()}, issuer: {c2pa_issuer})"
                    )
                    return _reject("C2PA signing timestamp is in the future")

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
                model_name=c2pa_model_name,
                perceptual_hash=perceptual_hash,
                c2pa_verified=c2pa_verified,
                c2pa_issuer=c2pa_issuer,
            )

            if filepath:
                bt.logging.success(
                    f"Stored verified content: {filepath} (size: {len(binary_data)} bytes, "
                    f"hash: {perceptual_hash[:16] if perceptual_hash else 'N/A'}..., "
                    f"c2pa_issuer: {c2pa_issuer}, model_name: {c2pa_model_name})"
                )
                return filepath, None
            else:
                bt.logging.error("ContentManager failed to store binary content")
                return None, "Failed to store binary content"

        except Exception as e:
            bt.logging.error(f"Error storing binary content: {e}")
            return None, f"Internal error: {e}"

    def _check_media_corrupted(self, binary_data: bytes, modality: str) -> bool:
        """
        Check if media data is corrupted/unreadable.

        Returns:
            True if corrupted, False if valid
        """
        try:
            if modality == "image":
                img = Image.open(io.BytesIO(binary_data))
                img.verify()  # Verify it's a valid image
                return False
            elif modality == "video":
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

        # httptools streams the body — handler is invoked on headers, so now is
        # captured before the upload
        verifier = get_verifier(
            self.wallet, self.metagraph, no_force_validator_permit=True,
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
            http="httptools",
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

    def get_generator_last_seen(self, hotkey: str) -> Optional[float]:
        """Get the last activity timestamp for a generator.

        Args:
            hotkey: The generator's hotkey (SS58 address)

        Returns:
            Unix timestamp of last activity, or None if never seen
        """
        return self.generator_last_seen.get(hotkey)

    def get_all_generator_last_seen(self) -> Dict[str, float]:
        """Get all generator last seen timestamps.

        Returns:
            Dict mapping hotkey to last activity timestamp
        """
        return self.generator_last_seen.copy()

    def is_generator_active(self, hotkey: str, max_inactive_hours: int = 24) -> bool:
        """Check if a generator has been active within the specified window.

        Args:
            hotkey: The generator's hotkey (SS58 address)
            max_inactive_hours: Maximum hours of inactivity before considered inactive

        Returns:
            True if generator was active within the window, False otherwise
        """
        last_seen = self.generator_last_seen.get(hotkey)
        if last_seen is None:
            return False

        hours_since_seen = (time.time() - last_seen) / 3600
        return hours_since_seen <= max_inactive_hours

    def get_active_generators(self, max_inactive_hours: int = 24) -> Dict[str, float]:
        """Get all generators that have been active within the specified window.

        Args:
            max_inactive_hours: Maximum hours of inactivity before considered inactive

        Returns:
            Dict mapping hotkey to last activity timestamp for active generators
        """
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600

        return {
            hotkey: last_seen
            for hotkey, last_seen in self.generator_last_seen.items()
            if (current_time - last_seen) <= max_inactive_seconds
        }

    async def shutdown(self):
        """Shutdown the webhook server gracefully"""
        if hasattr(self, "fast_api"):
            bt.logging.info("Shutting down webhook server...")
            self.fast_api.stop()
            bt.logging.info("Webhook server stopped")
        if hasattr(self, "media_executor"):
            self.media_executor.shutdown(wait=False, cancel_futures=True)

    def save_state(self, save_dir: str, filename: str):
        """Save challenge tasks state and generator liveness to disk"""
        try:
            with self.challenge_lock:
                bt.logging.info(f"Saving challenge tasks state: {len(self.challenge_tasks)} active tasks")
                current_time = time.time()
                stale_tasks = []
                for task_id, task_info in self.challenge_tasks.items():
                    if (
                        task_info.get("status", "pending") == "pending"
                        and current_time - task_info.get("sent_at", 0) > 7200
                    ):
                        stale_tasks.append(task_id)

                for task_id in stale_tasks:
                    del self.challenge_tasks[task_id]
                    bt.logging.debug(f"Removed stale task {task_id} during state save")

                tasks_to_save = self.challenge_tasks.copy()

            # Release lock before making blocking DB calls
            for task_id in stale_tasks:
                self.content_manager.update_challenge_outcome(
                    task_id=task_id,
                    status="failed",
                    failure_reason="challenge_timeout",
                )

            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(tasks_to_save, f)
            bt.logging.info(f"Successfully saved {len(tasks_to_save)} active challenge tasks to {filepath}")

            liveness_filepath = os.path.join(save_dir, "generator_liveness.pkl")
            max_liveness_age = 7 * 24 * 3600
            valid_liveness = {
                hotkey: ts
                for hotkey, ts in self.generator_last_seen.items()
                if (current_time - ts) <= max_liveness_age
            }
            with open(liveness_filepath, 'wb') as f:
                pickle.dump(valid_liveness, f)
            bt.logging.info(f"Successfully saved liveness data for {len(valid_liveness)} generators")
        except Exception as e:
            bt.logging.error(f"Failed to save challenge tasks state: {e}")

    def load_state(self, save_dir: str, filename: str):
        """Load challenge tasks state and generator liveness from disk"""
        try:
            filepath = os.path.join(save_dir, filename)
            valid_tasks = {}
            if not os.path.exists(filepath):
                bt.logging.debug(f"No challenge tasks state file found at {filepath}")
            else:
                with open(filepath, 'rb') as f:
                    loaded_tasks = pickle.load(f)

                current_time = time.time()
                for task_id, task_info in loaded_tasks.items():
                    is_processing = task_info.get("status") != "pending"
                    if is_processing or current_time - task_info.get("sent_at", 0) <= 7200:
                        valid_tasks[task_id] = task_info

            valid_liveness = {}
            liveness_filepath = os.path.join(save_dir, "generator_liveness.pkl")
            if os.path.exists(liveness_filepath):
                with open(liveness_filepath, 'rb') as f:
                    loaded_liveness = pickle.load(f)

                current_time = time.time()
                max_liveness_age = 7 * 24 * 3600  # 7 days
                valid_liveness = {
                    hotkey: ts
                    for hotkey, ts in loaded_liveness.items()
                    if (current_time - ts) <= max_liveness_age
                }

            with self.challenge_lock:
                self.challenge_tasks = valid_tasks
                self.generator_last_seen = valid_liveness

            recovered = self._recover_staged_callbacks()
            bt.logging.info(f"Loaded {len(valid_tasks)} active challenge tasks from {filepath}")
            bt.logging.info(f"Recovered {recovered} durably staged callbacks")
            bt.logging.info(f"Loaded liveness data for {len(valid_liveness)} generators")

            return True
        except Exception as e:
            bt.logging.error(f"Failed to load challenge tasks state: {e}")
            with self.challenge_lock:
                self.challenge_tasks = {}
                self.generator_last_seen = {}
            return False
