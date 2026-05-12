import asyncio
import inspect
import json
import os
import requests
import shlex
import subprocess
import sys
import time
import traceback
import threading
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

import bittensor as bt
import uvicorn
from fastapi import FastAPI, APIRouter, Request, Response, Depends
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from dotenv import load_dotenv

from gas.protocol.epistula import get_verifier
from gas.types import ArtifactChainMetadata, ArtifactR2Location, NeuronType, MinerType
from gas.utils import print_info
from gas.utils.chain_artifact_metadata_store import ChainArtifactMetadataStore
from gas.verification import detect_media_format
from neurons.base import BaseNeuron
from neurons.generator.task_manager import TaskManager, GenerationTask, TaskStatus
from neurons.generator.task_checkpoint_store import TaskCheckpointStore, resolve_state_dir
from gas.protocol.webhooks import send_success_webhook, send_failure_webhook
from neurons.generator.services.service_registry import ServiceRegistry

try:
    load_dotenv(".env.gen_miner")
except Exception:
    pass


class GenerativeMiner(BaseNeuron):
    neuron_type = NeuronType.MINER

    def __init__(self, config=None):
        super().__init__(config=config)
        self.init()

    def init(self):
        max_task_age_hours = getattr(self.config.miner, "max_task_age_hours", 24)
        self.task_manager = TaskManager(max_task_age_hours=max_task_age_hours)
        self.service_registry = ServiceRegistry(self.config)
        self.miner_type = MinerType(
            getattr(self.config.miner, "type", MinerType.GENERATOR.value).upper()
        )
        self.artifact_metadata_store = ChainArtifactMetadataStore(
            self.subtensor,
            self.config.netuid,
        )

        self.external_ip = self.config.axon.external_ip or self.config.axon.ip
        if not self.external_ip:
            self.external_ip = requests.get(
                "https://checkip.amazonaws.com", timeout=10
            ).text.strip()

        output_dir = getattr(self.config.miner, "output_dir", "generated_content")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.checkpoint_store = TaskCheckpointStore(resolve_state_dir(self.output_dir))
        self._restore_checkpointed_tasks()

        self._start_background_workers()

        self.block_callbacks.append(self.log_on_block)

    def _start_background_workers(self):
        """Start background threads for task processing and cleanup."""
        # Start task processing worker
        self.task_processor = threading.Thread(
            target=self._task_processing_loop, daemon=True
        )
        self.task_processor.start()

        # Start cleanup worker
        self.cleanup_worker = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_worker.start()

    def _task_processing_loop(self):
        """Background loop to process pending tasks."""
        while True:
            try:
                pending_tasks = self.task_manager.get_pending_tasks()

                if not pending_tasks:
                    time.sleep(1)  # No tasks, wait briefly
                    continue

                # Check concurrent task limit
                max_concurrent = getattr(self.config.miner, "max_concurrent_tasks", 5)
                processing_count = sum(
                    1
                    for task in self.task_manager.tasks.values()
                    if task.status.value == "processing"
                )

                if processing_count >= max_concurrent:
                    time.sleep(2)  # At capacity, wait longer
                    continue

                # Process one task
                task = pending_tasks[0]
                try:
                    asyncio.run(self._process_task(task))
                except Exception as e:
                    bt.logging.error(f"Error processing task {task.task_id}: {e}")
                    self.task_manager.mark_task_failed(task.task_id, str(e))
                    self.checkpoint_store.delete(task.task_id)
                    failed_task = self.task_manager.get_task(task.task_id) or task
                    send_failure_webhook(
                        failed_task,
                        self.wallet.hotkey,
                        self.external_ip,
                        self.config.axon.port,
                        max_retries=self.config.miner.webhook_max_retries,
                        retry_delay=self.config.miner.webhook_retry_delay,
                        timeout=self.config.miner.webhook_timeout,
                    )

            except Exception as e:
                bt.logging.error(f"Error in task processing loop: {e}")
                time.sleep(5)

    def _persist_task_snapshot(self, task_id: str) -> None:
        t = self.task_manager.get_task(task_id)
        if t:
            self.checkpoint_store.save(t)

    def _checkpoint_callback(self, task_id: str):
        def cb(data):
            self.task_manager.set_checkpoint(task_id, data)
            self._persist_task_snapshot(task_id)

        return cb

    def _restore_checkpointed_tasks(self) -> None:
        """Reload queued / interrupted tasks from disk so restarts can resume long jobs."""
        restored_processing: list = []
        for task in self.checkpoint_store.load_all():
            if task.status == TaskStatus.COMPLETED:
                self.checkpoint_store.delete(task.task_id)
                continue
            if task.status == TaskStatus.FAILED:
                self.checkpoint_store.delete(task.task_id)
                continue
            if task.status == TaskStatus.PENDING:
                self.task_manager.restore_task(task)
                bt.logging.info(f"Restored pending task {task.task_id} from disk")
                continue
            if task.status == TaskStatus.PROCESSING:
                if task.checkpoint:
                    self.task_manager.restore_task(task)
                    restored_processing.append(task)
                    bt.logging.info(
                        f"Restored processing task {task.task_id} "
                        f"(checkpoint kind={task.checkpoint.get('kind')})"
                    )
                else:
                    bt.logging.warning(
                        f"Dropping processing task {task.task_id} without checkpoint; "
                        "cannot resume external job"
                    )
                    self.checkpoint_store.delete(task.task_id)

        for task in restored_processing:
            threading.Thread(
                target=lambda t=task: asyncio.run(self._process_task(t)),
                daemon=True,
            ).start()

    async def _process_task(self, task):
        """Process a single task using the appropriate service."""
        bt.logging.info(f"🎯 Processing task {task.task_id}: modality={task.modality}")

        # Do not reject tasks based on (now - created_at): queue wait behind long jobs
        # would incorrectly trip MINER_TASK_TIMEOUT. Stale tasks are removed by
        # max_task_age_hours / cleanup_old_tasks instead.

        resume = (
            task.status == TaskStatus.PROCESSING and task.checkpoint is not None
        )
        if resume:
            bt.logging.info(
                f"Resuming interrupted task {task.task_id} from external checkpoint"
            )
            task = self.task_manager.get_task(task.task_id) or task
        else:
            if task.status == TaskStatus.PENDING:
                self.task_manager.mark_task_processing(task.task_id)
                task = self.task_manager.get_task(task.task_id) or task
                self._persist_task_snapshot(task.task_id)

        try:
            # Get the appropriate service for this task
            service = self.service_registry.get_service(task.modality)
            bt.logging.info(f"Using service: {service.__class__.__name__ if service else 'None'} for {task.modality}")
            if not service:
                raise ValueError(f"No service available for modality={task.modality}")

            start_time = time.time()
            result = await self._call_service_process(service, task)
            processing_time = time.time() - start_time

            # mark task complete & send webhook response
            result_data = result.get("data")
            bt.logging.info(
                f"Task {task.task_id} result_data: {len(result_data) if result_data else 0} bytes, "
                f"type={type(result_data)}, magic={result_data[:16].hex() if result_data else 'N/A'}"
            )

            self._save_to_output_dir(task, result_data)

            self.task_manager.mark_task_completed(
                task.task_id, result_data, result.get("url")
            )
            self.checkpoint_store.delete(task.task_id)
            send_success_webhook(
                task,
                result,
                self.wallet.hotkey,
                self.external_ip,
                self.config.axon.port,
                max_retries=self.config.miner.webhook_max_retries,
                retry_delay=self.config.miner.webhook_retry_delay,
                timeout=self.config.miner.webhook_timeout,
            )
            bt.logging.success(
                f"✅ Task {task.task_id} completed in {processing_time:.2f}s"
            )

        except Exception as e:
            self.task_manager.mark_task_failed(task.task_id, str(e))
            self.checkpoint_store.delete(task.task_id)
            failed_task = self.task_manager.get_task(task.task_id) or task
            send_failure_webhook(
                failed_task,
                self.wallet.hotkey,
                self.external_ip,
                self.config.axon.port,
                max_retries=self.config.miner.webhook_max_retries,
                retry_delay=self.config.miner.webhook_retry_delay,
                timeout=self.config.miner.webhook_timeout,
            )
            bt.logging.error(f"❌ Task {task.task_id} failed: {e}")

    async def _call_service_process(self, service, task):
        """Call service.process(), handling both sync and async services."""
        try:
            bt.logging.debug(f"🔧 Calling service: {service.__class__.__name__}")
            cb = self._checkpoint_callback(task.task_id)
            if inspect.iscoroutinefunction(service.process):
                # Async image services complete in one round-trip; no external checkpoint.
                result = await service.process(task)
            else:
                result = service.process_with_checkpoint(task, cb)

            # Validate service result
            if result is None:
                bt.logging.error(
                    f"Service {service.__class__.__name__} returned None for task {task.task_id}"
                )
                raise ValueError(
                    f"Service {service.__class__.__name__} returned no result"
                )

            if not isinstance(result, dict):
                bt.logging.error(
                    f"Service {service.__class__.__name__} returned non-dict result: {type(result)}"
                )
                raise ValueError(
                    f"Service returned invalid result type: {type(result)}"
                )

            bt.logging.debug(f"📊 Service result keys: {list(result.keys())}")

            # Validate required result structure
            if "data" not in result:
                bt.logging.error(f"Service {service.__class__.__name__} result missing 'data' field")
                raise ValueError("Service result missing required 'data' field")

            result_data = result["data"]
            if result_data is None:
                bt.logging.error(f"Service {service.__class__.__name__} returned None for data field")
                raise ValueError("Service result data field is None")

            bt.logging.info(f"✅ Service {service.__class__.__name__} completed successfully")
            return result

        except Exception as e:
            bt.logging.error(f"Error calling service {service.__class__.__name__}: {e}")
            raise

    def _log_available_services(self):
        bt.logging.info("🔍 Checking service availability for modalities...")

        modalities = ["image", "video"]
        available_count = 0
        total_endpoints = len(modalities)

        bt.logging.info("=" * 60)
        bt.logging.info("📋 GENERATIVE MINER SERVICE STATUS")
        bt.logging.info("=" * 60)

        for modality in modalities:
            service = self.service_registry.get_service(modality)

            endpoint_map = {
                "image": "POST /gen_image",
                "video": "POST /gen_video",
            }
            endpoint = endpoint_map.get(modality, modality)

            if service:
                bt.logging.info(f"✅ {endpoint:<18} → {service.name}")
                available_count += 1
            else:
                bt.logging.info(f"❌ {endpoint:<18} → No service available")

        bt.logging.info("=" * 60)
        bt.logging.info(f"📊 Summary: {available_count}/{total_endpoints} endpoints functional")

        if available_count == 0:
            bt.logging.warning("⚠️  No services available! Miner will reject all requests.")
            bt.logging.info("💡 Check API keys in .env.gen_miner and restart")
        elif available_count < total_endpoints:
            bt.logging.warning(
                f"⚠️  Partial functionality: {total_endpoints - available_count} endpoints unavailable"
            )
        else:
            bt.logging.success("🎉 All endpoints fully functional!")

        bt.logging.info("=" * 60)

    def _cleanup_loop(self):
        """Background loop to cleanup old tasks."""
        cleanup_interval = getattr(self.config.miner, "cleanup_interval", 3600)
        while True:
            try:
                removed_ids = self.task_manager.cleanup_old_tasks()
                for tid in removed_ids:
                    self.checkpoint_store.delete(tid)
                if removed_ids:
                    bt.logging.info(f"🧹 Cleaned up {len(removed_ids)} old tasks")
                time.sleep(cleanup_interval)
            except Exception as e:
                bt.logging.error(f"Error in cleanup loop: {e}")
                time.sleep(cleanup_interval)

    def _save_to_output_dir(self, task, data: bytes):
        """Save generated content and metadata to output directory."""
        if not data:
            return

        save_locally = os.getenv("MINER_SAVE_LOCALLY", "false").lower() in ("true", "1", "yes")
        if not save_locally:
            return

        ext = self._detect_extension(data, task.modality)
        modality_dir = task.modality if task.modality in ("image", "video") else "other"
        save_dir = self.output_dir / modality_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        media_path = save_dir / f"{task.task_id}.{ext}"
        meta_path = save_dir / f"{task.task_id}.json"

        try:
            media_path.write_bytes(data)
            bt.logging.info(f"💾 Saved {task.modality} to {media_path}")
        except Exception as e:
            bt.logging.warning(f"Failed to save {task.modality} to {media_path}: {e}")

        try:
            import json
            metadata = {
                "task_id": task.task_id,
                "modality": task.modality,
                "prompt": task.prompt,
                "parameters": task.parameters,
                "signed_by": task.signed_by,
                "created_at": task.created_at,
                "completed_at": time.time(),
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            bt.logging.warning(f"Failed to save metadata to {meta_path}: {e}")

    def _detect_extension(self, data: bytes, modality: str) -> str:
        """Detect file extension from binary data magic bytes."""
        ext = detect_media_format(data)
        if ext != ".bin":
            return ext.lstrip(".")
        return "png" if modality == "image" else "mp4"

    # API Endpoints
    async def _handle_generation_request(self, request: Request, modality: str):
        """Handle generation requests for both image and video."""
        try:
            bt.logging.info(f"Received {modality} generation request")

            # Parse request body
            try:
                body = await request.json()
                bt.logging.debug(f"Request body keys: {list(body.keys()) if body else 'None'}")
            except Exception as e:
                bt.logging.error(f"Failed to parse JSON body: {e}")
                return self._error_response(
                    f"Invalid JSON body: {str(e)}", 400
                )

            # Extract required fields
            prompt = body.get("prompt")
            if not prompt:
                bt.logging.error(f"Missing prompt field. Body: {body}")
                return self._error_response("Missing required field: prompt", 400)

            webhook_url = request.headers.get("X-Webhook-URL")
            if not webhook_url:
                bt.logging.error(f"Missing X-Webhook-URL header. Headers: {dict(request.headers)}")
                return self._error_response(
                    "Missing required header: X-Webhook-URL", 400
                )

            signed_by = request.headers.get("Epistula-Signed-By", "unknown")
            parameters = body.get("parameters", {}) or {}

            service = self.service_registry.get_service(modality)
            if not service:
                bt.logging.warning(f"Rejecting {modality} request: no service available")
                return self._error_response(
                    f"No service available for modality={modality}", 503
                )

            task_id = self.task_manager.create_task(
                modality=modality,
                prompt=prompt,
                parameters=parameters,
                webhook_url=webhook_url,
                signed_by=signed_by,
                input_data=None,  # placeholder for modifications
            )

            bt.logging.info(f"✅ Created {modality} task {task_id}: {prompt[:50]}...")

            created_task = self.task_manager.get_task(task_id)
            if created_task:
                self.checkpoint_store.save(created_task)

            return self._success_response(
                {
                    "task_id": task_id,
                    "status": "submitted",
                    "modality": modality,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "webhook_url": webhook_url,
                    "created_at": (
                        created_task.created_at if created_task else time.time()
                    ),
                },
                201,
            )

        except Exception as e:
            bt.logging.error(f"Error handling {modality} request: {e}")
            return self._error_response(
                f"Failed to process {modality} request: {str(e)}", 500
            )

    async def generate_image(self, request: Request):
        """Submit image generation task."""
        return await self._handle_generation_request(request, "image")

    async def generate_video(self, request: Request):
        """Submit video generation task."""
        return await self._handle_generation_request(request, "video")

    async def get_task_status(self, request: Request):
        """Get status of a specific task."""
        try:
            path_parts = request.url.path.split("/")
            if len(path_parts) < 3:
                return self._error_response("Invalid status URL", 400)

            task_id = path_parts[2]
            task = self.task_manager.get_task(task_id)

            if not task:
                return self._error_response("Task not found", 404)

            response_data = task.to_dict()
            return self._success_response(response_data)

        except Exception as e:
            bt.logging.error(f"Error getting task status: {e}")
            return self._error_response(str(e), 500)

    async def get_miner_info(self, request: Request):
        """Return comprehensive miner information."""
        services = self.service_registry.get_available_services()
        task_stats = self.task_manager.get_task_stats()

        return self._success_response(
            {
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "miner_type": self.miner_type.value,
                "services": services,
                "task_stats": task_stats,
                "config": {
                    "max_concurrent_tasks": getattr(
                        self.config.miner, "max_concurrent_tasks", 5
                    ),
                    "task_timeout": getattr(self.config.miner, "task_timeout", 300),
                    "max_task_age_hours": getattr(
                        self.config.miner, "max_task_age_hours", 24
                    ),
                    "cleanup_interval": getattr(
                        self.config.miner, "cleanup_interval", 3600
                    ),
                },
                "version": "1.0.0",
                "uptime": time.time() - getattr(self, "_start_time", time.time()),
            }
        )

    async def artifact_task(self, request: Request):
        """Accept a DPS encoder/captioner artifact assignment."""
        try:
            body = await request.json()
            task_id = body.get("task_id")
            role = MinerType((body.get("role") or "").upper())
            source = body.get("source") or {}
            signed_by = request.headers.get("Epistula-Signed-By")

            if role != self.miner_type:
                return self._error_response(
                    f"Miner role {self.miner_type.value} cannot accept {role.value} tasks",
                    409,
                )
            if not task_id:
                return self._error_response("Missing required field: task_id", 400)
            if source.get("type") != "r2":
                source = await self._source_from_validator_commitment(signed_by, role)
            if source.get("type") != "r2":
                return self._error_response("Artifact source must be type=r2", 400)

            processor_result = self._run_artifact_processor(task_id, role, source)
            output_metadata = self._build_output_metadata(
                task_id=task_id,
                role=role,
                processor_result=processor_result,
            )
            await self.artifact_metadata_store.store_artifact_metadata(
                self.wallet,
                output_metadata,
            )
            bt.logging.info(
                f"Published DPS {role.value.lower()} output metadata for task {task_id}"
            )
            return self._success_response(
                {
                    "accepted": True,
                    "task_id": task_id,
                    "role": role.value,
                    "output": output_metadata.r2.to_dict(),
                },
                202,
            )
        except ValueError as e:
            return self._error_response(str(e), 400)
        except Exception as e:
            bt.logging.error(f"Error handling artifact task: {e}")
            bt.logging.error(traceback.format_exc())
            return self._error_response(str(e), 500)

    def _run_artifact_processor(self, task_id: str, role: MinerType, source: Dict[str, Any]):
        command = getattr(self.config.dps_artifact, "processor_command", None)
        if not command:
            bt.logging.warning(
                "No DPS artifact processor command configured; publishing output metadata only"
            )
            return {}

        env = os.environ.copy()
        env.update(
            {
                "DPS_TASK_ID": task_id,
                "DPS_ROLE": role.value,
                "DPS_SOURCE_BUCKET": source.get("bucket", ""),
                "DPS_SOURCE_PATH": source.get("path") or source.get("prefix", ""),
                "DPS_SOURCE_MANIFEST_URL": source.get("manifest_url", ""),
                "DPS_SOURCE_MANIFEST_KEY": source.get("manifest_key", ""),
                "DPS_SOURCE_ENDPOINT_URL": source.get("endpoint_url", ""),
                "DPS_SOURCE_ACCESS_KEY_ID": source.get("access_key_id", ""),
                "DPS_SOURCE_SECRET_ACCESS_KEY": source.get("secret_access_key", ""),
                "DPS_OUTPUT_BUCKET": getattr(self.config.dps_artifact, "output_r2_bucket", "") or "",
                "DPS_OUTPUT_PREFIX": self._output_prefix(task_id),
                "DPS_OUTPUT_ENDPOINT_URL": getattr(
                    self.config.dps_artifact, "output_r2_endpoint_url", ""
                ) or "",
            }
        )
        result = subprocess.run(
            shlex.split(command),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Artifact processor failed with code {result.returncode}: {result.stderr}"
            )
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    async def _source_from_validator_commitment(
        self,
        validator_hotkey: Optional[str],
        role: MinerType,
    ):
        if not validator_hotkey or validator_hotkey not in self.metagraph.hotkeys:
            return {}
        validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
        metadata = await self.artifact_metadata_store.retrieve_artifact_metadata(
            uid=validator_uid,
            expected_kind="dps_input",
            role=role,
        )
        if metadata is None:
            metadata = await self.artifact_metadata_store.retrieve_artifact_metadata(
                uid=validator_uid,
                expected_kind="dps_input",
            )
        if metadata is None:
            return {}
        source = metadata.r2.to_dict()
        source["type"] = "r2"
        if "path" in source and "prefix" not in source:
            source["prefix"] = source["path"]
        return source

    def _build_output_metadata(
        self,
        task_id: str,
        role: MinerType,
        processor_result: Optional[Dict[str, Any]] = None,
    ):
        artifact_config = self.config.dps_artifact
        processor_result = processor_result or {}
        bucket = getattr(artifact_config, "output_r2_bucket", None)
        if not bucket:
            raise ValueError("--dps-artifact.output-r2-bucket is required")

        return ArtifactChainMetadata(
            kind="dps_output",
            role=role,
            task_id=task_id,
            artifact_format=getattr(artifact_config, "output_format", "npz"),
            artifact_hash=processor_result.get("artifact_hash"),
            r2=ArtifactR2Location(
                bucket=bucket,
                path=processor_result.get("path") or self._output_prefix(task_id),
                endpoint_url=getattr(artifact_config, "output_r2_endpoint_url", None),
                access_key_id=getattr(
                    artifact_config, "output_r2_read_access_key_id", None
                ),
                secret_access_key=getattr(
                    artifact_config, "output_r2_read_secret_access_key", None
                ),
                session_token=getattr(
                    artifact_config, "output_r2_read_session_token", None
                ),
                manifest_url=(
                    processor_result.get("manifest_url")
                    or getattr(artifact_config, "output_r2_manifest_url", None)
                ),
                manifest_key=(
                    processor_result.get("manifest_key")
                    or getattr(artifact_config, "output_r2_manifest_key", None)
                ),
            ),
        )

    def _output_prefix(self, task_id: str):
        base = getattr(self.config.dps_artifact, "output_r2_prefix", "dps-artifacts/")
        return f"{base.rstrip('/')}/{task_id}/"

    async def get_health_check(self, request: Request):
        """Health check endpoint."""
        try:
            services = self.service_registry.get_available_services()
            available_count = sum(1 for s in services if s["available"])

            if available_count == 0:
                return self._error_response("No services available", 503)

            return self._success_response(
                {
                    "status": "healthy",
                    "services_available": available_count,
                    "services_total": len(services),
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return self._error_response(f"Health check failed: {e}", 503)

    def _success_response(self, data: Dict[str, Any], status_code: int = 200):
        """Helper to create success responses."""
        import json

        return Response(
            content=json.dumps(data),
            media_type="application/json",
            status_code=status_code,
        )

    def _error_response(self, message: str, status_code: int = 400):
        """Helper to create error responses."""
        import json

        return Response(
            content=json.dumps({"error": message, "timestamp": time.time()}),
            media_type="application/json",
            status_code=status_code,
        )

    async def log_on_block(self, block):
        """Log information on each block."""
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

    def shutdown(self):
        """Shutdown the miner."""
        bt.logging.info("🛑 Shutting down GenerativeMiner...")
        if hasattr(self, "fast_api"):
            self.fast_api.stop()

    def run(self):
        """Main run method - sets up FastAPI server and starts the miner."""
        self._start_time = time.time()

        success = self._serve_on_network()
        if not success:
            bt.logging.error("❌ Failed to serve miner on network")
            return

        app = self._create_fastapi_app()
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
            http="httptools",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.success(f"Generative miner started (UID: {self.uid})")
        self._log_available_services()

        # Main loop
        try:
            while not self.exit_context.isExiting:
                time.sleep(1)
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.shutdown()

    def _serve_on_network(self) -> bool:
        """Register the miner on the Bittensor network."""
        try:
            bt.logging.info(
                f"🌐 Serving axon on {self.external_ip}:{self.config.axon.port}"
            )
            bt.logging.info(f"🔗 Network: {self.config.subtensor.chain_endpoint}")
            bt.logging.info(f"🆔 Netuid: {self.config.netuid}")
            return serve_extrinsic(
                subtensor=self.subtensor,
                wallet=self.wallet,
                ip=self.external_ip,
                port=self.config.axon.port,
                protocol=4,
                netuid=self.config.netuid,
                wait_for_finalization=True,
            )
        except Exception as e:
            bt.logging.error(f"Error serving on network: {e}")
            return False

    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="GenerativeMiner",
            description="Production-ready generative miner for Bitmind subnet",
            version="1.0.0",
        )
        router = APIRouter()
        verifier = get_verifier(
            self.wallet,
            self.metagraph,
            getattr(self.config, "no_force_validator_permit", False),
        )
        router.add_api_route(
            "/health",
            self.get_health_check,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )
        router.add_api_route(
            "/miner_info",
            self.get_miner_info,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )
        router.add_api_route(
            "/status/{task_id}",
            self.get_task_status,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )

        # endpoints called by validators
        router.add_api_route(
            "/gen_image",
            self.generate_image,
            dependencies=[Depends(verifier)],
            methods=["POST"],
        )
        router.add_api_route(
            "/gen_video",
            self.generate_video,
            dependencies=[Depends(verifier)],
            methods=["POST"],
        )
        router.add_api_route(
            "/artifact_task",
            self.artifact_task,
            dependencies=[Depends(verifier)],
            methods=["POST"],
        )

        app.include_router(router)
        return app


if __name__ == "__main__":
    try:
        miner = GenerativeMiner()
        miner.run()
    except KeyboardInterrupt:
        bt.logging.info("Miner interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
