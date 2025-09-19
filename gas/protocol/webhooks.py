import time
import threading
from typing import Optional, Dict, Any, TYPE_CHECKING

import requests
import bittensor as bt

from gas.protocol.epistula import generate_header

if TYPE_CHECKING:
    from neurons.generator.task_manager import GenerationTask


def send_success_webhook(
    task: "GenerationTask", 
    result: Dict[str, Any], 
    hotkey: bt.Keypair,
    external_ip: str,
    port: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0
):
    def _send():
        try:
            _send_webhook(task, result, True, hotkey, external_ip, port, max_retries, retry_delay, timeout)
        except Exception as e:
            bt.logging.error(f"Failed to send success webhook for task {task.task_id}: {e}")
    
    threading.Thread(target=_send, daemon=True).start()


def send_failure_webhook(
    task: "GenerationTask",
    hotkey: bt.Keypair, 
    external_ip: str,
    port: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0
):
    def _send():
        try:
            _send_webhook(task, {}, False, hotkey, external_ip, port, max_retries, retry_delay, timeout)
        except Exception as e:
            bt.logging.error(f"Failed to send failure webhook for task {task.task_id}: {e}")
    
    threading.Thread(target=_send, daemon=True).start()


def _send_webhook(
    task: "GenerationTask", 
    result: Dict[str, Any], 
    is_success: bool,
    hotkey: bt.Keypair,
    external_ip: str, 
    port: int,
    max_retries: int,
    retry_delay: float,
    timeout: float
):    
    for attempt in range(max_retries):
        try:
            success = _attempt_webhook_send(task, result, is_success, hotkey, timeout)
            if success:
                bt.logging.info(f"Webhook sent successfully for task {task.task_id}")
                return
            else:
                bt.logging.warning(f"Webhook attempt {attempt + 1} failed for task {task.task_id}")
                
        except Exception as e:
            bt.logging.error(f"Webhook attempt {attempt + 1} error for task {task.task_id}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay ** attempt)
    
    bt.logging.error(f"All webhook attempts failed for task {task.task_id}")


def _attempt_webhook_send(
    task: "GenerationTask",
    result: Dict[str, Any], 
    is_success: bool,
    hotkey: bt.Keypair,
    timeout: float
) -> bool:

    binary_data = b""
    try:
        if not is_success:
            content_type = "application/octet-stream"
            headers = {
                "Content-Type": content_type,
                "task-id": task.task_id,
                "task-status": "failed",
                "error-message": task.error_message or "Unknown error",
                **generate_header(hotkey, binary_data, task.signed_by),
            }
        else:
            binary_data = result.get("data")
            if not binary_data:
                bt.logging.error(f"Task {task.task_id} marked as successful but no binary data in result")
                return False

            bt.logging.debug(f"Sending webhook for task {task.task_id}: {len(binary_data)} bytes of {task.modality} data")

            if task.modality == "image":
                content_type = "image/png"
            elif task.modality == "video":
                content_type = "video/mp4"
            else:
                content_type = "application/octet-stream"

            headers = {
                "Content-Type": content_type,
                "task-id": task.task_id,
                "task-status": "completed",
                **generate_header(hotkey, binary_data, task.signed_by),
            }

            if result.get("metadata"):
                metadata = result["metadata"]
                for key, value in metadata.items():
                    header_key = f"x-meta-{key.replace('_', '-')}"
                    headers[header_key] = str(value)

        bt.logging.info("Sending webhook to:", task.webhook_url)
        response = requests.post(
            task.webhook_url, 
            data=binary_data, 
            headers=headers, 
            timeout=timeout
        )

        return response.status_code < 300

    except requests.RequestException as e:
        bt.logging.warning(f"Webhook request failed: {e}")
        return False
