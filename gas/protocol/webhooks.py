import json
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
    """Send success webhook notification asynchronously."""
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
    """Send failure webhook notification asynchronously."""
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
    """Send webhook with retry logic."""
    payload = _create_payload(task, result, is_success, external_ip, port)
    
    for attempt in range(max_retries):
        try:
            success = _attempt_webhook_send(task.webhook_url, payload, task.signed_by, hotkey, timeout)
            if success:
                bt.logging.info(f"Webhook sent successfully for task {task.task_id}")
                return
            else:
                bt.logging.warning(f"Webhook attempt {attempt + 1} failed for task {task.task_id}")
                
        except Exception as e:
            bt.logging.error(f"Webhook attempt {attempt + 1} error for task {task.task_id}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay ** attempt)  # Exponential backoff
    
    bt.logging.error(f"All webhook attempts failed for task {task.task_id}")


def _create_payload(
    task: "GenerationTask", 
    result: Dict[str, Any], 
    is_success: bool,
    external_ip: str,
    port: int
) -> Dict[str, Any]:
    """Create webhook payload."""
    payload = {
        "task_id": task.task_id,
        "status": task.status.value,
        "task_type": task.task_type,
        "modality": task.modality,
        "prompt": task.prompt,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
    }
    
    # Add processing time if available
    if task.started_at and task.completed_at:
        payload["processing_time"] = task.completed_at - task.started_at
    
    if is_success:
        # For successful tasks, include result information
        if result.get("url"):
            # If the service provided a direct URL (e.g., from OpenAI)
            payload["result_url"] = result["url"]
        elif task.result_data:
            # If we have binary data, provide a download endpoint
            payload["download_url"] = f"http://{external_ip}:{port}/download/{task.task_id}"
        
        # Include any additional result metadata
        if result.get("metadata"):
            payload["metadata"] = result["metadata"]
    else:
        # For failed tasks, include error information
        payload["error_message"] = task.error_message
    
    return payload


def _attempt_webhook_send(
    webhook_url: str, 
    payload: Dict[str, Any], 
    signed_by: str,
    hotkey: bt.Keypair,
    timeout: float
) -> bool:
    """Attempt to send a single webhook."""
    try:
        body_bytes = json.dumps(payload).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            **generate_header(hotkey, body_bytes, signed_by),
        }
        
        response = requests.post(
            webhook_url, 
            data=body_bytes, 
            headers=headers, 
            timeout=timeout
        )
        
        return response.status_code < 300
        
    except requests.RequestException as e:
        bt.logging.warning(f"Webhook request failed: {e}")
        return False
