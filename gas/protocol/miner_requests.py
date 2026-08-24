import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import BinaryIO, Dict, Optional

import bittensor as bt
import httpx
import requests

from gas.protocol.epistula import generate_header

GAS_API_BASE_URL = "https://gas.bitmind.ai"


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def calculate_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class UploadProgressReader:
    """File-like request body that reports upload progress and throughput."""

    def __init__(self, file_path: Path, report_interval: float = 0.5):
        self._file: BinaryIO = file_path.open("rb")
        self._total = file_path.stat().st_size
        self._uploaded = 0
        self._started_at = time.monotonic()
        self._last_report = 0.0
        self._report_interval = report_interval

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        self._uploaded += len(chunk)
        now = time.monotonic()
        if (
            self._uploaded == self._total
            or now - self._last_report >= self._report_interval
        ):
            self._report(now)
        return chunk

    def tell(self) -> int:
        return self._file.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        position = self._file.seek(offset, whence)
        self._uploaded = position
        return position

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "UploadProgressReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _report(self, now: float) -> None:
        elapsed = max(now - self._started_at, 1e-9)
        rate = self._uploaded / elapsed
        percent = 100.0 if not self._total else self._uploaded / self._total * 100
        remaining = max(self._total - self._uploaded, 0)
        eta = remaining / rate if rate else 0.0
        print(
            f"\r      {percent:5.1f}% "
            f"({self._uploaded / 1024 / 1024:.1f}/{self._total / 1024 / 1024:.1f} MB) "
            f"{rate / 1024 / 1024:.1f} MB/s ETA {eta:.0f}s",
            end="",
            flush=True,
        )
        self._last_report = now


def generate_presigned_url(
    wallet: bt.Wallet, 
    upload_endpoint: str, 
    filename: str, 
    file_size: int, 
    file_hash: str, 
    content_type: Optional[str] = None,
    modality: Optional[str] = None,
    vertical: Optional[str] = None
) -> dict:
    """Generate presigned upload URL from the API with modality and vertical parameters."""
    
    payload = {
        'filename': filename,
        'file_size': file_size,
        'expected_hash': file_hash,
    }
    if content_type:
        payload['content_type'] = content_type
    if modality:
        payload['modality'] = modality
    if vertical:
        payload['vertical'] = vertical
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    try:
        presigned_endpoint = upload_endpoint.rstrip('/') + '/presigned'
        response = requests.post(
            presigned_endpoint,
            data=payload_bytes,
            headers=headers,
            timeout=30
        )

        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "text": response.text}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": result
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Request failed: {str(e)}"}
        }


def upload_to_r2(
    presigned_url: str,
    file_path: Path,
    content_type: str = 'application/octet-stream',
) -> dict:
    """Stream a file directly to R2 using a presigned URL."""
    try:
        with UploadProgressReader(file_path) as upload:
            response = requests.put(
                presigned_url,
                data=upload,
                headers={
                    'Content-Type': content_type,
                    'Content-Length': str(len(upload)),
                },
                timeout=300,  # 5 minutes for large files
            )
        print()
        
        error_detail = None
        if response.status_code != 200:
            text = response.text or ""
            match = re.search(r'<Message>(.*?)</Message>', text) or \
                    re.search(r'<Code>(.*?)</Code>', text)
            error_detail = match.group(1) if match else (text[:200] or "Upload failed")

        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": {
                "message": "Upload successful" if response.status_code == 200 else error_detail,
                "etag": response.headers.get('ETag', ''),
            }
        }
        
    except requests.exceptions.RequestException as e:
        print()
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Upload failed: {str(e)}"}
        }


def confirm_upload(wallet: bt.Wallet, upload_endpoint: str, model_id: int, file_hash: str) -> dict:
    """Confirm file upload and finalize model record."""
    
    payload = {
        'model_id': model_id,
        'file_hash': file_hash
    }
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    try:
        confirm_endpoint = upload_endpoint.rstrip('/') + '/confirm'
        response = requests.post(
            confirm_endpoint,
            data=payload_bytes,
            headers=headers,
            timeout=30
        )
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "text": response.text}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": result
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Request failed: {str(e)}"}
        }


def upload_single_modality(
    wallet: bt.Wallet,
    file_path: str,
    modality: str,
    upload_endpoint: str,
    vertical: str = "general"
) -> dict:
    """Upload a single modality file (image or video model)."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_hash = calculate_file_sha256(file_path_obj)
    file_size = file_path_obj.stat().st_size
    filename = file_path_obj.name

    print(f"  File: {filename} ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Hash: {file_hash}")

    def extract_error(result: dict) -> str:
        resp = result.get('response', {})
        msg = resp.get('detail') or resp.get('error') or resp.get('message') or str(resp)
        status = result.get('status_code', 0)
        return f"HTTP {status}: {msg}" if status else str(msg)

    print(f"  [1/3] Requesting presigned URL...", end=' ', flush=True)
    presigned_result = generate_presigned_url(
        wallet,
        upload_endpoint,
        filename,
        file_size,
        file_hash,
        'application/octet-stream',
        modality,
        vertical
    )

    if not presigned_result['success']:
        print("FAILED")
        # 409 means this hash was already accepted — the file is already in R2.
        # Return a soft error so the caller can decide whether to skip or abort.
        if presigned_result.get('status_code') == 409:
            return {
                "success": False,
                "modality": modality,
                "step": "presigned_url_generation",
                "error": extract_error(presigned_result),
                "response": presigned_result['response'],
                "already_uploaded": True,
            }
        return {
            "success": False,
            "modality": modality,
            "step": "presigned_url_generation",
            "error": extract_error(presigned_result),
            "response": presigned_result['response']
        }
    print("done")

    presigned_data = presigned_result['response']['data']
    model_id = presigned_data['model_id']
    presigned_url = presigned_data['presigned_url']
    r2_key = presigned_data['r2_key']
    submissions_used = presigned_data.get('submissions_used')
    submissions_max = presigned_data.get('submissions_max')

    print(f"  [2/3] Uploading to R2...", end=' ', flush=True)
    upload_result = upload_to_r2(
        presigned_url,
        file_path_obj,
        'application/octet-stream',
    )

    if not upload_result['success']:
        print("FAILED")
        return {
            "success": False,
            "modality": modality,
            "step": "r2_upload",
            "model_id": model_id,
            "error": extract_error(upload_result),
            "response": upload_result['response']
        }
    print("done")

    print(f"  [3/3] Confirming upload...", end=' ', flush=True)
    confirm_result = confirm_upload(wallet, upload_endpoint, model_id, file_hash)

    if not confirm_result['success']:
        print("FAILED")
        return {
            "success": False,
            "modality": modality,
            "step": "upload_confirmation",
            "model_id": model_id,
            "error": extract_error(confirm_result),
            "response": confirm_result['response']
        }
    print("done")

    return {
        "success": True,
        "modality": modality,
        "model_id": model_id,
        "r2_key": r2_key,
        "file_hash": file_hash,
        "file_size": file_size,
        "submissions_used": submissions_used,
        "submissions_max": submissions_max,
    }


def fetch_performance(
    wallet: bt.Wallet,
    modality: Optional[str] = None,
    vertical: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict:
    """Query the miner's own benchmark performance via the GAS API.

    Returns a dict with 'success', 'runs' (list of dicts), and optionally 'error'.
    Each run dict contains: run_id, status, modality, vertical, sn34_score, mcc, brier.
    """
    base = api_url or os.environ.get("GAS_API_URL", GAS_API_BASE_URL)
    url = f"{base}/api/v1/miner/performance"

    params: Dict[str, str] = {}
    if modality:
        params["modality"] = modality
    if vertical:
        params["vertical"] = vertical

    headers = generate_header(wallet.hotkey, b"")

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
    except httpx.RequestError as e:
        return {"success": False, "runs": [], "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {
            "success": False,
            "runs": [],
            "error": f"API error {resp.status_code}: {resp.text}",
        }

    return {"success": True, "runs": resp.json()}


def fetch_models(
    wallet: bt.Wallet,
    modality: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict:
    """Query the miner's own model submissions via the GAS API.

    Returns a dict with 'success', 'models' (list of dicts), and optionally 'error'.
    Each model dict contains: model_id, modality, vertical, exam_status, benchmark_status,
    upload_timestamp, exam_reason, benchmark_reason, benchmark_runs.
    benchmark_runs is empty for models that failed or have not yet passed the entrance exam.
    """
    base = api_url or os.environ.get("GAS_API_URL", GAS_API_BASE_URL)
    url = f"{base}/api/v1/miner/models"

    params: Dict[str, str] = {}
    if modality:
        params["modality"] = modality

    headers = generate_header(wallet.hotkey, b"")

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
    except httpx.RequestError as e:
        return {"success": False, "models": [], "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {
            "success": False,
            "models": [],
            "error": f"API error {resp.status_code}: {resp.text}",
        }

    return {"success": True, "models": resp.json()}


def fetch_generator_performance(
    wallet: bt.Wallet,
    modality: Optional[str] = None,
    lookback_days: int = 7,
    api_url: Optional[str] = None,
) -> Dict:
    """Query the generator's own verification + fool aggregate via the GAS API.

    Signs the GET with ``generate_header(wallet.hotkey, b\"\")`` (Epistula v2, empty body).
    """
    base = api_url or os.environ.get("GAS_API_URL", GAS_API_BASE_URL)
    url = f"{base}/api/v1/generator/performance"

    params: Dict[str, str] = {}
    if modality:
        params["modality"] = modality
    params["lookback_days"] = str(lookback_days)

    headers = generate_header(wallet.hotkey, b"")

    try:
        # DuckDB + cache can exceed 30s on cold start
        resp = httpx.get(url, headers=headers, params=params, timeout=120.0)
    except httpx.RequestError as e:
        return {"success": False, "data": None, "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {
            "success": False,
            "data": None,
            "error": f"API error {resp.status_code}: {resp.text}",
        }

    return {"success": True, "data": resp.json()}
