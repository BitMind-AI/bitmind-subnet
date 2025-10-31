import hashlib
import json
from pathlib import Path
from typing import Optional

import bittensor as bt
import requests

from gas.protocol.epistula import generate_header


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_presigned_url(
    wallet: bt.wallet, 
    upload_endpoint: str, 
    filename: str, 
    file_size: int, 
    file_hash: str, 
    content_type: Optional[str] = None,
    modality: Optional[str] = None
) -> dict:
    """Generate presigned upload URL from the API with optional modality parameter."""
    
    # Prepare request payload
    payload = {
        'filename': filename,
        'file_size': file_size,
        'expected_hash': file_hash,
    }
    if content_type:
        payload['content_type'] = content_type
    if modality:
        payload['modality'] = modality
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    print(f"\n📡 Requesting presigned URL...")
    print(f"  Request signed by: {headers['Epistula-Signed-By']}")
    
    # Make the presigned URL request
    try:
        presigned_endpoint = upload_endpoint.rstrip('/') + '/presigned'
        print(f"  Endpoint: {presigned_endpoint}")
        
        response = requests.post(
            presigned_endpoint,
            data=payload_bytes,
            headers=headers,
            timeout=30
        )

        # Parse response
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


def upload_to_r2(presigned_url: str, file_content: bytes, content_type: str = 'application/octet-stream') -> dict:
    """Upload file directly to R2 using presigned URL."""
    
    print(f"🚢 Uploading file to R2...")
    print(f"  Content type: {content_type}")
    print(f"  File size: {len(file_content)} bytes")
    
    try:
        # Upload using PUT request to presigned URL
        response = requests.put(
            presigned_url,
            data=file_content,
            headers={'Content-Type': content_type},
            timeout=300  # 5 minutes for large files
        )
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": {
                "message": "Upload successful" if response.status_code == 200 else "Upload failed",
                "etag": response.headers.get('ETag', ''),
                "response_text": response.text
            }
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Upload failed: {str(e)}"}
        }


def confirm_upload(wallet: bt.wallet, upload_endpoint: str, model_id: int, file_hash: str) -> dict:
    """Confirm file upload and finalize model record."""
    
    payload = {
        'model_id': model_id,
        'file_hash': file_hash
    }
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    print(f"📡 Confirming upload...")
    print(f"  Model ID: {model_id}")
    print(f"  File hash: {file_hash}")
    
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
    wallet: bt.wallet,
    file_path: str,
    modality: str,
    upload_endpoint: str
) -> dict:
    """Upload a single modality file (image or video model)."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path_obj, 'rb') as f:
        file_content = f.read()

    file_hash = calculate_sha256(file_content)
    file_size = len(file_content)
    filename = file_path_obj.name

    print(f"\n{'='*70}")
    print(f"UPLOADING {modality.upper()} MODEL")
    print(f"{'='*70}")
    print(f"📁 File: {filename}")
    print(f"📊 Size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"🔐 Hash: {file_hash}")

    # Step 1: Generate presigned URL with modality
    print(f"\n[1/3] Generating presigned URL...")
    presigned_result = generate_presigned_url(
        wallet,
        upload_endpoint,
        filename,
        file_size,
        file_hash,
        'application/octet-stream',
        modality
    )

    if not presigned_result['success']:
        return {
            "success": False,
            "modality": modality,
            "step": "presigned_url_generation",
            "error": presigned_result['response'].get('error', 'Unknown error'),
            "response": presigned_result['response']
        }

    presigned_data = presigned_result['response']['data']
    model_id = presigned_data['model_id']
    presigned_url = presigned_data['presigned_url']
    r2_key = presigned_data['r2_key']

    print(f"✅ Presigned URL generated!")
    print(f"  Model ID: {model_id}")
    print(f"  R2 Key: {r2_key}")
    print(f"\n[2/3] Uploading to R2...")
    upload_result = upload_to_r2(presigned_url, file_content, 'application/octet-stream')

    if not upload_result['success']:
        return {
            "success": False,
            "modality": modality,
            "step": "r2_upload",
            "model_id": model_id,
            "error": upload_result['response'].get('error', 'Unknown error'),
            "response": upload_result['response']
        }

    print(f"✅ File uploaded to R2!")
    print(f"  ETag: {upload_result['response'].get('etag', 'N/A')}")

    print(f"\n[3/3] Confirming upload...")
    confirm_result = confirm_upload(wallet, upload_endpoint, model_id, file_hash)

    if not confirm_result['success']:
        return {
            "success": False,
            "modality": modality,
            "step": "upload_confirmation",
            "model_id": model_id,
            "error": confirm_result['response'].get('error', 'Unknown error'),
            "response": confirm_result['response']
        }

    print(f"✅ Upload confirmed!")
    return {
        "success": True,
        "modality": modality,
        "model_id": model_id,
        "r2_key": r2_key,
        "file_hash": file_hash,
        "file_size": file_size
    } 