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
    content_type: Optional[str] = None
) -> dict:
    """Generate presigned upload URL from the API."""
    
    # Prepare request payload
    payload = {
        'filename': filename,
        'file_size': file_size,
        'expected_hash': file_hash,
    }
    if content_type:
        payload['content_type'] = content_type
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    print(f"📡 Requesting presigned URL...")
    print(f"  Filename: {filename}")
    print(f"  File size: {file_size} bytes")
    print(f"  Expected hash: {file_hash}")
    print(f"  SS58 Address: {headers['Epistula-Signed-By']}")
    
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
        
        print(f"  Response status: {response.status_code}")
        print(f"  Response headers: {dict(response.headers)}")
        
        # Parse response
        try:
            result = response.json()
            print(f"  Response body: {result}")
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "text": response.text}
            print(f"  Response text: {response.text}")
        
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
    
    print(f"☁️ Uploading file to R2...")
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
    
    print(f"✅ Confirming upload...")
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


def upload_model_zip_presigned(wallet: bt.wallet, file_path: str, upload_endpoint: str) -> dict:
    """Upload file using presigned URL flow with Epistula authentication."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file content and calculate hash
    with open(file_path_obj, "rb") as f:
        file_content = f.read()

    file_hash = calculate_sha256(file_content)
    file_size = len(file_content)
    filename = file_path_obj.name

    print(f"📁 File Info:")
    print(f"  Path: {file_path}")
    print(f"  Name: {filename}")
    print(f"  Size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Hash: {file_hash}")
    print(f"  SS58 Address: {wallet.hotkey.ss58_address}")
    print(f"  Endpoint: {upload_endpoint}")

    # Step 1: Generate presigned URL
    print(f"\n{'='*60}")
    print(f"STEP 1: Generate Presigned URL")
    print(f"{'='*60}")
    
    presigned_result = generate_presigned_url(
        wallet, 
        upload_endpoint, 
        filename, 
        file_size, 
        file_hash,
        'application/octet-stream'
    )
    
    if not presigned_result['success']:
        return {
            "success": False,
            "step": "presigned_url_generation",
            "error": presigned_result['response'].get('error', 'Unknown error'),
            "response": presigned_result['response']
        }
    
    presigned_data = presigned_result['response']['data']
    model_id = presigned_data['model_id']
    presigned_url = presigned_data['presigned_url']
    r2_key = presigned_data['r2_key']
    expires_at = presigned_data['expires_at']
    
    print(f"✅ Presigned URL generated successfully!")
    print(f"  Model ID: {model_id}")
    print(f"  R2 Key: {r2_key}")
    print(f"  Expires at: {expires_at}")
    
    # Step 2: Upload file to R2
    print(f"\n{'='*60}")
    print(f"STEP 2: Upload File to R2")
    print(f"{'='*60}")
    
    upload_result = upload_to_r2(presigned_url, file_content, 'application/octet-stream')
    
    if not upload_result['success']:
        return {
            "success": False,
            "step": "r2_upload",
            "model_id": model_id,
            "error": upload_result['response'].get('error', 'Unknown error'),
            "response": upload_result['response']
        }
    
    print(f"✅ File uploaded to R2 successfully!")
    print(f"  ETag: {upload_result['response'].get('etag', 'N/A')}")
    
    # Step 3: Confirm upload
    print(f"\n{'='*60}")
    print(f"STEP 3: Confirm Upload")
    print(f"{'='*60}")
    
    confirm_result = confirm_upload(wallet, upload_endpoint, model_id, file_hash)
    
    if not confirm_result['success']:
        return {
            "success": False,
            "step": "upload_confirmation",
            "model_id": model_id,
            "error": confirm_result['response'].get('error', 'Unknown error'),
            "response": confirm_result['response']
        }
    
    confirm_data = confirm_result['response']['data']
    final_model_id = confirm_data['model_id']
    final_r2_key = confirm_data['r2_key']
    
    print(f"✅ Upload confirmed successfully!")
    print(f"  Model ID: {final_model_id}")
    print(f"  R2 Key: {final_r2_key}")
    print(f"  File Hash: {confirm_data['file_hash']}")
    
    return {
        "success": True,
        "model_id": final_model_id,
        "r2_key": r2_key,
        "file_hash": file_hash,
        "file_size": file_size
    } 