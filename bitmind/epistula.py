import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
from typing import Annotated, Any, Dict, Optional

import bittensor as bt
import numpy as np
import asyncio
import base64
import cv2
import ast
import time
import httpx
import aiohttp
from substrateinterface import Keypair

from bitmind.types import Modality, MinerType


EPISTULA_VERSION = str(2)


def generate_header(
    hotkey: Keypair,
    body: Any,
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    req_hash = None
    if isinstance(body, bytes):
        req_hash = sha256(body).hexdigest()
    else:
        req_hash = sha256(json.dumps(body).encode("utf-8")).hexdigest()

    headers = {
        "Epistula-Version": EPISTULA_VERSION,
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{req_hash}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers


def verify_signature(
    signature, body: bytes, timestamp, uuid, signed_for, signed_by, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 8000
    keypair = Keypair(ss58_address=signed_by)
    if timestamp + ALLOWED_DELTA_MS < now:
        return "Request is too stale"
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
    verified = keypair.verify(message, signature)
    if not verified:
        return "Signature Mismatch"
    return None


def create_header_hook(hotkey, axon_hotkey, model):
    async def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            request.headers[key] = header

    return add_headers


async def get_miner_type(
    uid: int,
    axon_info: bt.AxonInfo,
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Query a miner's type by hitting its miner_info endpoint.

    Args:
        uid: miner uid
        axon_info: miner AxonInfo
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        The miner's type as a string, or None if the request fails
    """
    response = {
        "status": 500,
        "error": "",
        "miner_type": None
    }

    try:
        base_url = f"http://{axon_info.ip}:{axon_info.port}"
        url = f"{base_url}/miner_info"

        headers = generate_header(hotkey, b"", axon_info.hotkey)
        timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=sock_connect_timeout
        )

        async with session.get(url, headers=headers, timeout=timeout) as res:
            response["status"] = res.status
            if res.status == 404:
                # backwards compatibility for legacy detectors without miner_info endpoint
                response["miner_type"] = "detector"
                return response
            elif res.status != 200:
                response["error"] = f"HTTP {res.status}"
                return response

            try:
                info = await res.json()
                if not isinstance(info, dict):
                    response["error"] = "Response was not a dictionary"
                    return response

                response["miner_type"] = info.get("miner_type")
                return response

            except json.JSONDecodeError:
                response["error"] = "Failed to decode JSON response"
                return response

    except asyncio.TimeoutError:
        response["status"] = 408
        response["error"] = "Request timed out"
    except aiohttp.ClientConnectorError as e:
        response["status"] = 503
        response["error"] = f"Connection error: {str(e)}"
    except aiohttp.ClientError as e:
        response["status"] = 400
        response["error"] = f"Network error: {str(e)}"
    except Exception as e:
        response["error"] = f"Unknown error: {str(e)}"

    return response


async def query_miner(
    uid: int,
    miner_type: MinerType,
    media: bytes,
    content_type: str,
    modality: Modality,
    axon_info: bt.AxonInfo,
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
    testnet_metadata: dict = None,
) -> Dict[str, Any]:
    """
    Query a miner with media data.

    Args:
        uid: miner uid
        media: encoded media
        content_type: determined by media_to_bytes
        modality: Type of media ('image' or 'video')
        axon_info: miner AxonInfo
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        Dictionary containing the response.
        prediction field will be None if any error is encountered, including
        if the response contains a prediction that doesn't sum to ~1.
    """
    response = {
        "uid": uid,
        "hotkey": axon_info.hotkey,
        "status": 500,
        "prediction": None,
        "error": "",
    }

    try:
        base_url = f"http://{axon_info.ip}:{axon_info.port}"
        if miner_type == MinerType.DETECTOR:
            url = f"{base_url}/detect_{modality}"
        elif miner_type == MinerType.SEGMENTER:
            url = f"{base_url}/segment_{modality}"

        headers = {
            "Content-Type": content_type,
            "X-Media-Type": modality,
            **generate_header(hotkey, media, axon_info.hotkey),
        }

        if testnet_metadata:
            for k, v in testnet_metadata.items():
                if miner_type == MinerType.SEGMENTER and k == 'mask':
                    resized_mask = cv2.resize(v, (128, 128))
                    _, buffer = cv2.imencode('.png', (resized_mask * 255).astype(np.uint8))
                    b64_mask = base64.b64encode(buffer).decode('utf-8')
                    headers["X-Testnet-mask"] = b64_mask
                elif k != 'mask':
                    headers[f"X-Testnet-{k}"] = str(v)

        async with session.post(
            url,
            headers=headers,
            data=media,
            timeout=aiohttp.ClientTimeout(
                total=total_timeout,
                connect=connect_timeout,
                sock_connect=sock_connect_timeout,
            ),
        ) as res:
            response["status"] = res.status
            if res.status != 200:
                response["error"] = f"HTTP {res.status} error"
                return response
            try:

                if miner_type == MinerType.DETECTOR:
                    data = await res.json()
                    if "prediction" not in data:
                        response["error"] = "Missing prediction in response"
                        return response

                    pred = [float(p) for p in data["prediction"]]

                    # handle binary predictions, assume [real, fake]
                    if len(pred) == 2:
                        pred = pred + [0.0]

                    pred = np.array(pred)

                    # error on predictions that don't sum to ~1 or contain values outside of [0., 1.]
                    if abs(sum(pred) - 1.0) > 1e-6 or np.any((pred < 0.0) | (pred > 1.0)):
                        raise ValueError

                    response["prediction"] = pred
                    return response

                elif miner_type == MinerType.SEGMENTER:
                    pred = await res.read()
                    if "X-Mask-Shape" not in res.headers:
                        raise ValueError("Missing X-Mask-Shape header")
                    mask_shape = [int(x) for x in res.headers["X-Mask-Shape"].split(",")]
                    pred = np.frombuffer(pred, dtype=np.float16).reshape(mask_shape)

                    if np.any((pred < 0.0) | (pred > 1.0)):
                        raise ValueError

                    response["prediction"] = pred
                    return response

            except json.JSONDecodeError:
                response["error"] = "Failed to decode JSON response"
                return response

            except (TypeError, ValueError) as e:
                response["error"] = (
                    f"Invalid prediction value. {e}"
                )
                return response

    except asyncio.TimeoutError:
        response["status"] = 408
        response["error"] = "Request timed out"
    except aiohttp.ClientConnectorError as e:
        response["status"] = 503
        response["error"] = f"Connection error: {str(e)}"
    except aiohttp.ClientError as e:
        response["status"] = 400
        response["error"] = f"Network error: {str(e)}"
    except Exception as e:
        response["error"] = f"Unknown error: {str(e)}"

    return response


