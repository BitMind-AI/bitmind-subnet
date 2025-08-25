import json
import asyncio
import aiohttp
import bittensor as bt
from typing import Dict, Any, Optional

from gas.protocol.epistula import generate_header
from gas.types import Modality, MediaType


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
        "uid": uid,
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
            if res.status != 200:
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


async def query_generative_miner(
    uid: int,
    axon_info: bt.AxonInfo,
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    prompt: str,
    modality: Modality,
    media_type: MediaType,
    webhook_url: str,
    parameters: Dict[str, Any],
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query a generative miner with a prompt.

    Args:
        uid: miner uid
        axon_info: miner AxonInfo
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        prompt: The prompt for the generative task.
        webhook_url: The URL for the miner to call back to.
        parameters: Additional parameters for the generation.
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        Dictionary containing the response.
    """
    response = {
        "uid": uid,
        "hotkey": axon_info.hotkey,
        "status": 500,
        "task_id": None,
        "error": "",
    }

    try:
        base_url = f"http://{axon_info.ip}:{axon_info.port}"
        #action = "gen" if media_type == MediaType.SYNTHETIC else "mod"
        #url = f"{base_url}/{action}_{modality.value}"
        url = f"{base_url}/gen_image"

        payload = {
            "prompt": prompt,
            "parameters": parameters,
        }
        body_bytes = json.dumps(payload).encode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-URL": webhook_url,
            **generate_header(hotkey, body_bytes, axon_info.hotkey),
        }

        async with session.post(
            url,
            headers=headers,
            data=body_bytes,
            timeout=aiohttp.ClientTimeout(
                total=total_timeout,
                connect=connect_timeout,
                sock_connect=sock_connect_timeout,
            ),
        ) as res:
            response["status"] = res.status
            if res.status != 202:
                response["error"] = f"HTTP {res.status} error: {await res.text()}"
                return response
            try:
                data = await res.json()
                if "task_id" not in data:
                    response["error"] = "Missing task_id in response"
                    return response

                response["task_id"] = data["task_id"]
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


async def query_orchestrator(
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    miner_uids: str,
    modality: Modality,
    media: bytes,
    source_type: str,
    source_name: str,
    label: int,
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query the Orchestrator's challenge-miners endpoint to challenge specific miners.

    Args:
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        miner_uids: Comma-separated string of miner UIDs to challenge
        modality: Modality.IMAGE or Modality.VIDEO
        source_type: generated, dataset, or scraped
        source_name: model name, dataset name, or source url
        media: Binary file data (image or video)
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        Dictionary containing the response.
    """
    response = {
        "status": 500,
        "error": "",
        "result": {}
    }

    try:
        base_url = f"https://orchestrator.bitmind.workers.dev"

        querystr = (
            f"minerUids={miner_uids}"
            f"&type={modality.value}"
            f"&source_type={source_type}"
            f"&source_name={source_name}"
            f"&label={label}"
        )
        url = f"{base_url}/challenge-miners?{querystr}"

        headers = generate_header(
            hotkey,
            media,
            signed_for=None
        )

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
                response["error"] = f"HTTP {res.status} error: {await res.text()}"
                return response
            try:
                return await res.json()

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