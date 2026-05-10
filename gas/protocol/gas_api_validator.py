"""Synchronous gas.bitmind.ai calls from validator worker threads (Epistula v2)."""

import json
import logging
from typing import Any, Dict, List, Optional

import bittensor as bt
import requests

from gas.protocol.epistula import generate_header

logger = logging.getLogger(__name__)

GAS_VERIFICATION_UPLOAD_PATH = "/api/v1/validator/generator-verification-upload"
_DEFAULT_TIMEOUT_S = 60
_MAX_ENTRIES = 5000


def _verification_stats_to_entries(
    verification_stats: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip media_ids; shape rows for gas_api POST body."""
    entries: List[Dict[str, Any]] = []
    for hotkey, st in verification_stats.items():
        if not hotkey:
            continue
        entries.append(
            {
                "generator_hotkey": hotkey,
                "generator_uid": st.get("uid"),
                "total_verified": int(st.get("total_verified") or 0),
                "total_failed": int(st.get("total_failed") or 0),
                "total_evaluated": int(st.get("total_evaluated") or 0),
                "pass_rate": float(st.get("pass_rate") or 0.0),
            }
        )
        if len(entries) >= _MAX_ENTRIES:
            logger.warning(
                "Verification upload capped at %s generators (truncate remainder)",
                _MAX_ENTRIES,
            )
            break
    return entries


def post_generator_verification_upload(
    wallet: bt.wallet,
    base_url: str,
    lookback_hours: float,
    verification_stats: Dict[str, Dict[str, Any]],
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> Optional[int]:
    """
    POST aggregated verification stats to gas_api after a successful HF upload cycle.

    Args:
        wallet: Validator wallet (signs with hotkey).
        base_url: e.g. config.benchmark_api_url
        lookback_hours: Must match the window used to build verification_stats.
        verification_stats: Output of ContentManager.get_verification_stats_last_n_hours.

    Returns:
        Number of rows inserted if HTTP 200, else None.
    """
    if not verification_stats:
        return None

    if not base_url:
        logger.warning("generator-verification-upload: base_url is None, skipping upload")
        return None

    entries = _verification_stats_to_entries(verification_stats)
    if not entries:
        return None

    payload = {
        "lookback_hours": float(lookback_hours),
        "entries": entries,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    url = base_url.rstrip("/") + GAS_VERIFICATION_UPLOAD_PATH
    headers = generate_header(wallet.hotkey, body, None)
    headers["Content-Type"] = "application/json"

    try:
        response = requests.post(url, data=body, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        logger.warning("generator-verification-upload request failed: %s", e)
        return None

    if response.status_code != 200:
        logger.warning(
            "generator-verification-upload HTTP %s: %s",
            response.status_code,
            (response.text or "")[:500],
        )
        return None

    try:
        data = response.json()
        inserted = int(data.get("inserted", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        logger.warning("generator-verification-upload: bad JSON response")
        return None

    logger.info(
        "Posted generator verification upload: %s rows (lookback_hours=%s)",
        inserted,
        lookback_hours,
    )
    return inserted
