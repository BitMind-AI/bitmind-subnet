"""
Protocol module for GAS subnet.

This module contains network communication utilities including:
- Epistula: Cryptographic headers and authentication for secure communication
- Encoding: Media encoding utilities for network transmission
"""

from .epistula import (
    generate_header,
    verify_signature,
    create_header_hook,
    get_verifier,
    determine_epistula_version_and_verify,
)

from .validator_requests import (
    get_miner_type,
    query_generative_miner,
    query_orchestrator,
)

from .encoding import (
    image_to_bytes,
    video_to_bytes,
    media_to_bytes,
)

__all__ = [
    # Epistula functions
    "generate_header",
    "verify_signature", 
    "create_header_hook",
    "get_miner_type",
    "query_generative_miner",
    "get_verifier",
    "determine_epistula_version_and_verify",
    # Encoding functions
    "image_to_bytes",
    "video_to_bytes", 
    "media_to_bytes",
] 