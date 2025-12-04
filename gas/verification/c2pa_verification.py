"""
C2PA Content Credentials verification for miner-submitted media.

Validates that content has authentic provenance from trusted AI generators
like OpenAI (DALL-E), Google (Gemini/Imagen), Adobe Firefly, etc.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import bittensor as bt

try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False
    bt.logging.warning("c2pa-python not installed. C2PA verification will be disabled.")


# Known trusted AI generator issuers
# These are organizations that embed C2PA credentials in their AI-generated content
TRUSTED_ISSUERS = [
    # OpenAI
    "openai", "openai.com", "dall-e", "sora", "chatgpt",

    # Google
    "google", "google.com", "google.llc",
    "imagen", "veo", "gemini", "deepmind",

    # Adobe
    "adobe", "adobe.com", "firefly", "contentauthenticity",

    # Microsoft
    "microsoft", "microsoft.com", "bing", "designer", "copilot",

    # Meta
    "meta", "meta.com", "facebook", "instagram",

    # Other
    "runway", "runwayml", "runwayml.com",
    "stability", "stability.ai",
    "pika", "pika.art",
    "canva", "canva.com",
    "shutterstock", "shutterstock.com",
]



class C2PAVerificationResult:
    """Result of C2PA verification."""
    
    def __init__(
        self,
        verified: bool = False,
        issuer: Optional[str] = None,
        is_trusted_issuer: bool = False,
        ai_generated: bool = False,
        manifest_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.verified = verified
        self.issuer = issuer
        self.is_trusted_issuer = is_trusted_issuer
        self.ai_generated = ai_generated
        self.manifest_data = manifest_data or {}
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "issuer": self.issuer,
            "is_trusted_issuer": self.is_trusted_issuer,
            "ai_generated": self.ai_generated,
            "error": self.error,
        }


def verify_c2pa(
    media_data: Union[bytes, str, Path],
    trusted_issuers: Optional[List[str]] = None,
) -> C2PAVerificationResult:
    """
    Verify C2PA content credentials in media.

    Args:
        media_data: Media as bytes or file path
        trusted_issuers: List of trusted issuer patterns (default: TRUSTED_ISSUERS)

    Returns:
        C2PAVerificationResult with verification details
    """
    if not C2PA_AVAILABLE:
        return C2PAVerificationResult(
            verified=False,
            error="c2pa-python library not installed"
        )

    if trusted_issuers is None:
        trusted_issuers = TRUSTED_ISSUERS

    temp_file = None
    try:
        # Handle bytes input by writing to temp file
        if isinstance(media_data, bytes):
            # Detect format from magic bytes
            suffix = _detect_format(media_data)
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_file.write(media_data)
            temp_file.close()
            file_path = temp_file.name
        else:
            file_path = str(media_data)

        # Read C2PA manifest from file
        try:
            with c2pa.Reader(file_path) as reader:
                manifest_json = reader.json()
        except Exception as e:
            # No C2PA manifest found - this is common for most images
            return C2PAVerificationResult(
                verified=False,
                error=f"No C2PA manifest found: {str(e)}"
            )

        if not manifest_json:
            return C2PAVerificationResult(
                verified=False,
                error="Empty C2PA manifest"
            )

        # Parse manifest to extract issuer and AI generation info
        issuer = _extract_issuer(manifest_json)
        ai_generated = _check_ai_generated(manifest_json)
        is_trusted = _is_trusted_issuer(issuer, trusted_issuers)

        return C2PAVerificationResult(
            verified=True,
            issuer=issuer,
            is_trusted_issuer=is_trusted,
            ai_generated=ai_generated,
            manifest_data={"raw": manifest_json},
        )

    except Exception as e:
        bt.logging.warning(f"C2PA verification error: {e}")
        return C2PAVerificationResult(
            verified=False,
            error=str(e)
        )
    finally:
        # Clean up temp file
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


def _detect_format(data: bytes) -> str:
    """Detect image/video format from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return ".png"
    elif data[:2] == b'\xff\xd8':
        return ".jpg"
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return ".webp"
    elif data[:4] == b'\x00\x00\x00\x1c' or data[:4] == b'\x00\x00\x00\x20':
        return ".mp4"
    elif data[:4] == b'\x1a\x45\xdf\xa3':
        return ".webm"
    else:
        return ".bin"


def _extract_issuer(manifest_json: str) -> Optional[str]:
    """Extract issuer/claim generator from C2PA manifest."""
    import json
    try:
        data = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
        
        # Check manifests for claim_generator or issuer
        manifests = data.get("manifests", {})
        for manifest_id, manifest in manifests.items():
            # Check claim_generator field
            claim_generator = manifest.get("claim_generator")
            if claim_generator:
                return claim_generator
            
            # Check signature info
            signature_info = manifest.get("signature_info", {})
            issuer = signature_info.get("issuer")
            if issuer:
                return issuer
        
        # Check active manifest
        active_manifest = data.get("active_manifest")
        if active_manifest and active_manifest in manifests:
            manifest = manifests[active_manifest]
            return manifest.get("claim_generator")
        
        return None
    except Exception as e:
        bt.logging.debug(f"Error extracting issuer: {e}")
        return None


def _check_ai_generated(manifest_json: str) -> bool:
    """Check if manifest indicates AI-generated content."""
    import json
    try:
        data = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
        
        manifests = data.get("manifests", {})
        for manifest_id, manifest in manifests.items():
            # Check assertions for AI generation indicators
            assertions = manifest.get("assertions", [])
            for assertion in assertions:
                label = assertion.get("label", "")
                
                # C2PA uses specific labels for AI-generated content
                if "c2pa.ai_generated" in label:
                    return True
                if "c2pa.ai" in label:
                    return True
                    
                # Check action data
                if assertion.get("data", {}).get("digitalSourceType") == "trainedAlgorithmicMedia":
                    return True
                if assertion.get("data", {}).get("digitalSourceType") == "compositeWithTrainedAlgorithmicMedia":
                    return True
        
        return False
    except Exception as e:
        bt.logging.debug(f"Error checking AI generation: {e}")
        return False


def _is_trusted_issuer(issuer: Optional[str], trusted_issuers: List[str]) -> bool:
    """Check if issuer matches any trusted issuer pattern."""
    if not issuer:
        return False
    
    issuer_lower = issuer.lower()
    for trusted in trusted_issuers:
        if trusted.lower() in issuer_lower:
            return True
    return False


def is_from_trusted_generator(
    media_data: Union[bytes, str, Path],
    require_ai_label: bool = False,
) -> bool:
    """
    Quick check if media is from a trusted AI generator.

    Args:
        media_data: Media as bytes or file path
        require_ai_label: If True, also require AI generation assertion

    Returns:
        True if from trusted generator with valid C2PA credentials
    """
    result = verify_c2pa(media_data)
    
    if not result.verified:
        return False
    
    if not result.is_trusted_issuer:
        return False
    
    if require_ai_label and not result.ai_generated:
        return False
    
    return True

