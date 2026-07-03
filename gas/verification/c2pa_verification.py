"""
C2PA Content Credentials verification for miner-submitted media.
 
Validates that content has authentic provenance from trusted AI generators
like OpenAI (DALL-E), Google (Gemini/Imagen), Adobe Firefly, etc.
 
Trust Anchors
-------------
c2pa-python >= 0.29.0 enforces trust anchor checking by default. Any cert not
chaining to a root in c2pa-rs's built-in store fires signingCredential.untrusted.
 
Several AI providers use private or non-CAI CAs that are not in that store:
  - Stability AI      → GlobalSign GCC R6 SMIME CA 2023 → GlobalSign Root CA - R6
  - Runway (pre-Gen4) → GlobalSign GCC R6 SMIME CA 2023 → GlobalSign Root CA - R6
  - Runway Gen4       → private CN=Stability AI self-signed root
  - Black Forest Labs → GlobalSign GCC R6 SMIME CA 2023 → GlobalSign Root CA - R6
  - Microsoft         → Microsoft Supply Chain RSA Root CA 2022 (private self-signed)
  - OpenAI (Truepic)  → Truepic WebClaimSigningCA (Truepic RootCA not embedded)
  - Adobe             → Adobe Product Services G4 → Adobe Root CA G2 (likely trusted natively)
  - Google            → Google C2PA Root CA G3 (fetched from pki.goog/c2pa/root-g3.crt)
  - ByteDance Seedance → GlobalSign GCC R45 SMIME CA 2025 → GlobalSign Secure Mail Root R45
 
PEM cert chains for these providers live in:
  gas/verification/trust_anchors/
 
When running on >= 0.29.0, _build_c2pa_reader() supplies them via the
Settings/Context API so signingCredential.untrusted is not raised for
legitimate content from these providers.

Note: OpenAI's Sora 2 (via OpenRouter) uses a self-signed end-entity cert
(OpenAI Media Service, CA=false) which c2pa-rs correctly rejects. This is
an OpenAI-side issue — no trust anchor can fix a non-CA self-signed cert.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import bittensor as bt

import c2pa


TRUST_ANCHORS_DIR = Path(__file__).parent / "trust_anchors"

# PEM files whose root CAs are NOT in c2pa-rs's built-in CAI trust list.
# Loaded at module init; gracefully skipped if files are absent.
_PRIVATE_CA_PEM_FILES = [
    "stability_ai.pem",       # GlobalSign GCC R6 SMIME chain
    "runway.pem",             # GlobalSign GCC R6 SMIME chain
    "runway_gen4.pem",        # private CN=Stability AI self-signed root
    "black_forest_labs.pem",  # GlobalSign GCC R6 SMIME chain
    "microsoft.pem",          # Microsoft Supply Chain RSA Root CA 2022
    "openai_truepic.pem",     # Truepic WebClaimSigningCA chain
    "adobe.pem",              # Adobe Product Services G4 chain
    "google_c2pa.pem",        # Google C2PA Root CA G3 (fetched from pki.goog)
    "globalsign_r45.pem",     # GlobalSign Secure Mail Root R45 (ByteDance/Seedance via OpenRouter)
    "bytedance_seedance.pem", # ByteDance Seedance C2PA (GlobalSign R45 root chain)
]


def _load_trust_anchors() -> Optional[str]:
    """
    Concatenate all custom trust-anchor PEM files into one string.

    Returns None if the trust_anchors directory is missing or empty, so callers
    can fall back to c2pa.Reader without a Context on older library versions.
    """
    if not TRUST_ANCHORS_DIR.is_dir():
        return None
    pem_parts: List[str] = []
    for fname in _PRIVATE_CA_PEM_FILES:
        fpath = TRUST_ANCHORS_DIR / fname
        if fpath.exists():
            pem_parts.append(fpath.read_text())
    return "".join(pem_parts) if pem_parts else None


# Cached at import time — avoids re-reading PEM files on every verification call.
_TRUST_ANCHORS_PEM: Optional[str] = _load_trust_anchors()

# Detect whether the installed c2pa-python supports the Settings/Context API
# (added in 0.29.0). We probe once at import and cache the result.
_C2PA_HAS_CONTEXT_API: bool = hasattr(c2pa, "Settings") and hasattr(c2pa, "Context")


def _open_c2pa_reader(file_path: str):
    """
    Return a c2pa.Reader context manager, optionally configured with custom
    trust anchors when the Settings/Context API is available (>= 0.29.0).

    Usage::

        with _open_c2pa_reader(path) as reader:
            manifest_json = reader.json()
    """
    if _C2PA_HAS_CONTEXT_API and _TRUST_ANCHORS_PEM:
        try:
            settings = c2pa.Settings.from_dict({
                "verify": {"verify_cert_anchors": True},
                "trust": {"user_anchors": _TRUST_ANCHORS_PEM},
            })
            ctx = c2pa.Context(settings)
            return c2pa.Reader(file_path, context=ctx)
        except Exception as e:
            bt.logging.debug(f"Could not build c2pa Context with trust anchors: {e}; falling back")
    return c2pa.Reader(file_path)


TRUSTED_CERT_ISSUERS = {
    "OpenAI",           # DALL-E, Sora
    "Google",           # Imagen, Gemini, Veo
    "Adobe",            # Firefly
    "Microsoft",        # Copilot, Designer
    "Meta",             # Imagine
    "Runway",           # Gen-3
    "Stability AI",     # Stable Diffusion, Stable Video
    "Black Forest Labs",  # FLUX
    "Midjourney",
    "Pika",             # Pika Labs
    "Luma",             # Dream Machine
    "Ideogram",
    "Leonardo",         # Leonardo.AI
    "Kuaishou", "Kling",  # Kling AI
    "MiniMax",
    "Haiper",
    "Lightricks",       # LTX Studio
    "Shutterstock",
    "ByteDance",        # Seedream
    "Byteplus",         # Seedance (ByteDance cloud/BytePlus)
}

TRUSTED_CA_ISSUERS = {
    # Traditional Certificate Authorities
    "DigiCert", "GlobalSign", "Entrust", "Sectigo", "Let's Encrypt",
    "Amazon", "Google", "Microsoft", "Apple", "Cloudflare",
    "Comodo", "GeoTrust", "Thawte", "VeriSign", "GoDaddy",
    "IdenTrust", "Symantec", "QuoVadis", "SwissSign",
    # AI companies (may appear as cert issuers in C2PA manifests)
    *TRUSTED_CERT_ISSUERS,
}

AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
}

PROVIDER_PROFILES = {
    "openai": {
        "aliases": ("OpenAI", "DALL-E", "Sora", "Truepic"),
        "requires_ai_source": False,
    },
    "google": {
        "aliases": ("Google", "Imagen", "Gemini", "Veo"),
        "requires_ai_source": False,
    },
    "adobe": {
        "aliases": ("Adobe", "Firefly"),
        "requires_ai_source": False,
    },
    "microsoft": {
        "aliases": ("Microsoft", "Copilot", "Designer"),
        "requires_ai_source": False,
    },
    "runway": {
        "aliases": ("Runway",),
        "requires_ai_source": False,
    },
    "stability_ai": {
        "aliases": ("Stability AI", "Stable Diffusion", "Stable Video"),
        "requires_ai_source": False,
    },
    "black_forest_labs": {
        "aliases": ("Black Forest Labs", "FLUX"),
        "requires_ai_source": False,
    },
    "bytedance": {
        "aliases": ("ByteDance", "BytePlus", "Seedance", "BytePlus_ModelArk"),
        "requires_ai_source": False,
    },
}


class C2PAVerificationResult:
    def __init__(
        self,
        verified: bool = False,
        signature_valid: bool = False,
        issuer: Optional[str] = None,
        cert_issuer: Optional[str] = None,
        is_trusted_issuer: bool = False,
        is_self_signed: bool = False,
        ai_generated: bool = False,
        model_name: Optional[str] = None,
        manifest_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        provider_profile: Optional[str] = None,
        metadata_complete: bool = False,
        policy_warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.verified = verified
        self.signature_valid = signature_valid
        self.issuer = issuer
        self.cert_issuer = cert_issuer
        self.is_trusted_issuer = is_trusted_issuer
        self.is_self_signed = is_self_signed
        self.ai_generated = ai_generated
        self.model_name = model_name
        self.manifest_data = manifest_data or {}
        self.validation_errors = validation_errors or []
        self.provider_profile = provider_profile
        self.metadata_complete = metadata_complete
        self.policy_warnings = policy_warnings or []
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "signature_valid": self.signature_valid,
            "issuer": self.issuer,
            "cert_issuer": self.cert_issuer,
            "is_trusted_issuer": self.is_trusted_issuer,
            "is_self_signed": self.is_self_signed,
            "ai_generated": self.ai_generated,
            "model_name": self.model_name,
            "provider_profile": self.provider_profile,
            "metadata_complete": self.metadata_complete,
            "policy_warnings": self.policy_warnings,
            "validation_errors": self.validation_errors,
            "error": self.error,
        }


def _classify_provider_profile(
    cert_subject: Optional[str],
    cert_issuer: Optional[str],
    claim_generator: Optional[str],
    ai_generated: bool,
) -> Dict[str, Any]:
    """Classify provider-specific C2PA metadata without enforcing policy."""
    haystack = " ".join(
        value for value in (cert_subject, claim_generator) if value
    ).lower()

    provider_profile = None
    for profile_name, profile in PROVIDER_PROFILES.items():
        aliases = profile.get("aliases", ())
        if any(alias.lower() in haystack for alias in aliases):
            provider_profile = profile_name
            break

    policy_warnings = []
    if not provider_profile:
        policy_warnings.append("unknown_provider_profile")
    if not cert_issuer:
        policy_warnings.append("missing_cert_issuer")
    if not cert_subject:
        policy_warnings.append("missing_cert_subject")
    if not claim_generator:
        policy_warnings.append("missing_claim_generator")
    if not ai_generated:
        profile_requires = (
            PROVIDER_PROFILES[provider_profile].get("requires_ai_source", True)
            if provider_profile
            else True
        )
        if profile_requires:
            policy_warnings.append("missing_ai_source_assertion")

    metadata_complete = bool(cert_issuer and (cert_subject or claim_generator))
    return {
        "provider_profile": provider_profile,
        "metadata_complete": metadata_complete,
        "policy_warnings": policy_warnings,
    }


def verify_c2pa(media_data: Union[bytes, str, Path]) -> C2PAVerificationResult:
    """Verify C2PA credentials with full cryptographic validation."""
    temp_file = None
    try:
        # Always read bytes to detect actual format (file extension might be wrong)
        if isinstance(media_data, bytes):
            data = media_data
        else:
            data = Path(media_data).read_bytes()

        # Create temp file with correct extension based on magic bytes
        suffix = detect_media_format(data)
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.write(data)
        temp_file.close()
        file_path = temp_file.name

        try:
            with _open_c2pa_reader(file_path) as reader:
                manifest_json = reader.json()
        except Exception as e:
            return C2PAVerificationResult(verified=False, error=f"No C2PA manifest found: {str(e)}")

        if not manifest_json:
            return C2PAVerificationResult(verified=False, error="Empty C2PA manifest")

        try:
            manifest_data = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
        except json.JSONDecodeError as e:
            return C2PAVerificationResult(verified=False, error=f"Invalid manifest JSON: {e}")

        validation_errors = _check_validation_status(manifest_data)
        if validation_errors:
            bt.logging.warning(f"C2PA signature validation failed: {validation_errors}")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=False,
                validation_errors=validation_errors,
                error=f"Signature validation failed: {'; '.join(validation_errors)}"
            )

        if not _has_verified_hard_binding(manifest_data):
            bt.logging.warning(
                "C2PA rejected: manifest has no verified hard-binding content hash "
                "(c2pa.hash.* assertion)"
            )
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                error="No verified hard-binding content hash assertion in C2PA manifest",
            )

        cert_info = _extract_certificate_info(manifest_data)
        is_self_signed = cert_info.get("is_self_signed", True)
        cert_issuer = cert_info.get("cert_issuer")
        cert_subject = cert_info.get("cert_subject")
        cert_chain_length = cert_info.get("cert_chain_length", 0)

        bt.logging.debug(
            f"C2PA cert info: issuer={cert_issuer}, subject={cert_subject}, "
            f"chain_len={cert_chain_length}, is_self_signed={is_self_signed}"
        )

        if is_self_signed:
            bt.logging.warning(f"C2PA rejected: self-signed certificate (subject: {cert_subject})")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=True,
                cert_issuer=cert_issuer,
                error="Self-signed certificates are not accepted"
            )

        if not _is_trusted_ca(cert_issuer):
            bt.logging.warning(f"C2PA rejected: untrusted CA: {cert_issuer}")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=False,
                cert_issuer=cert_issuer,
                error=f"Certificate issuer not from trusted CA: {cert_issuer}"
            )

        claim_generator = _extract_claim_generator(manifest_data)
        is_trusted = _is_trusted_issuer(cert_subject, cert_issuer, claim_generator)

        if not is_trusted:
            bt.logging.warning(
                f"C2PA rejected: untrusted generator. Subject: {cert_subject}, Issuer: {cert_issuer}, Claim: {claim_generator}"
            )
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=False,
                issuer=claim_generator,
                cert_issuer=cert_issuer,
                is_trusted_issuer=False,
                error=f"Not from a trusted AI generator: {cert_subject or claim_generator}"
            )

        ai_generated = _check_ai_generated(manifest_data)
        model_name = _extract_model_name(manifest_data)
        provider_audit = _classify_provider_profile(
            cert_subject=cert_subject,
            cert_issuer=cert_issuer,
            claim_generator=claim_generator,
            ai_generated=ai_generated,
        )

        bt.logging.info(
            f"C2PA verified: issuer={claim_generator}, model_name={model_name}, "
            f"cert_subject={cert_subject}, cert_issuer={cert_issuer}, "
            f"ai_generated={ai_generated}, "
            f"provider_profile={provider_audit['provider_profile']}, "
            f"metadata_complete={provider_audit['metadata_complete']}, "
            f"policy_warnings={provider_audit['policy_warnings']}"
        )

        return C2PAVerificationResult(
            verified=True,
            signature_valid=True,
            issuer=claim_generator,
            cert_issuer=cert_issuer,
            is_trusted_issuer=True,
            is_self_signed=False,
            ai_generated=ai_generated,
            model_name=model_name,
            manifest_data={
                "raw": manifest_json,
                "provider_audit": provider_audit,
            },
            provider_profile=provider_audit["provider_profile"],
            metadata_complete=provider_audit["metadata_complete"],
            policy_warnings=provider_audit["policy_warnings"],
        )

    except Exception as e:
        bt.logging.warning(f"C2PA verification error: {e}")
        return C2PAVerificationResult(verified=False, error=str(e))
    finally:
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


def detect_media_format(data: bytes) -> str:
    """
    Detect media format from magic bytes.

    Supports:
        - Images: PNG, JPEG, WebP, GIF, BMP, TIFF, HEIC/HEIF, AVIF
        - Video: MP4/M4V/MOV (ftyp-based), WebM, AVI, MKV
        - Audio: MP3, WAV, FLAC, OGG, AAC/M4A

    Args:
        data: Raw bytes of the media file (at least first 12 bytes recommended)

    Returns:
        File extension string (e.g., ".png", ".mp4") or ".bin" if unknown
    """
    if len(data) < 4:
        return ".bin"

    # === Images ===

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return ".png"

    # JPEG: FF D8 FF
    if data[:3] == b'\xff\xd8\xff':
        return ".jpg"

    # WebP: RIFF....WEBP
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return ".webp"

    # GIF: GIF87a or GIF89a
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return ".gif"

    # BMP: 42 4D (BM)
    if data[:2] == b'BM':
        return ".bmp"

    # TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        return ".tiff"

    # === ftyp-based formats (ISO Base Media File Format) ===
    # MP4, M4A, MOV, 3GP, HEIC, AVIF all use ftyp box at offset 4
    # Must check all brands in a single block to avoid unreachable code
    if len(data) >= 12 and data[4:8] == b'ftyp':
        brand = data[8:12]

        # Images: HEIC/HEIF
        if brand in (b'heic', b'heix', b'hevc', b'mif1', b'msf1'):
            return ".heic"
        if brand == b'avif':
            return ".avif"

        # Audio: M4A/AAC
        if brand in (b'M4A ', b'M4B '):
            return ".m4a"

        # Video: QuickTime MOV
        if brand == b'qt  ':
            return ".mov"

        # Video: 3GP/3G2
        if brand in (b'3gp4', b'3gp5', b'3gp6', b'3g2a'):
            return ".3gp"

        # Default: MP4 for other ftyp-based formats (isom, mp41, mp42, etc.)
        return ".mp4"

    # WebM/MKV: EBML header 1A 45 DF A3
    if data[:4] == b'\x1a\x45\xdf\xa3':
        # Both WebM and MKV use this header; WebM is a subset of MKV
        # For C2PA purposes, treat as WebM (more common for web video)
        return ".webm"

    # AVI: RIFF....AVI
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'AVI ':
        return ".avi"

    # === Audio ===

    # MP3: ID3 tag or frame sync (FF FB, FF FA, FF F3, FF F2)
    if data[:3] == b'ID3':
        return ".mp3"
    if data[:2] == b'\xff\xfb' or data[:2] == b'\xff\xfa':
        return ".mp3"
    if data[:2] == b'\xff\xf3' or data[:2] == b'\xff\xf2':
        return ".mp3"

    # WAV: RIFF....WAVE
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return ".wav"

    # FLAC: 66 4C 61 43 (fLaC)
    if data[:4] == b'fLaC':
        return ".flac"

    # OGG: 4F 67 67 53 (OggS)
    if data[:4] == b'OggS':
        return ".ogg"

    return ".bin"


# Affirmative validation codes proving the manifest is hard-bound to the media
# content (c2pa.hash.data for images, c2pa.hash.bmff for BMFF video, boxes for
# JPEG XL/boxes-hashed formats). Without one of these, a signature only proves
# the manifest itself is intact — not that the pixels/samples are unmodified.
HARD_BINDING_MATCH_CODES = {
    "assertion.dataHash.match",
    "assertion.bmffHash.match",
    "assertion.boxesHash.match",
    "assertion.collectionHash.match",
}


def _has_verified_hard_binding(manifest_data: Dict[str, Any]) -> bool:
    try:
        success = (
            manifest_data.get("validation_results", {})
            .get("activeManifest", {})
            .get("success", [])
        )
        return any(s.get("code") in HARD_BINDING_MATCH_CODES for s in success)
    except Exception as e:
        bt.logging.debug(f"Error checking hard binding: {e}")
        return False


def _check_validation_status(manifest_data: Dict[str, Any]) -> List[str]:
    errors = []
    try:
        validation_status = manifest_data.get("validation_status", [])
        critical_codes = [
            "assertion.hashedURI.mismatch",
            "assertion.dataHash.mismatch",
            # ISO BMFF (MP4/MOV): video tampering after signing surfaces here 
            "assertion.bmffHash.mismatch",
            "claim.signature.mismatch",
            "signingCredential.invalid",
            "signingCredential.expired",
            "signingCredential.revoked",
            "signingCredential.untrusted",
            "manifest.inaccessible",
            "assertion.inaccessible",
        ]
        for status in validation_status:
            code = status.get("code", "")
            explanation = status.get("explanation", "")
            if any(critical in code for critical in critical_codes):
                errors.append(f"{code}: {explanation}")
            elif "mismatch" in code.lower():
                errors.append(f"{code}: {explanation}")
            elif code and "error" in code.lower():
                errors.append(f"{code}: {explanation}")
    except Exception as e:
        bt.logging.debug(f"Error checking validation status: {e}")
        errors.append(f"Failed to parse validation status: {e}")
    return errors


def _extract_certificate_info(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "cert_issuer": None,
        "cert_subject": None,
        "is_self_signed": True,
        "cert_chain_length": 0,
    }
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return result

        signature_info = manifest.get("signature_info", {})
        cert_issuer = signature_info.get("issuer")
        result["cert_issuer"] = cert_issuer

        cert_chain = signature_info.get("cert_chain", [])
        result["cert_chain_length"] = len(cert_chain) if cert_chain else 0

        cert_subject = signature_info.get("subject")
        if not cert_subject:
            cert_serial = signature_info.get("cert_serial_number")
            if cert_serial:
                result["cert_subject"] = f"serial:{cert_serial}"
        else:
            result["cert_subject"] = cert_subject

        # Check if cert_issuer is from a trusted CA first
        if cert_issuer and _is_trusted_ca(cert_issuer):
            result["is_self_signed"] = False
        elif cert_issuer and result["cert_subject"]:
            # Only mark as self-signed if issuer == subject (actual self-signed)
            # Don't use cert_chain_length - many valid certs don't include full chain
            result["is_self_signed"] = cert_issuer.lower() == result["cert_subject"].lower()
        elif result["cert_chain_length"] > 1:
            result["is_self_signed"] = False
        # else: default is True (assume self-signed if we can't determine)

    except Exception as e:
        bt.logging.debug(f"Error extracting certificate info: {e}")

    return result


def _extract_claim_generator(manifest_data: Dict[str, Any]) -> Optional[str]:
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return None

        # Older c2pa-rs format: plain string
        claim_generator = manifest.get("claim_generator")
        if claim_generator:
            return claim_generator

        # Newer C2PA spec format (e.g. Google): structured list
        info_list = manifest.get("claim_generator_info", [])
        if info_list:
            name = info_list[0].get("name")
            if name:
                return name

        return None
    except Exception as e:
        bt.logging.debug(f"Error extracting claim generator: {e}")
        return None


def _extract_model_name(manifest_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract a specific model name from a C2PA manifest.

    Only returns a value when the manifest contains an unambiguous model
    identifier — not generic library/infrastructure names or marketing text.

    Checks (in priority order):
    1. `softwareAgent` on c2pa.actions assertions (e.g. older Adobe/Firefly)
    2. `parameters.model_name` on c2pa.actions (e.g. Seedance stores variant here)
    3. `claim_generator` UA string, filtered to non-infrastructure tokens
       (e.g. "DALL-E 3/1.0 c2pa-rs/0.36.1" -> "DALL-E 3")

    Google's manifests don't expose the specific model (Imagen/Veo/etc.) in
    either field, so this returns None for them.
    """
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return None

        for assertion in manifest.get("assertions", []):
            if "c2pa.actions" not in assertion.get("label", ""):
                continue
            action_data = assertion.get("data", {})
            for action in action_data.get("actions", []):
                software_agent = action.get("softwareAgent")
                if software_agent and isinstance(software_agent, str):
                    parsed = _parse_product_name(software_agent)
                    if parsed:
                        return parsed
                # Check params.model_name (used by Seedance/BytePlus).
                # Must come after softwareAgent because dict softwareAgents
                # (e.g. {"name":"BytePlus_ModelArk",...}) are infrastructure.
                params = action.get("parameters", {})
                model_name = params.get("model_name")
                if model_name:
                    return str(model_name)

        claim_generator = manifest.get("claim_generator")
        if claim_generator:
            return _parse_product_name(claim_generator)

    except Exception as e:
        bt.logging.debug(f"Error extracting model name: {e}")

    return None


def _parse_product_name(ua_string: str) -> Optional[str]:
    """
    Extract the primary product name from a UA-style string.

    Examples:
        "DALL-E 3/1.0 c2pa-rs/0.36.1"  -> "DALL-E 3"
        "Adobe Firefly/3.0"             -> "Adobe Firefly"
        "Google Imagen/3"               -> "Google Imagen"
        "c2pa-rs/0.36.1"               -> None  (infrastructure-only)
    """
    if not ua_string:
        return None

    # Skip known C2PA infrastructure library tokens
    INFRA_PREFIXES = ("c2pa", "libc2pa", "contentauth", "trustedpublisher")

    tokens = ua_string.strip().split()
    for token in tokens:
        product = token.split("/")[0].strip()
        if not product:
            continue
        if any(product.lower().startswith(p) for p in INFRA_PREFIXES):
            continue
        return product

    return None


def _is_trusted_ca(issuer: Optional[str]) -> bool:
    if not issuer:
        return False
    issuer_lower = issuer.lower()
    for trusted_ca in TRUSTED_CA_ISSUERS:
        if trusted_ca.lower() in issuer_lower:
            return True
    return False


def _check_ai_generated(manifest_data: Dict[str, Any]) -> bool:
    try:
        for manifest in manifest_data.get("manifests", {}).values():
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                if "c2pa.ai_generated" in label or "c2pa.ai" in label:
                    return True
                data = assertion.get("data", {})
                source_type = data.get("digitalSourceType", "")
                if source_type in AI_SOURCE_TYPES:
                    return True
                for action in data.get("actions", []):
                    action_source = action.get("digitalSourceType", "")
                    if action_source in AI_SOURCE_TYPES:
                        return True
        return False
    except Exception as e:
        bt.logging.debug(f"Error checking AI generation: {e}")
        return False


def _is_trusted_issuer(
    cert_subject: Optional[str],
    cert_issuer: Optional[str],
    claim_generator: Optional[str]
) -> bool:
    if not cert_subject and not cert_issuer:
        return False

    def check_against_trusted(value: str) -> bool:
        if not value:
            return False
        value_lower = value.lower()
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in value_lower or value_lower in trusted.lower():
                return True
        return False

    if cert_subject and check_against_trusted(cert_subject):
        return True

    if cert_issuer:
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in cert_issuer.lower():
                return True

    if claim_generator and (cert_subject or cert_issuer):
        claim_lower = claim_generator.lower()
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in claim_lower:
                return True

    return False



