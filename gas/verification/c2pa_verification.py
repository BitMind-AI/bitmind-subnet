"""
C2PA Content Credentials verification for miner-submitted media.

Validates that content has authentic provenance from trusted AI generators
like OpenAI (DALL-E), Google (Gemini/Imagen), Adobe Firefly, etc.
"""

import json
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


TRUSTED_CERT_ISSUERS = {
    "OpenAI, Inc.", "OpenAI", "openai.com",
    "Google LLC", "Google Inc.", "google.com",
    "Adobe Inc.", "Adobe Systems Incorporated", "adobe.com", "Adobe Content Authenticity",
    "Microsoft Corporation", "microsoft.com",
    "Meta Platforms, Inc.", "meta.com",
    "Runway AI, Inc.", "RunwayML, Inc.", "runwayml.com",
    "Stability AI Ltd.", "Stability AI", "stability.ai",
    "Pika Labs, Inc.", "pika.art",
    "Canva Pty Ltd", "canva.com",
    "Shutterstock, Inc.", "shutterstock.com",
}

TRUSTED_CA_ISSUERS = {
    "DigiCert", "GlobalSign", "Entrust", "Sectigo", "Let's Encrypt",
    "Amazon", "Google Trust Services", "Microsoft", "Apple", "Cloudflare",
}

AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
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
        manifest_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.verified = verified
        self.signature_valid = signature_valid
        self.issuer = issuer
        self.cert_issuer = cert_issuer
        self.is_trusted_issuer = is_trusted_issuer
        self.is_self_signed = is_self_signed
        self.ai_generated = ai_generated
        self.manifest_data = manifest_data or {}
        self.validation_errors = validation_errors or []
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
            "validation_errors": self.validation_errors,
            "error": self.error,
        }


def verify_c2pa(media_data: Union[bytes, str, Path]) -> C2PAVerificationResult:
    """Verify C2PA credentials with full cryptographic validation."""
    if not C2PA_AVAILABLE:
        return C2PAVerificationResult(verified=False, error="c2pa-python library not installed")

    temp_file = None
    try:
        if isinstance(media_data, bytes):
            suffix = _detect_format(media_data)
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_file.write(media_data)
            temp_file.close()
            file_path = temp_file.name
        else:
            file_path = str(media_data)

        try:
            with c2pa.Reader(file_path) as reader:
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

        cert_info = _extract_certificate_info(manifest_data)
        is_self_signed = cert_info.get("is_self_signed", True)
        cert_issuer = cert_info.get("cert_issuer")
        cert_subject = cert_info.get("cert_subject")

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

        bt.logging.info(
            f"C2PA verified: issuer={claim_generator}, cert_subject={cert_subject}, "
            f"cert_issuer={cert_issuer}, ai_generated={ai_generated}"
        )

        return C2PAVerificationResult(
            verified=True,
            signature_valid=True,
            issuer=claim_generator,
            cert_issuer=cert_issuer,
            is_trusted_issuer=True,
            is_self_signed=False,
            ai_generated=ai_generated,
            manifest_data={"raw": manifest_json},
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


def _detect_format(data: bytes) -> str:
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
    return ".bin"


def _check_validation_status(manifest_data: Dict[str, Any]) -> List[str]:
    errors = []
    try:
        validation_status = manifest_data.get("validation_status", [])
        critical_codes = [
            "assertion.hashedURI.mismatch",
            "assertion.dataHash.mismatch",
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
        
        if cert_issuer and result["cert_subject"]:
            result["is_self_signed"] = (
                cert_issuer.lower() == result["cert_subject"].lower() or
                result["cert_chain_length"] <= 1
            )
        elif result["cert_chain_length"] > 1:
            result["is_self_signed"] = False
        
        if cert_issuer and _is_trusted_ca(cert_issuer):
            result["is_self_signed"] = False
            
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
        
        if manifest:
            return manifest.get("claim_generator")
        return None
    except Exception as e:
        bt.logging.debug(f"Error extracting claim generator: {e}")
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
                source_type = assertion.get("data", {}).get("digitalSourceType", "")
                if source_type in AI_SOURCE_TYPES:
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


def is_from_trusted_generator(media_data: Union[bytes, str, Path], require_ai_label: bool = False) -> bool:
    """Check if media is from a trusted AI generator with valid cryptographic proof."""
    result = verify_c2pa(media_data)
    if not result.verified or not result.signature_valid:
        return False
    if result.is_self_signed or not result.is_trusted_issuer:
        return False
    if require_ai_label and not result.ai_generated:
        return False
    return True
