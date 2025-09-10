from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import bittensor as bt
import numpy as np
from PIL import Image
import torch
import cv2

import clip

from gas.cache.content_manager import ContentManager
from gas.cache.types import MediaEntry, VerificationResult
from gas.generation.prompt_generator import PromptGenerator
from gas.types import Modality


_clip_model_cache = {}


def calculate_clip_alignment(
    media_path: str, prompt: str, model_name: str = "ViT-B/32"
) -> Optional[float]:
    """
    Calculate CLIP alignment score between media and text prompt.

    Args:
        media_path: Path to image or video file
        prompt: Text prompt to compare against
        model_name: CLIP model variant to use
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Lazy load and cache CLIP model first
        if model_name not in _clip_model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device=device)
            _clip_model_cache[model_name] = (model, preprocess, device)
            bt.logging.debug(f"Loaded CLIP model: {model_name} on {device}")

        model, preprocess, device = _clip_model_cache[model_name]

        text_tokens = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        media_path = Path(media_path)
        if not media_path.exists():
            bt.logging.error(f"Media file not found: {media_path}")
        return 0.0

        # Handle images
        if media_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            image = Image.open(media_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
            final_score = float(similarity.cpu().numpy()[0])

        # Handle videos via multi-frame consensus
        elif media_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            cap = cv2.VideoCapture(str(media_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                bt.logging.error(f"Could not read video: {media_path}")
                return 0.0

            # Extract 8 evenly spaced frames
            num_sample_frames = min(8, total_frames)
            frame_indices = np.linspace(
                0, total_frames - 1, num_sample_frames, dtype=int
            )

            frame_scores = []
            consensus_threshold = 0.25  # Minimum score for a frame to "pass"
            min_consensus_ratio = 0.6  # At least 60% of frames must pass

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image_tensor = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        frame_features = model.encode_image(image_tensor)
                        frame_features = frame_features / frame_features.norm(
                            dim=-1, keepdim=True
                        )

                        frame_similarity = torch.cosine_similarity(
                            frame_features, text_features, dim=-1
                        )
                        frame_score = float(frame_similarity.cpu().numpy()[0])
                        frame_scores.append(frame_score)

            cap.release()

            if not frame_scores:
                bt.logging.error(f"Could not extract frames from video: {media_path}")
                return 0.0

            # Multi-frame consensus calculation
            passing_frames = sum(
                1 for score in frame_scores if score >= consensus_threshold
            )
            consensus_ratio = passing_frames / len(frame_scores)
            avg_frame_score = np.mean(frame_scores)
            max_frame_score = max(frame_scores)

            bt.logging.info(
                f"Video analysis: {passing_frames}/{len(frame_scores)} frames passed threshold {consensus_threshold:.2f}"
            )
            bt.logging.info(
                f"Frame scores: min={min(frame_scores):.3f}, max={max_frame_score:.3f}, avg={avg_frame_score:.3f}"
            )
            bt.logging.info(
                f"Consensus ratio: {consensus_ratio:.2f} (required: {min_consensus_ratio:.2f})"
            )

            if consensus_ratio < min_consensus_ratio:
                consensus_penalty = consensus_ratio / min_consensus_ratio
                final_score = avg_frame_score * consensus_penalty
                bt.logging.info(
                    f"Consensus penalty applied: {final_score:.3f} (was {avg_frame_score:.3f})"
                )
            else:
                final_score = avg_frame_score * 0.8 + max_frame_score * 0.2
                bt.logging.info(
                    f"Consensus bonus applied: {final_score:.3f} (avg: {avg_frame_score:.3f}, max: {max_frame_score:.3f})"
                )

        else:
            bt.logging.error(f"Unsupported media format: {media_path.suffix}")
            return 0.0

        bt.logging.info(f"CLIP alignment score: {final_score:.4f}")

        return final_score
        
    except Exception as e:
        bt.logging.error(f"Error in CLIP alignment calculation: {e}")
        return None


def calculate_verification_score(
    original_prompt: str,
    media_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate verification score using CLIP visual-text alignment."""
    try:
        if not media_path:
            bt.logging.error("Media path required for CLIP-based verification")
            return {
                "score": 0.0,
                "error": "Missing media path",
                "method": "clip_alignment",
            }

        clip_score = calculate_clip_alignment(media_path, original_prompt)

        if clip_score is None:
            bt.logging.error("CLIP alignment calculation failed")
            return {
                "score": 0.0,
                "error": "CLIP calculation failed",
                "method": "clip_alignment",
            }

        return {
            "score": clip_score,
            "clip_score": clip_score,
            "original_prompt": original_prompt,
            "method": "clip_alignment",
        }

    except Exception as e:
        bt.logging.error(f"Error in verification scoring: {e}")
        return {
            "score": 0.0,
            "error": str(e),
            "method": "clip_alignment",
        }


def verify_single_media(
    content_manager: ContentManager,
    media_entry: MediaEntry,
    threshold: float = 0.25,
) -> VerificationResult:
    """Verify a single miner media entry using CLIP alignment."""
    try:
        if not media_entry.prompt_id:
            bt.logging.error(
                f"Media entry {media_entry.id} has no associated prompt_id"
            )
            return VerificationResult(media_entry=media_entry, passed=False)

        original_prompt = content_manager.get_prompt_by_id(media_entry.prompt_id)
        if not original_prompt:
            bt.logging.error(
                f"Could not retrieve original prompt {media_entry.prompt_id}"
            )
            return VerificationResult(media_entry=media_entry, passed=False)

        verification_score = calculate_verification_score(
            original_prompt=original_prompt,
            media_path=media_entry.file_path,
        )

        score = verification_score.get("score", 0.0)
        passed = score >= threshold
        verification_score["passed"] = passed
        verification_score["threshold"] = threshold

        bt.logging.info("=" * 80)
        bt.logging.info(
            f"🔍 VERIFICATION RESULT ({media_entry.modality}) - Media ID: {media_entry.id}"
        )
        bt.logging.info(
            f"📊 Score: {score:.3f} | Passed: {'✅' if passed else '❌'} | Threshold: {threshold}"
        )

        clip_score = verification_score.get("clip_score")
        if clip_score is not None:
            bt.logging.info(f"🎯 CLIP Alignment: {clip_score:.3f} (raw)")
        else:
            bt.logging.info("🎯 CLIP Alignment: Calculation failed")

        bt.logging.info(f"📝 Original prompt: {original_prompt}")
        bt.logging.info("=" * 80)

        return VerificationResult(
            media_entry=media_entry,
            original_prompt=original_prompt,
            verification_score=verification_score,
            passed=passed,
        )

    except Exception as e:
        bt.logging.error(f"Error verifying media {media_entry.id}: {e}")
        return VerificationResult(media_entry=media_entry, passed=False)


def run_verification_batch(
    content_manager: ContentManager,
    batch_size: int = 10,
    threshold: float = 0.25,
) -> List[VerificationResult]:
    """Run CLIP verification on a batch of unverified miner media."""
    bt.logging.info(f"Starting verification batch (batch_size={batch_size})")

    unverified_media = content_manager.get_unverified_miner_media()
    bt.logging.info(f"Found {len(unverified_media)} unverified miner media entries")
    if not unverified_media:
        bt.logging.info("No unverified miner media found")
        return []

    batch = unverified_media[:batch_size]
    results = []

    for i, media_entry in enumerate(batch, 1):
        bt.logging.debug(f"Processing media {i}/{len(batch)}: {media_entry.file_path}")

        result = verify_single_media(
            content_manager=content_manager,
            media_entry=media_entry,
            threshold=threshold,
        )
        results.append(result)

        if result.passed and result.verification_score:
            try:
                content_manager.mark_miner_media_verified(media_entry.id)
                bt.logging.debug(f"Marked media {media_entry.id} as verified")
            except Exception as e:
                bt.logging.error(f"Error marking media as verified: {e}")

    successful_verifications = sum(
        1 for r in results if r.verification_score is not None
    )
    passed_verifications = sum(1 for r in results if r.passed)

    bt.logging.info(
        f"Verification batch complete: {successful_verifications}/{len(results)} successful, "
        f"{passed_verifications}/{len(results)} passed verification"
    )

    return results


def get_verification_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "successful": 0,
            "passed": 0,
            "failed": 0,
            "error_rate": 0.0,
        }

    total = len(results)
    successful = sum(1 for r in results if r.verification_score is not None)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if r.verification_score is None)

    scores = [
        (
            r.verification_score.get("score", 0.0)
            if isinstance(r.verification_score, dict)
            else r.verification_score
        )
        for r in results
        if r.verification_score is not None
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total": total,
        "successful": successful,
        "passed": passed,
        "failed": failed,
        "error_rate": failed / total if total > 0 else 0.0,
        "pass_rate": passed / successful if successful > 0 else 0.0,
        "average_score": avg_score,
    }


def clear_model_cache():
    global _clip_model_cache
    _clip_model_cache.clear()
    bt.logging.debug("Cleared CLIP model cache")
