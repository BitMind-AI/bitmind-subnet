from typing import List, Dict, Any, Optional
import bittensor as bt

from gas.cache.content_manager import ContentManager
from gas.cache.types import MediaEntry, VerificationResult
from .clip_utils import calculate_clip_alignment_consensus, preload_clip_models


def run_verification(
    content_manager: ContentManager,
    batch_size: Optional[int] = 10,
    threshold: float = 0.25,
    clip_batch_size: int = 32,
) -> List[VerificationResult]:
    """
    Run CLIP verification on unverified miner media.

    Args:
        content_manager: ContentManager instance
        batch_size: Number of media entries to process from database (None = process all)
        threshold: Verification threshold
        clip_batch_size: Batch size for CLIP operations (adjust based on GPU memory)
    """
    bt.logging.info(
        f"Starting verification batch (batch_size={batch_size}, clip_batch_size={clip_batch_size})"
    )

    unverified_media = content_manager.get_unverified_miner_media()
    bt.logging.info(f"Found {len(unverified_media)} unverified miner media entries")
    if not unverified_media:
        bt.logging.info("No unverified miner media found")
        return []
            
    preload_clip_models()

    if batch_size is None:
        batch = unverified_media
        bt.logging.debug(f"Processing all {len(batch)} unverified media entries")
    else:
        batch = unverified_media[:batch_size]
        bt.logging.debug(f"Processing {len(batch)} of {len(unverified_media)} unverified media entries")

    results = verify_media(
        content_manager=content_manager,
        media_entries=batch,
        threshold=threshold,
        batch_size=clip_batch_size,
    )

    bt.logging.debug("Writing verification results to db")
    # Mark successful verifications in database
    for result in results:
        if result.passed and result.verification_score:
            try:
                content_manager.mark_miner_media_verified(result.media_entry.id)
                bt.logging.debug(f"Marked media {result.media_entry.id} as verified")
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


def verify_media(
    content_manager: ContentManager,
    media_entries: List[MediaEntry],
    threshold: float = 0.25,
    batch_size: int = 32,
) -> List[VerificationResult]:
    """
    Verify a batch of miner media entries using CLIP consensus scoring.

    Args:
        content_manager: ContentManager instance
        media_entries: List of MediaEntry objects to verify
        threshold: Verification threshold
        batch_size: Batch size for CLIP operations

    Returns:
        List of VerificationResult objects
    """
    try:
        if not media_entries:
            return []

        bt.logging.debug(
            f"Starting batch verification of {len(media_entries)} media entries"
        )

        media_paths = []
        prompts = []
        valid_entries = []

        for media_entry in media_entries:
            if not media_entry.prompt_id:
                bt.logging.warning(
                    f"Media entry {media_entry.id} has no associated prompt_id"
                )
                continue

            original_prompt = content_manager.get_prompt_by_id(media_entry.prompt_id)
            if not original_prompt:
                bt.logging.warning(
                    f"Could not retrieve original prompt {media_entry.prompt_id}"
                )
                continue

            media_paths.append(media_entry.file_path)
            prompts.append(original_prompt)
            valid_entries.append(media_entry)

        if not valid_entries:
            bt.logging.warning("No valid media entries found for batch verification")
            return []

        bt.logging.debug(
            f"Processing {len(valid_entries)} valid entries in batched mode"
        )

        # Run consensus verification
        consensus_results = calculate_clip_alignment_consensus(
            media_paths, prompts, batch_size
        )

        if consensus_results is None:
            bt.logging.error("Consensus verification failed")
            return [
                VerificationResult(media_entry=entry, passed=False)
                for entry in valid_entries
            ]

        # Create verification results
        verification_results = []

        for i, (media_entry, consensus_result) in enumerate(
            zip(valid_entries, consensus_results)
        ):
            consensus_score = consensus_result["consensus_score"]
            passed = consensus_score >= threshold

            verification_score = {
                "score": consensus_score,
                "clip_score": consensus_score,
                "original_prompt": prompts[i],
                "method": "clip_consensus",
                "consensus_details": consensus_result,
                "passed": passed,
                "threshold": threshold,
            }

            result = VerificationResult(
                media_entry=media_entry,
                original_prompt=prompts[i],
                verification_score=verification_score,
                passed=passed,
            )

            verification_results.append(result)

            # Log individual results
            bt.logging.debug(
                f"Media {media_entry.id}: score={consensus_score:.3f}, "
                f"passed={'✅' if passed else '❌'}, "
                f"models={consensus_result['num_models']}/3"
            )

        # Summary logging
        successful_count = len(verification_results)
        passed_count = sum(1 for r in verification_results if r.passed)
        avg_score = sum(
            r.verification_score["score"] for r in verification_results
        ) / len(verification_results)

        bt.logging.info(
            f"✅ Verification complete: {successful_count}/{len(media_entries)} processed, "
            f"{passed_count} passed (pass rate: {passed_count/successful_count:.1%}, "
            f"avg score: {avg_score:.3f})"
        )

        return verification_results

    except Exception as e:
        bt.logging.error(f"Error in batch verification: {e}")
        return [
            VerificationResult(media_entry=entry, passed=False)
            for entry in media_entries
        ]


def get_verification_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    """
    Generate a summary of verification results.

    Args:
        results: List of VerificationResult objects

    Returns:
        Dictionary with summary statistics
    """
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

