from collections import defaultdict
from statistics import mean, median
from typing import List, Dict, Any, Optional, Tuple

import bittensor as bt
import torch

from gas.cache.content_manager import ContentManager
from gas.cache.types import MediaEntry, VerificationResult
from .clip_utils import (
    calculate_clip_alignment_consensus,
    preload_clip_models,
    find_near_duplicate_by_embedding,
    serialize_features,
)


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

    pending_media = content_manager.get_miner_media(verification_status="pending")
    bt.logging.info(f"Found {len(pending_media)} pending verification miner media entries")
    if not pending_media:
        bt.logging.info("No pending verification miner media found")
        return []
            
    preload_clip_models()
    bt.logging.info("CLIP models loaded, starting verification")

    if batch_size is None:
        batch = pending_media
        bt.logging.debug(f"Processing all {len(batch)} pending media entries")
    else:
        batch = pending_media[:batch_size]
        bt.logging.info(f"Processing {len(batch)} of {len(pending_media)} pending media entries")

    results = verify_media(
        content_manager=content_manager,
        media_entries=batch,
        threshold=threshold,
        batch_size=clip_batch_size,
    )

    bt.logging.info("Verification batch complete, updating database")
    for result in results:
        vs = result.verification_score or {}
        is_corrupted = isinstance(vs, dict) and vs.get("corrupted", False)
        if result.verification_score and not is_corrupted:
            try:
                if result.passed:
                    content_manager.mark_miner_media_verified(result.media_entry.id)
                    bt.logging.debug(f"Marked media {result.media_entry.id} as verified")
                else:
                    content_manager.mark_miner_media_failed_verification(result.media_entry.id)
                    bt.logging.debug(f"Marked media {result.media_entry.id} as failed verification")
            except Exception as e:
                bt.logging.error(f"Error marking media verification status: {e}")

    successful_verifications = sum(
        1 for r in results if r.verification_score is not None
        and not (isinstance(r.verification_score, dict) and r.verification_score.get("corrupted"))
    )
    passed_verifications = sum(1 for r in results if r.passed)
    failed_verifications = sum(1 for r in results if r.verification_score is not None and not r.passed)

    bt.logging.info(
        f"Verification batch complete: {successful_verifications}/{len(results)} processed, "
        f"{passed_verifications} passed, {failed_verifications} failed verification"
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
            f"Starting verification for {len(media_entries)} media entries"
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

        # Separate images and videos for optimal batching
        image_entries = []
        image_paths = []
        image_prompts = []
        video_entries = []
        video_paths = []
        video_prompts = []
        
        for entry, path, prompt in zip(valid_entries, media_paths, prompts):
            from pathlib import Path
            if Path(path).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                video_entries.append(entry)
                video_paths.append(path)
                video_prompts.append(prompt)
            else:
                image_entries.append(entry)
                image_paths.append(path)
                image_prompts.append(prompt)
        
        bt.logging.info(
            f"Split into {len(image_entries)} images and {len(video_entries)} videos"
        )

        all_results = []
        
        # Process images with large batch size (128)
        image_features = None
        if image_entries:
            bt.logging.info(f"Processing {len(image_entries)} images (batch_size=128)")
            result = calculate_clip_alignment_consensus(
                image_paths, image_prompts, batch_size=128, return_features=True
            )
            if result is not None:
                image_consensus, image_features = result
                all_results.extend(zip(image_entries, image_prompts, image_consensus))
        
        # Process videos with small batch size (32)
        video_features = None
        if video_entries:
            bt.logging.info(f"Processing {len(video_entries)} videos (batch_size=32)")
            result = calculate_clip_alignment_consensus(
                video_paths, video_prompts, batch_size=32, return_features=True
            )
            if result is not None:
                video_consensus, video_features = result
                all_results.extend(zip(video_entries, video_prompts, video_consensus))
        
        if not all_results:
            bt.logging.error("Consensus verification failed for all media")
            return [
                VerificationResult(media_entry=entry, passed=False)
                for entry in valid_entries
            ]

        # Create verification results and detect corrupted videos
        verification_results = []
        corrupted_videos = []

        for media_entry, prompt, consensus_result in all_results:
            # Check if video was corrupted/unreadable (indicated by None score from all models)
            if consensus_result.get("corrupted", False) or consensus_result["consensus_score"] is None:
                bt.logging.warning(
                    f"Corrupted/invalid video detected: {media_entry.file_path} (ID: {media_entry.id})"
                )
                corrupted_videos.append(media_entry)
                result = VerificationResult(
                    media_entry=media_entry,
                    original_prompt=prompt,
                    verification_score={
                        "score": 0.0,
                        "consensus_details": consensus_result,
                        "corrupted": True,
                        "passed": False,
                        "threshold": threshold,
                    },
                    passed=False,
                )
                verification_results.append(result)
                continue
            
            consensus_score = consensus_result["consensus_score"]
            passed = consensus_score >= threshold

            verification_score = {
                "score": consensus_score,
                "clip_score": consensus_score,
                "original_prompt": prompt,
                "method": "clip_consensus",
                "consensus_details": consensus_result,
                "passed": passed,
                "threshold": threshold,
            }

            result = VerificationResult(
                media_entry=media_entry,
                original_prompt=prompt,
                verification_score=verification_score,
                passed=passed,
            )

            verification_results.append(result)

            # Per-entry structured log: makes it possible to tail verifier output
            # and see exactly which (miner, generator-model, modality) is producing
            # which scores.
            individual_scores = consensus_result.get("individual_scores", {}) or {}
            per_model_str = " ".join(
                f"{m}={s:.3f}" for m, s in sorted(individual_scores.items())
            )
            bt.logging.info(
                f"[VERIFY] modality={media_entry.modality.value if hasattr(media_entry.modality, 'value') else media_entry.modality} "
                f"miner_uid={media_entry.uid} "
                f"hotkey={(media_entry.hotkey or '')[:8]} "
                f"gen_model={media_entry.model_name} "
                f"c2pa_issuer={media_entry.c2pa_issuer} "
                f"score={consensus_score:.3f} "
                f"threshold={threshold:.3f} "
                f"passed={'PASS' if passed else 'FAIL'} "
                f"per_model=[{per_model_str}] "
                f"media_id={media_entry.id}"
            )

        # Delete corrupted/invalid videos
        if corrupted_videos:
            bt.logging.warning(
                f"Deleting {len(corrupted_videos)} corrupted/invalid video(s) from storage"
            )
            for media_entry in corrupted_videos:
                try:
                    success = content_manager.delete_media(media_id=media_entry.id)
                    if success:
                        bt.logging.info(f"Deleted corrupted video: {media_entry.file_path}")
                    else:
                        bt.logging.error(f"Failed to delete corrupted video: {media_entry.file_path}")
                except Exception as e:
                    bt.logging.error(f"Error deleting corrupted video {media_entry.file_path}: {e}")

        # Embedding-based duplicate detection
        # Build feature index from CLIP features (same order as entries)
        entry_to_features = {}
        if image_features is not None and len(image_entries) == len(image_features):
            for i, entry in enumerate(image_entries):
                entry_to_features[entry.id] = image_features[i]
        if video_features is not None and len(video_entries) == len(video_features):
            for i, entry in enumerate(video_entries):
                entry_to_features[entry.id] = video_features[i]

        if entry_to_features:
            current_batch_ids = list(entry_to_features.keys())
            stored_embeddings = content_manager.get_embeddings_for_duplicate_check(
                exclude_ids=current_batch_ids, limit=5000,
            )

            for result in verification_results:
                if not result.passed:
                    continue
                entry = result.media_entry
                features = entry_to_features.get(entry.id)
                if features is None:
                    continue

                near_dup = find_near_duplicate_by_embedding(features, stored_embeddings)
                if near_dup is not None:
                    dup_id, sim = near_dup
                    bt.logging.warning(
                        f"REJECTED near-duplicate by embedding: media_id={entry.id} "
                        f"matches {dup_id} with cosine_sim={sim:.4f}"
                    )
                    result.passed = False
                    if result.verification_score:
                        result.verification_score["embedding_duplicate"] = True
                        result.verification_score["embedding_duplicate_of"] = dup_id

                # Store embedding for future duplicate checks
                try:
                    embedding_blob = serialize_features(features)
                    content_manager.store_clip_embedding(entry.id, embedding_blob)
                except Exception as e:
                    bt.logging.debug(f"Failed to store CLIP embedding for {entry.id}: {e}")

        _log_breakdowns(verification_results, threshold)

        return verification_results

    except Exception as e:
        bt.logging.error(f"Error in batch verification: {e}")
        return [
            VerificationResult(media_entry=entry, passed=False)
            for entry in media_entries
        ]


def _bucket_stats(results: List[VerificationResult], threshold: float) -> Dict[str, Any]:
    """Aggregate count / pass-rate / score-stats for a slice of results."""
    scored = [r for r in results if r.verification_score]
    scores = [r.verification_score["score"] for r in scored]
    n = len(results)
    n_scored = len(scored)
    n_passed = sum(1 for r in scored if r.passed)
    return {
        "n": n,
        "n_scored": n_scored,
        "n_passed": n_passed,
        "pass_rate": (n_passed / n_scored) if n_scored else 0.0,
        "mean": mean(scores) if scores else 0.0,
        "median": median(scores) if scores else 0.0,
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
    }


def _log_breakdowns(results: List[VerificationResult], threshold: float) -> None:
    """Log per-modality / per-generator-model / per-miner score breakdowns."""
    if not results:
        return

    overall = _bucket_stats(results, threshold)
    bt.logging.info(
        f"[VERIFY-SUMMARY] total={overall['n']} scored={overall['n_scored']} "
        f"passed={overall['n_passed']} pass_rate={overall['pass_rate']:.1%} "
        f"score(mean/med/min/max)="
        f"{overall['mean']:.3f}/{overall['median']:.3f}/"
        f"{overall['min']:.3f}/{overall['max']:.3f} "
        f"threshold={threshold:.3f}"
    )

    corrupted_count = 0
    embedding_duplicate_count = 0
    embedding_duplicate_by_miner: Dict[Tuple[Any, str], int] = defaultdict(int)
    for r in results:
        score = r.verification_score or {}
        if isinstance(score, dict):
            details = score.get("consensus_details") or {}
            if details.get("corrupted") or score.get("corrupted"):
                corrupted_count += 1
            if score.get("embedding_duplicate"):
                embedding_duplicate_count += 1
                m = r.media_entry
                embedding_duplicate_by_miner[(m.uid, (m.hotkey or "")[:8])] += 1

    if corrupted_count or embedding_duplicate_count:
        bt.logging.warning(
            f"[VERIFY-ABUSE-SUMMARY] corrupted={corrupted_count} "
            f"embedding_duplicates={embedding_duplicate_count}"
        )
        for key, count in sorted(
            embedding_duplicate_by_miner.items(),
            key=lambda kv: -kv[1],
        ):
            bt.logging.warning(
                f"[VERIFY-ABUSE-BY-MINER] key={key!r} embedding_duplicates={count}"
            )

    def _log_group(label: str, groups: Dict[Any, List[VerificationResult]]) -> None:
        # Sort by sample count descending so the most active buckets surface first.
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        for key, rs in ordered:
            s = _bucket_stats(rs, threshold)
            if s["n_scored"] == 0:
                bt.logging.info(
                    f"[VERIFY-{label}] key={key!r} n={s['n']} (no scored samples)"
                )
                continue
            bt.logging.info(
                f"[VERIFY-{label}] key={key!r} n={s['n']} scored={s['n_scored']} "
                f"passed={s['n_passed']} pass_rate={s['pass_rate']:.1%} "
                f"score(mean/med/min/max)="
                f"{s['mean']:.3f}/{s['median']:.3f}/"
                f"{s['min']:.3f}/{s['max']:.3f}"
            )

    by_modality: Dict[str, List[VerificationResult]] = defaultdict(list)
    by_gen_model: Dict[str, List[VerificationResult]] = defaultdict(list)
    by_miner: Dict[Tuple[Any, str], List[VerificationResult]] = defaultdict(list)

    for r in results:
        m = r.media_entry
        modality = m.modality.value if hasattr(m.modality, "value") else (m.modality or "unknown")
        by_modality[modality].append(r)
        by_gen_model[m.model_name or "(none)"].append(r)
        by_miner[(m.uid, (m.hotkey or "")[:8])].append(r)

    _log_group("BY-MODALITY", by_modality)
    _log_group("BY-GEN-MODEL", by_gen_model)
    _log_group("BY-MINER", by_miner)


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
            "errors": 0,
            "error_rate": 0.0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "average_score": 0.0,
        }

    total = len(results)
    successful = sum(1 for r in results if r.verification_score is not None)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if r.verification_score is not None and not r.passed)
    errors = sum(1 for r in results if r.verification_score is None)

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
        "successful": successful,  # Successfully processed (passed + failed)
        "passed": passed,          # Passed verification threshold
        "failed": failed,          # Failed verification threshold  
        "errors": errors,          # Errors during verification process
        "error_rate": errors / total if total > 0 else 0.0,
        "pass_rate": passed / successful if successful > 0 else 0.0,
        "fail_rate": failed / successful if successful > 0 else 0.0,
        "average_score": avg_score,
    }

