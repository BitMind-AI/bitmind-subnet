"""
Miner media verification pipeline.

This module provides the main verification logic for miner-submitted media,
including caption generation, prompt retrieval, and similarity scoring.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import bittensor as bt
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from gas.cache.content_manager import ContentManager
from gas.cache.types import MediaEntry
from gas.generation.prompt_generator import PromptGenerator
from gas.types import Modality


_semantic_model_cache = {}


def calculate_semantic_similarity(
    text1: str, 
    text2: str, 
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Calculate semantic similarity between two texts using sentence transformers.
    
    Args:
        text1: First text (original prompt)
        text2: Second text (generated caption)
        model_name: Sentence transformer model name
        
    Returns:
        Dictionary with score details
    """
    try:
        # Lazy load and cache model
        if model_name not in _semantic_model_cache:
            _semantic_model_cache[model_name] = SentenceTransformer(model_name)
            bt.logging.debug(f"Loaded semantic similarity model: {model_name}")
        
        model = _semantic_model_cache[model_name]
        embeddings = model.encode([text1, text2])
        
        cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        score = max(0.0, min(1.0, float(cosine_sim)))
        
        return {
            "name": "semantic_similarity",
            "score": score,
            "weight": 1.0,
            "model_used": model_name,
            "embedding_dims": len(embeddings[0])
        }
        
    except Exception as e:
        bt.logging.error(f"Error in semantic similarity calculation: {e}")
        return {
            "name": "semantic_similarity", 
            "score": 0.0,
            "weight": 1.0,
            "error": str(e)
        }


class VerificationResult:    
    def __init__(
        self,
        media_entry: MediaEntry,
        original_prompt: Optional[str] = None,
        generated_caption: Optional[str] = None,
        verification_score: Optional[Dict[str, Any]] = None,
        passed: bool = False
    ):
        self.media_entry = media_entry
        self.original_prompt = original_prompt
        self.generated_caption = generated_caption
        self.verification_score = verification_score
        self.passed = passed


def generate_caption_for_media(
    prompt_generator: PromptGenerator,
    media_entry: MediaEntry
) -> Optional[str]:
    """
    Generate a caption for the given media entry.
    
    Args:
        prompt_generator: PromptGenerator instance
        media_entry: MediaEntry to generate caption for
        
    Returns:
        Generated caption or None if failed
    """
    try:
        if media_entry.modality == Modality.IMAGE:
            image_path = Path(media_entry.file_path)
            if not image_path.exists():
                bt.logging.error(f"Image file not found: {image_path}")
                return None
                
            image = Image.open(image_path)
            caption = prompt_generator.generate_prompt_from_image(
                image=image,
                intended_modality="image",
                max_new_tokens=50
            )
            return caption
            
        elif media_entry.modality == Modality.VIDEO:
            # TODO: Implement video caption generation
            bt.logging.info(f"Video caption generation not yet implemented for {media_entry.file_path}")
            return ""
            
        else:
            bt.logging.error(f"Unknown modality: {media_entry.modality}")
            return None
            
    except Exception as e:
        bt.logging.error(f"Error generating caption for {media_entry.file_path}: {e}")
        return None


def verify_single_media(
    content_manager: ContentManager,
    prompt_generator: PromptGenerator,
    media_entry: MediaEntry,
    threshold: float = 0.5
) -> VerificationResult:
    """
    Verify a single miner media entry.
    
    Args:
        content_manager: ContentManager instance
        prompt_generator: PromptGenerator instance  
        media_entry: MediaEntry to verify
        threshold: Threshold for determining pass/fail (default: 0.5)
        
    Returns:
        VerificationResult containing all verification data
    """
    try:
        # Step 1: Get original prompt
        if not media_entry.prompt_id:
            bt.logging.error(f"Media entry {media_entry.id} has no associated prompt_id")
            return VerificationResult(media_entry=media_entry, passed=False)
            
        original_prompt = content_manager.get_prompt_by_id(media_entry.prompt_id)
        if not original_prompt:
            bt.logging.error(f"Could not retrieve original prompt {media_entry.prompt_id}")
            return VerificationResult(media_entry=media_entry, passed=False)
        
        # Step 2: Generate caption
        generated_caption = generate_caption_for_media(prompt_generator, media_entry)
        if not generated_caption:
            bt.logging.error(f"Could not generate caption for media {media_entry.id}")
            return VerificationResult(
                media_entry=media_entry,
                original_prompt=original_prompt,
                passed=False
            )
        
        # Step 3: Calculate verification score
        verification_score = calculate_semantic_similarity(
            text1=original_prompt,
            text2=generated_caption,
        )

        ### TODO add other metrics
        
        # Determine pass/fail based on threshold
        score = verification_score.get('score', 0.0)
        passed = score >= threshold
        verification_score['passed'] = passed
        verification_score['threshold'] = threshold
        
        bt.logging.debug(
            f"Verified media {media_entry.id}: "
            f"score={score:.3f}, passed={passed}"
        )
        
        return VerificationResult(
            media_entry=media_entry,
            original_prompt=original_prompt,
            generated_caption=generated_caption,
            verification_score=verification_score,
            passed=passed
        )
        
    except Exception as e:
        bt.logging.error(f"Error verifying media {media_entry.id}: {e}")
        return VerificationResult(media_entry=media_entry, passed=False)


def run_verification_batch(
    content_manager: ContentManager,
    prompt_generator: PromptGenerator,
    batch_size: int = 10,
    threshold: float = 0.5
) -> List[VerificationResult]:
    """
    Run verification on a batch of unverified miner media.
    
    Args:
        content_manager: ContentManager instance
        prompt_generator: PromptGenerator instance
        batch_size: Maximum number of items to process in this batch
        threshold: Threshold for determining pass/fail (default: 0.5)
        
    Returns:
        List of VerificationResult objects
    """
    bt.logging.info(f"Starting verification batch (batch_size={batch_size})")
    
    # Get unverified media
    unverified_media = content_manager.get_unverified_miner_media()
    bt.logging.info(f"Found {len(unverified_media)} unverified miner media entries")
    if not unverified_media:
        bt.logging.info("No unverified miner media found")
        return []
        
    # Process batch
    batch = unverified_media[:batch_size]
    results = []
    
    for i, media_entry in enumerate(batch, 1):
        bt.logging.debug(f"Processing media {i}/{len(batch)}: {media_entry.file_path}")
        
        result = verify_single_media(
            content_manager=content_manager,
            prompt_generator=prompt_generator,
            media_entry=media_entry,
            threshold=threshold
        )
        
        results.append(result)
        
        if result.passed and result.verification_score:
            try:
                content_manager.mark_miner_media_verified(media_entry.id)
                bt.logging.debug(f"Marked media {media_entry.id} as verified")
            except Exception as e:
                bt.logging.error(f"Error marking media as verified: {e}")
    
    successful_verifications = sum(1 for r in results if r.verification_score is not None)
    passed_verifications = sum(1 for r in results if r.passed)
    
    bt.logging.info(
        f"Verification batch complete: {successful_verifications}/{len(results)} successful, "
        f"{passed_verifications}/{len(results)} passed verification"
    )
    
    return results


def get_verification_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "successful": 0, "passed": 0, "failed": 0, "error_rate": 0.0}
    
    total = len(results)
    successful = sum(1 for r in results if r.verification_score is not None)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if r.verification_score is None)
    
    scores = [
        r.verification_score.get('score', 0)
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
        "average_score": avg_score
    }


def clear_model_cache():
    global _semantic_model_cache
    _semantic_model_cache.clear()
    bt.logging.debug("Cleared semantic similarity model cache")

