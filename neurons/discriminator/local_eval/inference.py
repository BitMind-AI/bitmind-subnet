import os
import time
from typing import Dict, Optional

import bittensor as bt
import cv2
import numpy as np
from PIL import Image

from .data import video_payload



async def run_image_inference(
    session,
    input_specs,
    exam_results: Dict,
    dataset,
    dataset_split,
    verbosity: int = 0,
    max_samples: Optional[int] = None
) -> float:
    """Test model on image dataset."""
    
    try:
        # Require dataset to be provided by caller
        if dataset is None or dataset_split is None:
            bt.logging.error("Image dataset and split must be provided by caller")
            exam_results["image_results"]["error"] = "Dataset not provided"
            return 0.0
        
        # Test on samples
        correct = 0
        total = 0
        inference_times = []
        
        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:  # Limit number of evaluated images
                break
            
            try:
                # Get image and label
                image = sample.get('media_image')
                label = sample.get('label')
                model_name = sample.get('model_name', 'unknown')
                
                if image is None or label is None:
                    continue
                
                # Prepare image for inference (keep as uint8, no normalization)
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                    image = image.resize((224, 224))  # Standard size
                    image_array = np.array(image, dtype=np.uint8)  # Keep as uint8
                    image_array = np.transpose(image_array, (2, 0, 1))  # CHW format
                    image_array = np.expand_dims(image_array, 0)  # Add batch dimension
                
                # Run inference
                start = time.time()
                outputs = session.run(None, {input_specs[0].name: image_array})
                inference_time = (time.time() - start) * 1000  # ms
                inference_times.append(inference_time)
                
                # Get prediction - raw logits from model
                logits = outputs[0]
                
                # Apply softmax to convert logits to probabilities
                exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                
                # Handle 3-class output (most common for these models)
                if probabilities.shape[-1] == 3:
                    # Output is [batch, 3] where classes are typically [Human, AI, Unknown/Mixed]
                    # or [AI, Human, Unknown] - need to check which is which
                    pred_probs = probabilities[0]  # Get first batch item
                    
                    # Verbosity-controlled logging
                    if verbosity >= 3:
                        bt.logging.info(f"  Image {i+1} raw logits: {logits[0].tolist()}")
                        bt.logging.info(f"  Image {i+1} probabilities: {pred_probs.tolist()}")
                        bt.logging.info(f"  Image {i+1} shape: {probabilities.shape}, Argmax: {np.argmax(pred_probs)}")
                    elif verbosity == 2:
                        bt.logging.info(f"  Image {i+1} probabilities: {pred_probs.tolist()}")
                    
                    predicted_class = int(np.argmax(pred_probs))
                    # If class 0 has highest prob, it's likely human (1), else AI (0)
                    # This matches the inference output pattern
                    predicted_class = 1 if predicted_class == 0 else 0
                elif probabilities.shape[-1] == 2:  # Binary classification
                    pred_probs = probabilities[0]
                    if verbosity >= 3:
                        bt.logging.info(f"  Image {i+1} raw logits (binary): {logits[0].tolist()}")
                        bt.logging.info(f"  Image {i+1} probabilities (binary): {pred_probs.tolist()}")
                    elif verbosity == 2:
                        bt.logging.info(f"  Image {i+1} probabilities (binary): {pred_probs.tolist()}")
                    predicted_class = int(np.argmax(pred_probs))
                else:  # Single output
                    if verbosity >= 2:
                        bt.logging.info(f"  Image {i+1} raw output (single): {probabilities.flatten().tolist()}")
                    predicted_class = int(probabilities.flatten()[0] > 0.5)
                
                # Convert label to int (-1 -> 0 for AI, 1 -> 1 for human)
                true_label = 0 if int(label) == -1 else 1
                
                is_correct = predicted_class == true_label
                if is_correct:
                    correct += 1
                total += 1
                
                # Verbose output for each image sample
                if verbosity >= 1:
                    result_symbol = "✅" if is_correct else "❌"
                    if max_samples is not None:
                        bt.logging.info(f"{result_symbol} Image {i+1}/{max_samples}: {'CORRECT' if is_correct else 'WRONG'} (True={true_label}, Pred={predicted_class}, Model={model_name})")
                    else:
                        bt.logging.info(f"{result_symbol} Image {i+1}: {'CORRECT' if is_correct else 'WRONG'} (True={true_label}, Pred={predicted_class}, Model={model_name})")
                
            except Exception as e:
                bt.logging.warning(f"Failed to process image sample {i}: {e}")
                exam_results["errors"].append(f"Image inference error: {str(e)[:100]}")
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        exam_results["image_results"] = {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "avg_inference_time_ms": avg_inference_time,
            "dataset_split": dataset_split
        }
        
        bt.logging.info(f"📊 Image accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy
        
    except Exception as e:
        bt.logging.error(f"❌ Image testing failed: {e}")
        exam_results["image_results"]["error"] = str(e)
        return 0.0


async def run_video_inference(
    session,
    input_specs,
    exam_results: Dict,
    dataset,
    dataset_config,
    verbosity: int = 0,
    temp_dir: str = None,
    max_samples: Optional[int] = None
) -> float:
    """Test model on video dataset."""
    
    try:
        if dataset is None or dataset_config is None:
            bt.logging.error("Video dataset and config must be provided by caller")
            exam_results["video_results"]["error"] = "Dataset not provided"
            return 0.0
        
        if verbosity >= 1:
            bt.logging.info(f"📥 Preparing video payload for {dataset_config}...")
        with video_payload(temp_dir=temp_dir, dataset_config=dataset_config) as temp_dir:
            
            correct = 0
            total = 0
            inference_times = []
            for i, sample in enumerate(dataset):
                if max_samples is not None and i >= max_samples:
                    break
                
                try:
                    video_ref = sample['video']
                    filename = video_ref.split('/')[-1]
                    video_path = os.path.join(temp_dir, filename)
                    label = sample.get('label')
                    model_name = sample.get('model_name', 'unknown')
                    
                    if not os.path.exists(video_path) or label is None:
                        continue
                    
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    frame_count = 0
                    max_frames = 24
                    
                    while cap.isOpened() and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize frame (keep as uint8, no normalization)
                        frame = cv2.resize(frame, (224, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Keep as uint8, don't normalize
                        frames.append(frame)
                        frame_count += 1
                    
                    cap.release()
                    
                    if not frames:
                        continue
                    
                    # Convert frames to 5D tensor for video model [1, T, C, H, W]
                    # Stack frames: list of [H, W, C] -> [T, H, W, C]
                    video_array = np.stack(frames, axis=0)
                    # Transpose to [T, C, H, W]
                    video_array = np.transpose(video_array, (0, 3, 1, 2))
                    # Add batch dimension to get [1, T, C, H, W]
                    video_array = np.expand_dims(video_array, 0)
                    # Ensure uint8 type
                    video_array = video_array.astype(np.uint8)
                    
                    if verbosity >= 3:
                        bt.logging.info(f"Video tensor shape: {video_array.shape}, dtype: {video_array.dtype}")
                    
                    # Run inference on entire video
                    start = time.time()
                    outputs = session.run(None, {input_specs[0].name: video_array})
                    inference_time = (time.time() - start) * 1000
                    inference_times.append(inference_time)
                    
                    # Get prediction - raw logits from model
                    logits = outputs[0]
                    
                    # Apply softmax to convert logits to probabilities
                    exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                    
                    # Handle 3-class output
                    if probabilities.shape[-1] == 3:
                        pred_probs = probabilities[0]  # Get first batch item
                        
                        # Verbosity-controlled logging
                        if verbosity >= 3:
                            bt.logging.info(f"  Video {i+1} raw logits: {logits[0].tolist()}")
                            bt.logging.info(f"  Video {i+1} probabilities: {pred_probs.tolist()}")
                            bt.logging.info(f"  Video {i+1} shape: {probabilities.shape}, Argmax: {np.argmax(pred_probs)}")
                        elif verbosity == 2:
                            bt.logging.info(f"  Video {i+1} probabilities: {pred_probs.tolist()}")
                        
                        final_prediction = int(np.argmax(pred_probs))
                        # Map to binary: class 0 -> human (1), others -> AI (0)
                        final_prediction = 1 if final_prediction == 0 else 0
                    elif probabilities.shape[-1] == 2:
                        pred_probs = probabilities[0]
                        if verbosity >= 3:
                            bt.logging.info(f"  Video {i+1} raw logits (binary): {logits[0].tolist()}")
                            bt.logging.info(f"  Video {i+1} probabilities (binary): {pred_probs.tolist()}")
                        elif verbosity == 2:
                            bt.logging.info(f"  Video {i+1} probabilities (binary): {pred_probs.tolist()}")
                        final_prediction = int(np.argmax(pred_probs))
                    else:
                        if verbosity >= 2:
                            bt.logging.info(f"  Video {i+1} raw output (single): {probabilities.flatten().tolist()}")
                        final_prediction = int(probabilities.flatten()[0] > 0.5)
                    
                    
                    true_label = 0 if int(label) == -1 else 1
                    
                    is_correct = final_prediction == true_label
                    if is_correct:
                        correct += 1
                    
                    if verbosity >= 1:
                        result_symbol = "✅" if is_correct else "❌"
                        if max_samples is not None:
                            bt.logging.info(f"{result_symbol} Video {i+1}/{max_samples}: {'CORRECT' if is_correct else 'WRONG'} (True={true_label}, Pred={final_prediction}, Frames={len(frames)}, Model={model_name})")
                        else:
                            bt.logging.info(f"{result_symbol} Video {i+1}: {'CORRECT' if is_correct else 'WRONG'} (True={true_label}, Pred={final_prediction}, Frames={len(frames)}, Model={model_name})")
                    total += 1
                    
                except Exception as e:
                    bt.logging.warning(f"Failed to process video sample {i}: {e}")
                    exam_results["errors"].append(f"Video inference error: {str(e)[:100]}")
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0.0
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            
            exam_results["video_results"] = {
                "accuracy": accuracy,
                "total_samples": total,
                "correct_predictions": correct,
                "avg_inference_time_ms": avg_inference_time,
                "dataset_config": dataset_config
            }
            
            bt.logging.info(f"📊 Video accuracy: {accuracy:.2%} ({correct}/{total})")
            return accuracy
    
    except Exception as e:
        bt.logging.error(f"❌ Video testing failed: {e}")
        exam_results["video_results"]["error"] = str(e)
        return 0.0