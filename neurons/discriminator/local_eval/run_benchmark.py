import argparse
import asyncio
import os
import time
from typing import Dict
import logging
import bittensor as bt
import sys

logger = bt.logging

try:
    bt.logging.setLevel(logging.INFO)
    if not bt.logging.handlers:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        bt.logging.addHandler(stdout_handler)
except Exception:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

ACCURACY_THRESHOLD = .8


async def run_model_exam(
    file_hash: str,
    hotkey: str,
    image_model_path: str = None,
    video_model_path: str = None,
    verbosity: int = 0,
    # prune_old_data removed
    stream_images: bool = True,
    temp_dir: str = None,
    max_samples: int | None = None,
) -> Dict:
    """Run a model examination for image and/or video detectors.

    This function tests the provided image and/or video models against the BitMind datasets
    and returns the results including pass/fail status and various metrics.

    Args:
        file_hash (str): Unique identifier for the model file.
        hotkey (str): Identifier for the miner running the test.
        image_model_path (str, optional): Path to the image detector ONNX model. Defaults to None.
        video_model_path (str, optional): Path to the video detector ONNX model. Defaults to None.
        verbosity (int, optional): Level of verbosity for logging. Defaults to 0.
        prune_old_data removed.
        stream_images (bool, optional): Whether to stream images instead of downloading. Defaults to True.
        temp_dir (str, optional): Directory for temporary video extraction. Defaults to None.

    Returns:
        Dict: A dictionary containing the exam results, including pass/fail status and metrics.
    """
    # Delayed imports so HF_HOME/TMPDIR can be set in main() before these modules load
    from .inference import (
        run_image_inference as test_image_inference,
        run_video_inference as test_video_inference,
    )
    from .data import load_latest_image_dataset, load_latest_video_dataset

    import onnxruntime as ort
    
    exam_results = {
        "hotkey": hotkey,
        "file_hash": file_hash,
        "timestamp": time.time(),
        "passed": False,
        "validation": {},
        "image_results": {},
        "video_results": {},
        "errors": [],
        "metrics": {},
        "image_model_path": image_model_path,
        "video_model_path": video_model_path,
    }
    
    start_time = time.time()
    
    try:
        # Step 2: Load detector models for inference
        logger.info("🔧 Loading detector models for inference testing...")
        
        # Determine which benchmarks to run based on provided paths
        run_image = bool(image_model_path)
        run_video = bool(video_model_path)
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # Create session options for faster loading
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            
            # Load image model
            image_session = None
            if image_model_path and os.path.exists(image_model_path) and run_image:
                logger.info(f"📦 Loading image detector...")
                image_load_start = time.time()
                image_session = ort.InferenceSession(image_model_path, sess_options=sess_options, providers=providers)
                image_load_time = time.time() - image_load_start
                logger.info(f"✅ Loaded image detector in {image_load_time:.2f} seconds")
            
            # Load video model  
            video_session = None
            if video_model_path and os.path.exists(video_model_path) and run_video:
                logger.info(f"📦 Loading video detector (this may take 30-60s for large models)...")
                video_load_start = time.time()
                video_session = ort.InferenceSession(video_model_path, sess_options=sess_options, providers=providers)
                video_load_time = time.time() - video_load_start
                logger.info(f"✅ Loaded video detector in {video_load_time:.2f} seconds")
            
            if not image_session and not video_session:
                raise Exception("No models could be loaded")
            
            # Get model specs from first available session for validation
            session = image_session or video_session
            input_specs = session.get_inputs()
            output_specs = session.get_outputs()
            
            exam_results["validation"]["input_shape"] = str(input_specs[0].shape)
            exam_results["validation"]["input_type"] = str(input_specs[0].type)
            exam_results["validation"]["output_shape"] = str(output_specs[0].shape)
            
        except Exception as e:
            logger.error(f"❌ Failed to load model for inference: {e}")
            exam_results["errors"].append(f"ONNX runtime error: {str(e)}")
            exam_results["passed"] = False  # ONNX runtime errors = exam failed
            #await save_exam_results(cache_dir, exam_results)
            return exam_results
        
        # Step 3: Test on image dataset
        if run_image and image_session:
            logger.info("🖼️ Testing on image dataset with image_detector...")
            image_dataset, image_split = load_latest_image_dataset(streaming=stream_images)
            if image_dataset is None or image_split is None:
                logger.error("Image dataset discovery failed")
                exam_results["image_results"]["error"] = "No dataset available"
            else:
                # prune removed
                image_accuracy = await test_image_inference(
                    image_session, image_session.get_inputs(), exam_results, dataset=image_dataset, dataset_split=image_split, verbosity=verbosity, max_samples=max_samples
                )
                exam_results["image_results"]["accuracy"] = image_accuracy
        
        # Step 4: Test on video dataset  
        if run_video and video_session:
            logger.info("🎬 Testing on video dataset with video_detector...")
            video_dataset, video_config = load_latest_video_dataset(streaming=True, split='train')
            # prune removed
            video_accuracy = await test_video_inference(
                video_session, video_session.get_inputs(), exam_results, dataset=video_dataset, dataset_config=video_config, verbosity=verbosity, temp_dir=temp_dir, max_samples=max_samples
            )
            exam_results["video_results"]["accuracy"] = video_accuracy
        
        # Step 5: Calculate overall results
        accuracies = []
        if "accuracy" in exam_results["image_results"]:
            accuracies.append(exam_results["image_results"]["accuracy"])
        if "accuracy" in exam_results["video_results"]:
            accuracies.append(exam_results["video_results"]["accuracy"])
        
        if accuracies:
            overall_accuracy = sum(accuracies) / len(accuracies)
            exam_results["metrics"]["overall_accuracy"] = overall_accuracy
            exam_results["passed"] = overall_accuracy >= ACCURACY_THRESHOLD
            
            logger.info(f"📊 Overall accuracy: {overall_accuracy:.2%}")
            if exam_results['passed']:
                logger.info("✅ Exam PASSED")
            else:
                logger.info("❌ Exam FAILED")
        else:
            exam_results["errors"].append("No tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Exam failed with error: {e}")
        exam_results["errors"].append(f"Exam error: {str(e)}")
    
    finally:
        exam_results["metrics"]["exam_duration_seconds"] = time.time() - start_time

        # Save exam results to model volume
        #await save_exam_results(cache_dir, exam_results)
        
    return exam_results


def main():
    parser = argparse.ArgumentParser(description="Run image/video detector benchmarks against BitMind datasets")
    parser.add_argument("--image_model", type=str, default=None, help="Path to image detector ONNX model")
    parser.add_argument("--video_model", type=str, default=None, help="Path to video detector ONNX model")
    parser.add_argument("--hotkey", type=str, default="local", help="Identifier for the run")
    parser.add_argument("--file_hash", type=str, default="local", help="Model file hash identifier")
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv)")
    # prune-old-data option removed
    # Stream images by default; allow disabling with --no-stream-images
    parser.add_argument("--no-stream-images", dest="stream_images", action="store_false", help="Disable streaming images; download locally")
    parser.set_defaults(stream_images=True)
    parser.add_argument("--hf-home", type=str, default=None, help="Override Hugging Face cache root (sets HF_HOME)")
    parser.add_argument("--temp-dir", type=str, default=None, help="Directory for temporary video extraction (overrides TMPDIR)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate per modality (use all if omitted)")
    args = parser.parse_args()

    if not args.image_model and not args.video_model:
        parser.error("At least one of --image_model or --video_model must be provided")

    # Optionally override Hugging Face cache root
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    # Optionally override TMPDIR for temporary files (extraction, temp downloads)
    if args.temp_dir:
        try:
            os.makedirs(args.temp_dir, exist_ok=True)
        except Exception:
            pass
        os.environ["TMPDIR"] = args.temp_dir

    # Prepare minimal exam results envelope
    exam_results = {
        "hotkey": args.hotkey,
        "file_hash": args.file_hash,
        "timestamp": time.time(),
        "passed": False,
        "validation": {},
        "image_results": {},
        "video_results": {},
        "errors": [],
        "metrics": {},
        "image_model_path": args.image_model,
        "video_model_path": args.video_model,
    }

    async def _run():
        results = await run_model_exam(
            file_hash=args.file_hash,
            hotkey=args.hotkey,
            image_model_path=args.image_model,
            video_model_path=args.video_model,
            verbosity=args.v,
            # prune removed
            stream_images=args.stream_images,
            temp_dir=args.temp_dir,
            max_samples=args.max_samples,
        )
        return results

    results = asyncio.run(_run())
    # Simple stdout print for CLI usage
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()