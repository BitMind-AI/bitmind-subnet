import os
import traceback
from PIL import Image
import io
import time
import requests
from unittest.mock import MagicMock, patch

import sys
from unittest.mock import MagicMock

# Mock bittensor before importing services
mock_bt = MagicMock()
sys.modules["bittensor"] = mock_bt

from neurons.generator.services.fal_service import FalAIService
from neurons.generator.task_manager import TaskManager
from gas.verification.c2pa_verification import verify_c2pa

# Set API key if one isn't already set
os.environ.setdefault(
    "FAL_KEY",
    "key-YOUR-API-KEY"  # replace with your test key
)

def save_image(img_bytes, filename):
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{filename}"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path

def validate_image(img_bytes):
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:
        return False


def run_model_test(service, manager, model, modality="image"):
    print(f"\n=== Running generation test for model: {model} ({modality}) ===")

    task_id = manager.create_task(
        modality=modality,
        prompt="A futuristic city with flying cars",
        parameters={
            "model": model,
            "seed": 777,
            "duration": "5" if modality == "video" else None
        },
        webhook_url=None,
        signed_by="test-suite"
    )

    task = manager.get_task(task_id)

    try:
        start_time = time.time()
        result = service.process(task)
        elapsed = time.time() - start_time

        data_bytes = result["data"]
        meta = result["metadata"]

        print(f"✔ Generated media in {elapsed:.2f}s ({len(data_bytes)/1024:.1f} KB)")
        print(f"✔ Metadata keys: {list(meta.keys())}")

        # Save output
        ext = "mp4" if modality == "video" else "png"
        out = save_image(data_bytes, f"{model.replace('/', '_')}.{ext}")
        print(f"✔ Saved to {out}")

        if modality == "image":
            # Validate image integrity
            assert validate_image(data_bytes), "Image failed Pillow validation"
            
            # Check C2PA
            try:
                c2pa_result = verify_c2pa(data_bytes)
                if c2pa_result.verified:
                    print(f"✅ C2PA Verified! Issuer: {c2pa_result.issuer}")
                else:
                    print(f"⚠️ C2PA Not Verified: {c2pa_result.error}")
            except Exception as e:
                print(f"⚠️ C2PA Check Failed: {e}")

        # Validate metadata
        assert meta["model"] == model
        assert meta["provider"] == "fal.ai"
        assert "source_url" in meta

        print(f"=== Model {model} PASSED ===")

    except Exception:
        print(f"=== Model {model} FAILED ===")
        print(traceback.format_exc())


def test_invalid_api_key():
    print("\n=== Testing invalid API key ===")
    original_key = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = "invalid-key"

    service = FalAIService()
    manager = TaskManager()

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": "fal-ai/flux/dev"},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        print("❌ Should have failed with invalid API key!")
    except Exception as e:
        print(f"✔ Correctly failed: {e}")
    finally:
        if original_key:
            os.environ["FAL_KEY"] = original_key


def run_full_test_suite():
    print("\n========== Fal.ai Full Test Suite ==========\n")

    service = FalAIService()
    manager = TaskManager()

    if not service.is_available() or service.api_key == "key-YOUR-API-KEY":
        print("❌ API key missing or default — skipping live tests")
        print("ℹ️  Set FAL_KEY environment variable to run live tests")
    else:
        # Test Image Model
        run_model_test(service, manager, "fal-ai/flux/dev", "image")
        
        # Test Video Model
        run_model_test(service, manager, "fal-ai/kling-video/v1/standard/text-to-video", "video")

    # Negative tests
    test_invalid_api_key()

    print("\n========== All Tests Completed ==========\n")


if __name__ == "__main__":
    run_full_test_suite()
