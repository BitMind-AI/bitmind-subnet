import os
import traceback
from PIL import Image
import io
import time

from neurons.generator.services.replicate_service import ReplicateService, Models
from neurons.generator.task_manager import TaskManager

# Note: Set REPLICATE_API_TOKEN in your environment before running tests


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


def run_model_test(service, manager, model):
    print(f"\n=== Running generation test for model: {model} ===")

    task_id = manager.create_task(
        modality="image",
        prompt="A neon cyberpunk city with flying cars",
        parameters={
            "model": model,
            "aspect_ratio": "16:9",
            "seed": 777,
        },
        webhook_url=None,
        signed_by="test-suite"
    )

    task = manager.get_task(task_id)

    try:
        start_time = time.time()
        result = service.process(task)
        elapsed = time.time() - start_time

        img_bytes = result["data"]
        meta = result["metadata"]

        print(f"✔ Generated image in {elapsed:.2f}s ({len(img_bytes)/1024:.1f} KB)")
        print(f"✔ Metadata keys: {list(meta.keys())}")

        out = save_image(img_bytes, f"replicate_{model.replace('.', '_')}.png")
        print(f"✔ Saved to {out}")

        assert validate_image(img_bytes), "Image failed Pillow validation"

        assert meta["model"] == model
        assert meta["provider"] == "replicate"
        assert meta["generation_time"] > 0
        assert "prediction_id" in meta

        print(f"=== Model {model} PASSED ===")

    except Exception:
        print(f"=== Model {model} FAILED ===")
        print(traceback.format_exc())


def test_invalid_api_key():
    print("\n=== Testing invalid API key ===")
    os.environ["REPLICATE_API_TOKEN"] = "invalid-token"

    service = ReplicateService()
    manager = TaskManager()

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": Models.FLUX_SCHNELL},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        raise AssertionError("❌ Should have failed with invalid API token!")
    except AssertionError:
        raise
    except Exception as e:
        print(f"✔ Correctly failed: {e}")


def test_invalid_model(service, manager):
    print("\n=== Testing invalid model ===")

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": "invalid-model"},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        raise AssertionError("❌ Should have failed with invalid model!")
    except ValueError as e:
        print(f"✔ Correctly raised ValueError: {e}")


def test_sdxl_with_negative_prompt(service, manager):
    print("\n=== Testing SDXL with negative prompt ===")

    task_id = manager.create_task(
        modality="image",
        prompt="A beautiful sunset over mountains",
        parameters={
            "model": Models.SDXL,
            "negative_prompt": "low quality, blurry, distorted",
            "width": 1024,
            "height": 1024,
        },
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    result = service.process(task)
    assert result["data"] is not None
    assert result["metadata"]["model"] == Models.SDXL
    print("✔ SDXL with negative prompt succeeded")


def run_full_test_suite():
    print("\n========== Replicate Full Test Suite ==========\n")

    os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")

    service = ReplicateService()
    manager = TaskManager()

    if not service.is_available():
        print("❌ API token missing — cannot run tests")
        return

    for model in [Models.FLUX_SCHNELL, Models.FLUX_DEV]:
        run_model_test(service, manager, model)

    test_sdxl_with_negative_prompt(service, manager)
    test_invalid_model(service, manager)
    test_invalid_api_key()

    print("\n========== All Tests Completed ==========\n")


if __name__ == "__main__":
    run_full_test_suite()
