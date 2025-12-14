import os
import traceback
import time

from neurons.generator.services.runway_service import RunwayService, Models
from neurons.generator.task_manager import TaskManager

# Set API key if one isn't already set
os.environ.setdefault(
    "RUNWAYML_API_SECRET",
    "your-api-key-here"  # replace with your test key
)


def save_video(video_bytes, filename):
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{filename}"
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    return out_path


def validate_video(video_bytes):
    """Basic validation that bytes look like a video file."""
    # MP4 files start with ftyp box
    if len(video_bytes) < 12:
        return False
    # Check for MP4 signature (ftyp at offset 4)
    if video_bytes[4:8] == b'ftyp':
        return True
    # Check for WebM signature
    if video_bytes[:4] == b'\x1a\x45\xdf\xa3':
        return True
    return False


def run_model_test(service, manager, model, duration=5):
    print(f"\n=== Running generation test for model: {model}, duration: {duration}s ===")

    # gen3a_turbo requires image_to_video, so we provide a test image URL
    task_id = manager.create_task(
        modality="video",
        prompt="A timelapse on a sunny day with clouds flying by",
        parameters={
            "model": model,
            "duration": duration,
            "width": 1280,
            "height": 768,
            "input_image_url": "https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_(cropped).jpg",
        },
        webhook_url=None,
        signed_by="test-suite"
    )

    task = manager.get_task(task_id)

    try:
        start_time = time.time()
        result = service.process(task)
        elapsed = time.time() - start_time

        video_bytes = result["data"]
        meta = result["metadata"]

        print(f"✔ Generated video in {elapsed:.2f}s ({len(video_bytes)/1024:.1f} KB)")
        print(f"✔ Metadata keys: {list(meta.keys())}")

        # Save output
        out = save_video(video_bytes, f"{model.replace('.', '_')}_{duration}s.mp4")
        print(f"✔ Saved to {out}")

        # Validate video
        assert validate_video(video_bytes), "Video failed basic validation"

        # Validate metadata
        assert meta["model"] == model
        assert meta["provider"] == "runway"
        assert meta["duration"] == duration
        assert meta["generation_time"] > 0

        print(f"=== Model {model} PASSED ===")

    except Exception:
        print(f"=== Model {model} FAILED ===")
        print(traceback.format_exc())


def test_invalid_api_key():
    print("\n=== Testing invalid API key ===")
    original_key = os.environ.get("RUNWAYML_API_SECRET")
    os.environ["RUNWAYML_API_SECRET"] = "invalid-key"

    service = RunwayService()
    manager = TaskManager()

    task_id = manager.create_task(
        modality="video",
        prompt="test prompt",
        parameters={"model": Models.GEN3A_TURBO},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        result = service.process(task)
        # If we get here, the API call unexpectedly succeeded
        raise AssertionError("❌ Should have failed with invalid API key!")
    except AssertionError:
        # Re-raise AssertionError so test properly fails
        raise
    except Exception as e:
        # Expected: API should reject invalid key
        print(f"✔ Correctly failed: {e}")
    finally:
        # Restore original environment
        if original_key:
            os.environ["RUNWAYML_API_SECRET"] = original_key
        elif "RUNWAYML_API_SECRET" in os.environ:
            del os.environ["RUNWAYML_API_SECRET"]


def test_invalid_modality(service, manager):
    print("\n=== Testing invalid modality ===")

    task_id = manager.create_task(
        modality="image",  # RunwayService only supports video
        prompt="test prompt",
        parameters={"model": Models.GEN3A_TURBO},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        raise AssertionError("❌ Should have failed with invalid modality!")
    except ValueError as e:
        print(f"✔ Correctly rejected image modality: {e}")


def test_aspect_ratio_conversion(service):
    print("\n=== Testing aspect ratio conversion ===")

    # Test landscape
    ratio = service._get_aspect_ratio(1920, 1080)
    assert ratio == "1280:768", f"Expected 1280:768, got {ratio}"
    print("✔ 1920x1080 -> 1280:768")

    # Test portrait
    ratio = service._get_aspect_ratio(1080, 1920)
    assert ratio == "768:1280", f"Expected 768:1280, got {ratio}"
    print("✔ 1080x1920 -> 768:1280")

    # Test square (defaults to landscape)
    ratio = service._get_aspect_ratio(1024, 1024)
    assert ratio == "1280:768", f"Expected 1280:768, got {ratio}"
    print("✔ 1024x1024 -> 1280:768")


def test_service_info(service):
    print("\n=== Testing service info ===")

    info = service.get_service_info()
    assert info["name"] == "Runway"
    assert info["type"] == "api"
    assert "video_generation" in info["supported_tasks"]["video"]
    print(f"✔ Service info: {info}")


def test_model_env_validation():
    """Test that invalid RUNWAY_MODEL env values fallback with warning."""
    print("\n=== Testing model env validation ===")
    
    import os
    from neurons.generator.services.runway_service import RunwayService, Models, MODEL_INFO
    
    # Test valid model names
    assert Models.GEN3A_TURBO in MODEL_INFO, "gen3a_turbo should be in MODEL_INFO"
    assert Models.GEN4_5 in MODEL_INFO, "gen4.5 should be in MODEL_INFO"
    print(f"✔ Valid models: {list(MODEL_INFO.keys())}")
    
    # Test that invalid model falls back to default
    original = os.environ.get("RUNWAY_MODEL")
    os.environ["RUNWAY_MODEL"] = "invalid_model_name"
    
    test_service = RunwayService()
    assert test_service.default_model == Models.GEN3A_TURBO, \
        f"Invalid model should fallback to GEN3A_TURBO, got {test_service.default_model}"
    print("✔ Invalid model 'invalid_model_name' correctly falls back to gen3a_turbo")
    
    # Test that valid model is accepted
    os.environ["RUNWAY_MODEL"] = "gen3a_turbo"
    test_service = RunwayService()
    assert test_service.default_model == "gen3a_turbo", \
        f"Expected gen3a_turbo, got {test_service.default_model}"
    print("✔ Valid model 'gen3a_turbo' correctly accepted")
    
    # Restore original
    if original:
        os.environ["RUNWAY_MODEL"] = original
    elif "RUNWAY_MODEL" in os.environ:
        del os.environ["RUNWAY_MODEL"]
    
    print("✔ Model env validation passed")


def run_full_test_suite():
    print("\n========== Runway Service Full Test Suite ==========\n")

    service = RunwayService()
    manager = TaskManager()

    # Unit tests (no API calls)
    test_aspect_ratio_conversion(service)
    test_service_info(service)
    test_invalid_modality(service, manager)
    test_model_env_validation()

    if not service.is_available():
        print("\n❌ API key missing — skipping API tests")
        print("Set RUNWAYML_API_SECRET to run full tests")
        return

    # API tests (requires valid key)
    print("\n--- Running API tests (requires valid RUNWAYML_API_SECRET) ---")

    # Test Gen-3 Alpha Turbo with 5s video
    run_model_test(service, manager, Models.GEN3A_TURBO, duration=5)

    # Negative tests
    test_invalid_api_key()

    print("\n========== All Tests Completed ==========\n")


if __name__ == "__main__":
    run_full_test_suite()
