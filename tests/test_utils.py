"""
Tests for gas.utils.utils module.

Covers utility functions: get_file_modality, get_metadata, ExitContext,
run_in_thread, and the fail_with_none decorator.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gas.utils.utils import (
    get_file_modality,
    get_metadata,
    ExitContext,
    fail_with_none,
    run_in_thread,
)


class TestGetFileModality(unittest.TestCase):
    """Tests for get_file_modality."""

    def test_image_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            self.assertEqual(get_file_modality(f"test{ext}"), "image", f"Failed for {ext}")

    def test_video_extensions(self):
        for ext in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"]:
            self.assertEqual(get_file_modality(f"test{ext}"), "video", f"Failed for {ext}")

    def test_unknown_extension(self):
        self.assertEqual(get_file_modality("test.txt"), "file")
        self.assertEqual(get_file_modality("test.pdf"), "file")

    def test_case_insensitive(self):
        self.assertEqual(get_file_modality("test.JPG"), "image")
        self.assertEqual(get_file_modality("test.MP4"), "video")

    def test_path_with_directories(self):
        self.assertEqual(get_file_modality("/path/to/image.png"), "image")

    def test_no_extension(self):
        self.assertEqual(get_file_modality("noext"), "file")


class TestGetMetadata(unittest.TestCase):
    """Tests for get_metadata."""

    def test_returns_empty_when_no_json(self):
        result = get_metadata("/nonexistent/path/file.png")
        self.assertEqual(result, {})

    def test_reads_existing_json(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img:
            img_path = img.name
        json_path = os.path.splitext(img_path)[0] + ".json"
        try:
            with open(json_path, "w") as f:
                json.dump({"key": "value"}, f)
            result = get_metadata(img_path)
            self.assertEqual(result, {"key": "value"})
        finally:
            os.unlink(img_path)
            os.unlink(json_path)

    def test_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img:
            img_path = img.name
        json_path = os.path.splitext(img_path)[0] + ".json"
        try:
            with open(json_path, "w") as f:
                f.write("not valid json{{{")
            result = get_metadata(img_path)
            self.assertEqual(result, {})
        finally:
            os.unlink(img_path)
            os.unlink(json_path)


class TestExitContext(unittest.TestCase):
    """Tests for ExitContext."""

    def test_initial_state(self):
        ctx = ExitContext()
        self.assertFalse(ctx.isExiting)
        self.assertFalse(bool(ctx))

    def test_start_exit(self):
        ctx = ExitContext()
        ctx.startExit()
        self.assertTrue(ctx.isExiting)
        self.assertTrue(bool(ctx))

    def test_double_exit_raises_system_exit(self):
        ctx = ExitContext()
        ctx.startExit()
        with self.assertRaises(SystemExit):
            ctx.startExit()


class TestFailWithNone(unittest.TestCase):
    """Tests for fail_with_none decorator."""

    def test_successful_function(self):
        @fail_with_none("error msg")
        def good_func():
            return 42
        self.assertEqual(good_func(), 42)

    def test_failing_function_returns_none(self):
        @fail_with_none("error msg")
        def bad_func():
            raise ValueError("boom")
        self.assertIsNone(bad_func())

    def test_preserves_arguments(self):
        @fail_with_none()
        def add(a, b):
            return a + b
        self.assertEqual(add(1, 2), 3)


class TestRunInThread(unittest.TestCase):
    """Tests for run_in_thread."""

    def test_returns_result(self):
        result = run_in_thread(lambda: 42, timeout=5)
        self.assertEqual(result, 42)

    def test_timeout_raises(self):
        import time
        with self.assertRaises(TimeoutError):
            run_in_thread(lambda: time.sleep(10), timeout=1)


if __name__ == "__main__":
    unittest.main()
