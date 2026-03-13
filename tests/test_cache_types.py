"""
Tests for gas.cache.types module.

Covers PromptEntry, MediaEntry, VerificationResult dataclasses
and their serialization behavior.
"""

import time
import unittest

from gas.cache.types import PromptEntry, MediaEntry, VerificationResult, Media
from gas.types import Modality, MediaType, SourceType


class TestPromptEntry(unittest.TestCase):
    """Tests for PromptEntry dataclass."""

    def test_to_dict(self):
        entry = PromptEntry(
            id="p1",
            content="a cat",
            content_type="prompt",
            created_at=1000.0,
        )
        d = entry.to_dict()
        self.assertEqual(d["id"], "p1")
        self.assertEqual(d["content"], "a cat")
        self.assertEqual(d["used_count"], 0)
        self.assertIsNone(d["last_used"])
        self.assertIsNone(d["source_media_id"])

    def test_optional_fields(self):
        entry = PromptEntry(
            id="p2",
            content="test",
            content_type="search_query",
            created_at=2000.0,
            used_count=5,
            last_used=2500.0,
            source_media_id="m1",
            modality="image",
        )
        self.assertEqual(entry.used_count, 5)
        self.assertEqual(entry.modality, "image")


class TestMediaEntry(unittest.TestCase):
    """Tests for MediaEntry dataclass."""

    def test_created_at_auto_set(self):
        before = time.time()
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
        )
        after = time.time()
        self.assertGreaterEqual(entry.created_at, before)
        self.assertLessEqual(entry.created_at, after)

    def test_explicit_created_at_preserved(self):
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.IMAGE,
            media_type=MediaType.REAL,
            created_at=12345.0,
        )
        self.assertEqual(entry.created_at, 12345.0)

    def test_to_dict_converts_enums(self):
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            source_type=SourceType.GENERATED,
        )
        d = entry.to_dict()
        self.assertEqual(d["modality"], "image")
        self.assertEqual(d["media_type"], "synthetic")
        self.assertEqual(d["source_type"], "generated")

    def test_default_source_type(self):
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.IMAGE,
            media_type=MediaType.REAL,
        )
        self.assertEqual(entry.source_type, SourceType.GENERATED)

    def test_miner_fields(self):
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            source_type=SourceType.MINER,
            uid=42,
            hotkey="hk_abc",
            verified=True,
        )
        self.assertEqual(entry.uid, 42)
        self.assertEqual(entry.hotkey, "hk_abc")
        self.assertTrue(entry.verified)

    def test_c2pa_fields(self):
        entry = MediaEntry(
            id="m1",
            prompt_id="p1",
            file_path="/tmp/test.png",
            modality=Modality.VIDEO,
            media_type=MediaType.REAL,
            c2pa_verified=True,
            c2pa_issuer="Adobe",
        )
        self.assertTrue(entry.c2pa_verified)
        self.assertEqual(entry.c2pa_issuer, "Adobe")


class TestVerificationResult(unittest.TestCase):
    """Tests for VerificationResult."""

    def test_default_passed_false(self):
        entry = MediaEntry(
            id="m1", prompt_id="p1", file_path="/tmp/t.png",
            modality=Modality.IMAGE, media_type=MediaType.SYNTHETIC,
        )
        result = VerificationResult(media_entry=entry)
        self.assertFalse(result.passed)
        self.assertIsNone(result.verification_score)
        self.assertIsNone(result.original_prompt)

    def test_with_scores(self):
        entry = MediaEntry(
            id="m1", prompt_id="p1", file_path="/tmp/t.png",
            modality=Modality.IMAGE, media_type=MediaType.SYNTHETIC,
        )
        result = VerificationResult(
            media_entry=entry,
            original_prompt="a dog",
            generated_caption="a dog running",
            verification_score={"clip": 0.85},
            passed=True,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.verification_score["clip"], 0.85)


class TestMedia(unittest.TestCase):
    """Tests for Media dataclass."""

    def test_basic_construction(self):
        m = Media(
            modality=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            media_content="placeholder",
            format="JPEG",
        )
        self.assertEqual(m.modality, Modality.IMAGE)
        self.assertIsNone(m.prompt_id)
        self.assertIsNone(m.model_name)

    def test_with_generation_args(self):
        m = Media(
            modality=Modality.VIDEO,
            media_type=MediaType.SYNTHETIC,
            media_content=None,
            format="MP4",
            model_name="test-model",
            generation_args={"steps": 50},
        )
        self.assertEqual(m.generation_args["steps"], 50)


if __name__ == "__main__":
    unittest.main()
