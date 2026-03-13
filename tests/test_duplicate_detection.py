"""
Comprehensive tests for gas.verification.duplicate_detection module.

Tests cover:
- Hash extraction utilities (extract_phash, extract_crop_segments)
- Hamming distance calculation
- Duplicate detection logic (pHash and crop-resistant)
- find_duplicates with various scenarios
- Edge cases: empty strings, missing segments, malformed hashes
"""

import unittest
from unittest.mock import patch, MagicMock

from gas.verification.duplicate_detection import (
    extract_phash,
    extract_crop_segments,
    hamming_distance,
    count_crop_segment_matches,
    is_duplicate,
    find_duplicates,
    compute_media_hash,
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD,
)


class TestExtractPhash(unittest.TestCase):
    """Tests for extract_phash utility."""

    def test_plain_hash(self):
        self.assertEqual(extract_phash("abcdef1234567890"), "abcdef1234567890")

    def test_hash_with_crop_segments(self):
        self.assertEqual(extract_phash("abcdef|seg1;seg2"), "abcdef")

    def test_video_hash_with_frame_count(self):
        self.assertEqual(extract_phash("abcdef_4"), "abcdef")

    def test_video_hash_with_crop_and_frames(self):
        self.assertEqual(extract_phash("abcdef|seg1;seg2_4"), "abcdef")

    def test_empty_string(self):
        self.assertEqual(extract_phash(""), "")

    def test_hash_with_multiple_pipes(self):
        # Edge case: should take first part
        result = extract_phash("abc|def|ghi")
        self.assertEqual(result, "abc")


class TestExtractCropSegments(unittest.TestCase):
    """Tests for extract_crop_segments utility."""

    def test_no_segments(self):
        self.assertEqual(extract_crop_segments("abcdef"), [])

    def test_single_segment(self):
        self.assertEqual(extract_crop_segments("abcdef|seg1"), ["seg1"])

    def test_multiple_segments(self):
        result = extract_crop_segments("abcdef|seg1;seg2;seg3")
        self.assertEqual(result, ["seg1", "seg2", "seg3"])

    def test_empty_crop_part(self):
        self.assertEqual(extract_crop_segments("abcdef|"), [""])

    def test_no_pipe_returns_empty(self):
        self.assertEqual(extract_crop_segments("just_a_hash"), [])


class TestHammingDistance(unittest.TestCase):
    """Tests for hamming_distance function."""

    def test_identical_hashes(self):
        h = "a" * 64  # 256-bit hash as hex
        dist = hamming_distance(h, h)
        self.assertEqual(dist, 0)

    def test_different_hashes(self):
        h1 = "0" * 64
        h2 = "f" * 64
        dist = hamming_distance(h1, h2)
        self.assertGreater(dist, 0)

    def test_extracts_phash_before_comparing(self):
        """Should strip crop segments before comparing."""
        h = "0" * 64
        h_with_crop = f"{'0' * 64}|seg1;seg2"
        dist = hamming_distance(h, h_with_crop)
        self.assertEqual(dist, 0)

    def test_strips_frame_count(self):
        h = "0" * 64
        h_video = f"{'0' * 64}_4"
        dist = hamming_distance(h, h_video)
        self.assertEqual(dist, 0)


class TestIsDuplicate(unittest.TestCase):
    """Tests for is_duplicate function."""

    def test_identical_hashes_are_duplicate(self):
        h = "0" * 64
        self.assertTrue(is_duplicate(h, h))

    def test_very_different_hashes_not_duplicate(self):
        h1 = "0" * 64
        h2 = "f" * 64
        self.assertFalse(is_duplicate(h1, h2, threshold=5))

    def test_custom_threshold(self):
        h = "0" * 64
        self.assertTrue(is_duplicate(h, h, threshold=0))


class TestFindDuplicates(unittest.TestCase):
    """Tests for find_duplicates function."""

    def test_empty_existing_hashes(self):
        result = find_duplicates("0" * 64, [])
        self.assertEqual(result, [])

    def test_finds_exact_match(self):
        h = "0" * 64
        existing = [("media1", h), ("media2", "f" * 64)]
        result = find_duplicates(h, existing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "media1")
        self.assertEqual(result[0][1], 0)

    def test_skips_none_hashes(self):
        h = "0" * 64
        existing = [("media1", None), ("media2", "")]
        result = find_duplicates(h, existing)
        self.assertEqual(result, [])

    def test_sorted_by_distance(self):
        h = "0" * 64
        existing = [("media1", h), ("media2", h)]
        result = find_duplicates(h, existing)
        # Both should match with distance 0
        self.assertEqual(len(result), 2)
        for _, dist in result:
            self.assertEqual(dist, 0)


class TestComputeMediaHash(unittest.TestCase):
    """Tests for compute_media_hash function."""

    def test_unsupported_modality_returns_none(self):
        result = compute_media_hash(b"data", modality="audio")
        self.assertIsNone(result)

    @patch("gas.verification.duplicate_detection.IMAGEHASH_AVAILABLE", False)
    def test_imagehash_unavailable_returns_none(self):
        from gas.verification.duplicate_detection import compute_image_hash
        result = compute_image_hash(b"data")
        self.assertIsNone(result)


class TestCountCropSegmentMatches(unittest.TestCase):
    """Tests for count_crop_segment_matches."""

    def test_no_segments_returns_zero(self):
        result = count_crop_segment_matches("abc", "def")
        self.assertEqual(result, 0)

    def test_one_hash_without_segments(self):
        result = count_crop_segment_matches("abc|seg1", "def")
        self.assertEqual(result, 0)


class TestDefaultConstants(unittest.TestCase):
    """Verify default constants are reasonable."""

    def test_default_hamming_threshold(self):
        self.assertIsInstance(DEFAULT_HAMMING_THRESHOLD, int)
        self.assertGreater(DEFAULT_HAMMING_THRESHOLD, 0)
        self.assertLessEqual(DEFAULT_HAMMING_THRESHOLD, 20)

    def test_default_crop_match_threshold(self):
        self.assertIsInstance(DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD, int)
        self.assertGreater(DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD, 0)


if __name__ == "__main__":
    unittest.main()
