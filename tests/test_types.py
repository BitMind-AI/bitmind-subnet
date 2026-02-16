"""
Tests for gas.types module.

Covers enum behavior, MediaType int_value mapping, DatasetConfig validation,
ModelConfig construction, and DiscriminatorModelId serialization roundtrip.
"""

import json
import unittest

from gas.types import (
    Modality,
    MediaType,
    SourceType,
    DatasetConfig,
    ModelConfig,
    ModelTask,
    DiscriminatorModelId,
    DiscriminatorModelMetadata,
    SOURCE_TYPE_TO_DB_NAME_FIELD,
    SOURCE_TYPE_TO_NAME,
)


class TestModality(unittest.TestCase):
    def test_values(self):
        self.assertEqual(Modality.IMAGE.value, "image")
        self.assertEqual(Modality.VIDEO.value, "video")

    def test_from_string(self):
        self.assertEqual(Modality("image"), Modality.IMAGE)
        self.assertEqual(Modality("video"), Modality.VIDEO)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            Modality("audio")


class TestMediaType(unittest.TestCase):
    def test_int_values(self):
        self.assertEqual(MediaType.REAL.int_value, 0)
        self.assertEqual(MediaType.SYNTHETIC.int_value, 1)
        self.assertEqual(MediaType.SEMISYNTHETIC.int_value, 2)

    def test_from_string(self):
        self.assertEqual(MediaType("real"), MediaType.REAL)
        self.assertEqual(MediaType("synthetic"), MediaType.SYNTHETIC)


class TestSourceType(unittest.TestCase):
    def test_all_source_types_in_mappings(self):
        for st in SourceType:
            self.assertIn(st, SOURCE_TYPE_TO_NAME)
            self.assertIn(st, SOURCE_TYPE_TO_DB_NAME_FIELD)


class TestDatasetConfig(unittest.TestCase):
    def test_default_source_format_image(self):
        cfg = DatasetConfig(path="test/path", modality=Modality.IMAGE, media_type=MediaType.REAL)
        self.assertEqual(cfg.source_format, "parquet")

    def test_default_source_format_video(self):
        cfg = DatasetConfig(path="test/path", modality=Modality.VIDEO, media_type=MediaType.REAL)
        self.assertEqual(cfg.source_format, "zip")

    def test_string_modality_converted(self):
        cfg = DatasetConfig(path="test/path", modality="image", media_type="real")
        self.assertIsInstance(cfg.modality, Modality)
        self.assertIsInstance(cfg.media_type, MediaType)

    def test_custom_source_format_preserved(self):
        cfg = DatasetConfig(
            path="test/path",
            modality=Modality.IMAGE,
            media_type=MediaType.REAL,
            source_format="custom",
        )
        self.assertEqual(cfg.source_format, "custom")


class TestModelConfig(unittest.TestCase):
    def test_default_media_type_t2i(self):
        cfg = ModelConfig(path="model", task=ModelTask.TEXT_TO_IMAGE, pipeline_cls="cls")
        self.assertEqual(cfg.media_type, MediaType.SYNTHETIC)

    def test_default_media_type_i2i(self):
        cfg = ModelConfig(path="model", task=ModelTask.IMAGE_TO_IMAGE, pipeline_cls="cls")
        self.assertEqual(cfg.media_type, MediaType.SEMISYNTHETIC)

    def test_to_dict_keys(self):
        cfg = ModelConfig(path="model", task=ModelTask.TEXT_TO_IMAGE, pipeline_cls="cls")
        d = cfg.to_dict()
        expected_keys = {
            "pipeline_cls", "from_pretrained_args", "generation_args",
            "use_autocast", "enable_model_cpu_offload",
            "enable_sequential_cpu_offload", "vae_enable_slicing",
            "vae_enable_tiling", "scheduler", "save_args",
            "pipeline_stages", "clear_memory_on_stage_end",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_custom_generation_args(self):
        cfg = ModelConfig(
            path="model",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls="cls",
            generation_args={"steps": 50},
        )
        self.assertEqual(cfg.generation_args["steps"], 50)


class TestDiscriminatorModelId(unittest.TestCase):
    def test_hash_truncated_to_16(self):
        mid = DiscriminatorModelId(key="mykey", hash="a" * 32)
        self.assertEqual(len(mid.hash), 16)

    def test_short_hash_preserved(self):
        mid = DiscriminatorModelId(key="mykey", hash="abc")
        self.assertEqual(mid.hash, "abc")

    def test_compressed_str_roundtrip(self):
        original = DiscriminatorModelId(key="test/model", hash="abcdef1234567890")
        compressed = original.to_compressed_str()
        restored = DiscriminatorModelId.from_compressed_str(compressed)
        self.assertEqual(original, restored)

    def test_equality(self):
        a = DiscriminatorModelId(key="k", hash="h")
        b = DiscriminatorModelId(key="k", hash="h")
        c = DiscriminatorModelId(key="k", hash="x")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_not_equal_to_other_types(self):
        mid = DiscriminatorModelId(key="k", hash="h")
        self.assertNotEqual(mid, "not_a_model_id")


class TestDiscriminatorModelMetadata(unittest.TestCase):
    def test_dict_roundtrip(self):
        mid = DiscriminatorModelId(key="k", hash="h")
        meta = DiscriminatorModelMetadata(id=mid, block=100)
        d = meta.to_dict()
        restored = DiscriminatorModelMetadata.from_dict(d)
        self.assertEqual(restored.id, mid)
        self.assertEqual(restored.block, 100)


class TestModelTask(unittest.TestCase):
    def test_task_values(self):
        self.assertEqual(ModelTask.TEXT_TO_IMAGE.value, "t2i")
        self.assertEqual(ModelTask.TEXT_TO_VIDEO.value, "t2v")
        self.assertEqual(ModelTask.IMAGE_TO_IMAGE.value, "i2i")
        self.assertEqual(ModelTask.IMAGE_TO_VIDEO.value, "i2v")


if __name__ == "__main__":
    unittest.main()
