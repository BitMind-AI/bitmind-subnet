from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import json


class NeuronType(Enum):
    VALIDATOR = "VALIDATOR"
    MINER = "MINER"


class MinerType(Enum):
    DISCRIMINATOR = "DISCRIMINATOR"
    GENERATOR = "GENERATOR"
    ENCODER = "ENCODER"
    CAPTIONER = "CAPTIONER"


class DiscriminatorType(Enum):
    DETECTOR = "DETECTOR"
    SEGMENTER = "SEGMENTER"


class FileType(Enum):
    PARQUET = auto()
    ZIP = auto()
    VIDEO = auto()
    IMAGE = auto()


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class MediaType(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    SEMISYNTHETIC = "semisynthetic"

    @property
    def int_value(self):
        """Get the integer value for this media type"""
        mapping = {
            MediaType.REAL: 0,
            MediaType.SYNTHETIC: 1,
            MediaType.SEMISYNTHETIC: 2,
        }
        return mapping[self]


class SourceType(str, Enum):
    """Canonical source types for media provenance."""

    SCRAPER = "scraper"
    DATASET = "dataset"
    GENERATED = "generated"
    MINER = "miner"


SOURCE_TYPE_TO_NAME: Dict[SourceType, str] = {
    SourceType.GENERATED: "model_name",
    SourceType.DATASET: "dataset_name",
    SourceType.SCRAPER: "download_url",
    SourceType.MINER: "hotkey",
}


SOURCE_TYPE_TO_DB_NAME_FIELD: Dict[SourceType, str] = {
    SourceType.GENERATED: "model_name",
    SourceType.DATASET: "dataset_name",
    SourceType.SCRAPER: "scraper_name",
    SourceType.MINER: "hotkey",
}


@dataclass
class DatasetConfig:
    """For datasets used by the Validator"""
    path: str  # HuggingFace path
    modality: Modality
    media_type: MediaType
    file_format: str = ""
    source_format: str = ""
    priority: int = 1  # Optional: priority for sampling (higher is more frequent)
    enabled: bool = True

    def __post_init__(self):
        """Validate and set defaults"""
        if not self.source_format:
            if self.modality == Modality.IMAGE:
                self.source_format = "parquet"
            elif self.modality == Modality.VIDEO:
                self.source_format = "zip"

        if isinstance(self.modality, str):
            self.modality = Modality(self.modality.lower())

        if isinstance(self.media_type, str):
            self.media_type = MediaType(self.media_type.lower())


class ModelTask(str, Enum):
    "Tasks supported by validator models"
    TEXT_TO_IMAGE = "t2i"
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_IMAGE = "i2i"
    IMAGE_TO_VIDEO = "i2v"


class ModelConfig:
    """
    Configuration for a generative AI model.

    Attributes:
        path: The Hugging Face model path or identifier
        task: The primary task of the model (T2I, T2V, I2I)
        media_type: Type of output (synthetic or semisynthetic)
        pipeline_cls: Pipeline class used to load the model
        pretrained_args: Arguments for the from_pretrained method
        generation_args: Default arguments for generation
        tags: List of tags for categorizing the model
        use_autocast: Whether to use autocast during generation
        scheduler: Optional scheduler configuration
        scheduler_cls: Optional scheduler class
        scheduler_args: Optional scheduler args
    """

    def __init__(
        self,
        path: str,
        task: ModelTask,
        pipeline_cls: Union[Any, Dict[str, Any]],
        media_type: Optional[MediaType] = None,
        pretrained_args: Dict[str, Any] = None,
        generation_args: Dict[str, Any] = None,
        tags: List[str] = None,
        use_autocast: bool = True,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        vae_enable_slicing: bool = False,
        vae_enable_tiling: bool = False,
        scheduler: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        pipeline_stages: List[Dict[str, Any]] = None,
        clear_memory_on_stage_end: bool = False,
        lora_model_id: str = None,
        lora_loading_args: Dict[str, Any] = None,
    ):
        self.path = path
        self.task = task
        self.pipeline_cls = pipeline_cls
        self.media_type = media_type

        if self.media_type is None:
            self.media_type = (
                MediaType.SEMISYNTHETIC
                if task == ModelTask.IMAGE_TO_IMAGE
                else MediaType.SYNTHETIC
            )

        self.pretrained_args = pretrained_args or {}
        self.generation_args = generation_args or {}
        self.tags = tags or []
        self.use_autocast = use_autocast
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.vae_enable_slicing = vae_enable_slicing
        self.vae_enable_tiling = vae_enable_tiling
        self.scheduler = scheduler
        self.save_args = save_args or {}
        self.pipeline_stages = pipeline_stages
        self.clear_memory_on_stage_end = clear_memory_on_stage_end
        self.lora_model_id = lora_model_id
        self.lora_loading_args = lora_loading_args

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format"""
        return {
            "pipeline_cls": self.pipeline_cls,
            "from_pretrained_args": self.pretrained_args,
            "generation_args": self.generation_args,
            "use_autocast": self.use_autocast,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
            "vae_enable_slicing": self.vae_enable_slicing,
            "vae_enable_tiling": self.vae_enable_tiling,
            "scheduler": self.scheduler,
            "save_args": self.save_args,
            "pipeline_stages": self.pipeline_stages,
            "clear_memory_on_stage_end": self.clear_memory_on_stage_end,
        }


# =============================================================================
# MODEL MANAGEMENT TYPES
# =============================================================================

@dataclass
class DiscriminatorModelId:
    """Unique identifier for a discriminator model"""

    key: str  # Remote storage key
    hash: str  # Model dir hash

    def __post_init__(self):
        """Validate and truncate commit and hash to 16 chars"""
        if self.hash and len(self.hash) > 16:
            self.hash = self.hash[:16]

    def to_compressed_str(self) -> str:
        """Convert to compressed string for chain storage"""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def to_dict(self) -> dict:
        """Convert to dictionary for registry storage"""
        data = {
            "key": self.key,
            "hash": self.hash,
        }
        return data

    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "DiscriminatorModelId":
        """Create DiscriminatorModelId from compressed string"""
        data = json.loads(compressed_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DiscriminatorModelId":
        """Create DiscriminatorModelId from dictionary"""
        return cls(
            key=data["key"],
            hash=data["hash"],
        )

    def __eq__(self, other):
        if not isinstance(other, DiscriminatorModelId):
            return False
        return (
            self.key == other.key
            and self.hash == other.hash
        )


@dataclass
class DiscriminatorModelMetadata:
    """Metadata for a discriminator model stored on chain"""

    id: DiscriminatorModelId
    block: int  # Block number when metadata was stored

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": {
                "key": self.id.key,
                "hash": self.id.hash,
            },
            "block": self.block,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscriminatorModelMetadata":
        """Create DiscriminatorModelMetadata from dictionary"""
        model_id = DiscriminatorModelId(
            key=data["id"]["key"],
            hash=data["id"]["hash"],
        )
        return cls(id=model_id, block=data["block"])


@dataclass
class ArtifactTaskSpec:
    """Encoding/captioning constraints for a DPS artifact task."""

    resolution: Optional[str] = None
    max_frames: Optional[int] = None
    encoding_model: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "resolution": self.resolution,
            "max_frames": self.max_frames,
            "encoding_model": self.encoding_model,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ArtifactTaskSpec"]:
        if not data:
            return None
        max_frames = data.get("max_frames")
        if max_frames == "":
            max_frames = None
        return cls(
            resolution=data.get("resolution"),
            max_frames=int(max_frames) if max_frames is not None else None,
            encoding_model=data.get("encoding_model")
            or data.get("vae_encoder")
            or data.get("model"),
        )


@dataclass
class ArtifactR2Location:
    """R2 location and scoped credentials for DPS artifact exchange."""

    bucket: str
    path: str
    endpoint_url: Optional[str] = None
    region: Optional[str] = "auto"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    manifest_url: Optional[str] = None
    manifest_key: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "bucket": self.bucket,
            "path": self.path,
            "endpoint_url": self.endpoint_url,
            "region": self.region,
            "access_key_id": self.access_key_id,
            "secret_access_key": self.secret_access_key,
            "session_token": self.session_token,
            "manifest_url": self.manifest_url,
            "manifest_key": self.manifest_key,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactR2Location":
        return cls(
            bucket=data["bucket"],
            path=data.get("path") or data.get("prefix", ""),
            endpoint_url=data.get("endpoint_url"),
            region=data.get("region", "auto"),
            access_key_id=data.get("access_key_id"),
            secret_access_key=data.get("secret_access_key"),
            session_token=data.get("session_token"),
            manifest_url=data.get("manifest_url"),
            manifest_key=data.get("manifest_key"),
        )


@dataclass
class ArtifactChainMetadata:
    """DPS artifact metadata stored in a hotkey chain commitment."""

    kind: str
    role: MinerType
    r2: ArtifactR2Location
    version: int = 1
    task_id: Optional[str] = None
    artifact_format: Optional[str] = None
    artifact_hash: Optional[str] = None
    artifact_spec: Optional[ArtifactTaskSpec] = None

    def to_dict(self) -> dict:
        data = {
            "v": self.version,
            "kind": self.kind,
            "role": self.role.value,
            "r2": self.r2.to_dict(),
            "task_id": self.task_id,
            "artifact_format": self.artifact_format,
            "artifact_hash": self.artifact_hash,
            "artifact_spec": (
                self.artifact_spec.to_dict()
                if self.artifact_spec is not None
                else None
            ),
        }
        return {key: value for key, value in data.items() if value is not None}

    def registry_key(self) -> str:
        task_part = self.task_id or "default"
        return f"{self.kind}:{self.role.value}:{task_part}"

    def to_compressed_str(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "ArtifactChainMetadata":
        data = json.loads(compressed_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactChainMetadata":
        if data.get("kind") not in ("dps_input", "dps_output"):
            raise ValueError(f"Unsupported artifact metadata kind: {data.get('kind')}")
        return cls(
            version=int(data.get("v", 1)),
            kind=data["kind"],
            role=MinerType(data["role"]),
            r2=ArtifactR2Location.from_dict(data["r2"]),
            task_id=data.get("task_id"),
            artifact_format=data.get("artifact_format"),
            artifact_hash=data.get("artifact_hash"),
            artifact_spec=ArtifactTaskSpec.from_dict(data.get("artifact_spec")),
        )


@dataclass
class ChainMetadataRegistry:
    """Single-commitment registry for multiple subnet metadata records."""

    entries: Dict[str, dict] = field(default_factory=dict)
    discriminator_model: Optional[DiscriminatorModelId] = None
    version: int = 1

    kind: str = "dps_registry"

    def to_compressed_str(self) -> str:
        data = {
            "v": self.version,
            "kind": self.kind,
            "entries": self.entries,
        }
        if self.discriminator_model is not None:
            data["discriminator_model"] = self.discriminator_model.to_dict()
        return json.dumps(data, separators=(",", ":"))

    def upsert_artifact(self, metadata: ArtifactChainMetadata):
        self.entries[metadata.registry_key()] = metadata.to_dict()

    def get_artifacts(
        self,
        expected_kind: Optional[str] = None,
        role: Optional[MinerType] = None,
        task_id: Optional[str] = None,
    ) -> List[ArtifactChainMetadata]:
        artifacts = []
        for entry in self.entries.values():
            try:
                artifact = ArtifactChainMetadata.from_dict(entry)
            except Exception:
                continue
            if expected_kind and artifact.kind != expected_kind:
                continue
            if role and artifact.role != role:
                continue
            if task_id and artifact.task_id != task_id:
                continue
            artifacts.append(artifact)
        return artifacts

    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "ChainMetadataRegistry":
        data = json.loads(compressed_str)
        if data.get("kind") == "dps_registry":
            discriminator_model = None
            if data.get("discriminator_model"):
                discriminator_model = DiscriminatorModelId.from_dict(
                    data["discriminator_model"]
                )
            return cls(
                version=int(data.get("v", 1)),
                entries=data.get("entries", {}),
                discriminator_model=discriminator_model,
            )

        registry = cls()
        if data.get("kind") in ("dps_input", "dps_output"):
            artifact = ArtifactChainMetadata.from_dict(data)
            registry.upsert_artifact(artifact)
            return registry

        if "key" in data and "hash" in data:
            registry.discriminator_model = DiscriminatorModelId.from_dict(data)
            return registry

        raise ValueError("Unsupported chain metadata payload")


class ValidatorConfig(BaseModel):
    skip_weight_set: Optional[bool] = False
    set_weights_on_start: Optional[bool] = False
    max_concurrent_organics: Optional[int] = 2
