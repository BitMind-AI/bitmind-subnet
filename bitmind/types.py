from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union


class NeuronType(Enum):
    VALIDATOR = "VALIDATOR"
    VALIDATOR_PROXY = "VALIDATOR_PROXY"
    MINER = "MINER"


class MinerType(Enum):
    SEGMENTER = "SEGMENTER"
    DETECTOR = "DETECTOR"


class FileType(Enum):
    PARQUET = auto()
    ZIP = auto()
    VIDEO = auto()
    IMAGE = auto()


class CacheType(str, Enum):
    MEDIA = "media"
    COMPRESSED = "compressed"


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class MediaType(str, Enum):
    REAL = "real", 0
    SYNTHETIC = "synthetic", 1
    SEMISYNTHETIC = "semisynthetic", 2

    def __new__(cls, str_value, int_value):
        obj = str.__new__(cls, str_value)
        obj._value_ = str_value
        obj.int_value = int_value
        return obj


@dataclass
class CacheUpdaterConfig:
    num_sources_per_dataset: int = 1
    num_items_per_source: int = 100


@dataclass
class CacheConfig:
    """Configuration for a cache at base_dir / {modality} / {media_type}"""

    modality: str
    media_type: str
    base_dir: Path = Path("~/.cache/sn34").expanduser()
    tags: Optional[List[str]] = None
    max_compressed_gb: float = 100.0
    max_media_gb: float = 10.0

    def get_path(self):
        media_cache_path = Path(self.base_dir) / self.modality / self.media_type
        media_cache_path.mkdir(exist_ok=True, parents=True)
        return media_cache_path


@dataclass
class DatasetConfig:
    path: str  # HuggingFace path
    type: Modality
    media_type: MediaType
    tags: List[str] = field(default_factory=list)
    file_format: str = ""
    compressed_format: str = ""
    priority: int = 1  # Optional: priority for sampling (higher is more frequent)
    enabled: bool = True

    def __post_init__(self):
        """Validate and set defaults"""
        if not self.compressed_format:
            if self.type == Modality.IMAGE:
                self.compressed_format = "parquet"
            elif self.type == Modality.VIDEO:
                self.compressed_format = "zip"

        if isinstance(self.tags, str):
            self.tags = [t.strip() for t in self.tags.split(",")]

        if isinstance(self.type, str):
            self.type = Modality(self.type.lower())

        if isinstance(self.media_type, str):
            self.media_type = MediaType(self.media_type.lower())


class ModelTask(str, Enum):
    """Type of task the model is designed for"""

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
        generate_args: Default arguments for generation
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
        generate_args: Dict[str, Any] = None,
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
        self.generate_args = generate_args or {}
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
            "generate_args": self.generate_args,
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


class ValidatorConfig(BaseModel):
    skip_weight_set: Optional[bool] = False
    set_weights_on_start: Optional[bool] = False
    max_concurrent_organics: Optional[int] = 2
