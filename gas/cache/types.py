from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np
import time
import uuid

from gas.types import Modality, MediaType, SourceType


@dataclass
class Media:
    """Input data for creating media - what you want to write"""

    modality: Modality  # Modality.IMAGE or Modality.VIDEO
    media_type: MediaType  # MediaType.REAL, MediaType.SYNTHETIC, or MediaType.SEMISYNTHETIC
    media_content: Any  # PIL Image, video frames, etc.
    format: str  # "JPEG", "PNG", "MP4", etc.
    prompt_id: Optional[str] = None  # None for dataset media, str for generated media

    # synthetic & semisynthetic
    model_name: Optional[str] = None
    generation_args: Optional[Dict[str, Any]] = None

    # semisynthetic
    original_media_id: Optional[str] = None
    mask_content: Optional[np.ndarray] = None
    
    # Common metadata
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PromptEntry:
    """Represents a prompt entry in the database"""

    id: str
    content: str
    content_type: str  # "prompt" or "search_query"
    created_at: float
    used_count: int = 0
    last_used: Optional[float] = None
    source_media_id: Optional[str] = None  # ID of the media that was used to generate this prompt

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MediaEntry:
    """Represents a media entry linked to a prompt - database record for stored media"""

    id: str
    prompt_id: str
    file_path: str
    modality: Modality  # image or video
    media_type: MediaType  # real, synthetic, semisynthetic
    source_type: SourceType = SourceType.GENERATED  # "scraper", "dataset", "generated"

    # For synthetic media (generated content)
    model_name: Optional[str] = None

    # For real media (downloaded content)
    download_url: Optional[str] = None
    scraper_name: Optional[str] = None

    # For dataset media
    dataset_name: Optional[str] = None
    dataset_source_file: Optional[str] = None
    dataset_index: Optional[str] = None

    # Common fields
    created_at: float = None
    generation_args: Optional[Dict[str, Any]] = None  # Generation parameters when source_type='generated'
    resolution: Optional[tuple[int, int]] = None  # (width, height)
    file_size: Optional[int] = None  # in bytes
    format: Optional[str] = None  # File format (e.g., "PNG", "JPEG", "MP4")

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("modality"), Modality):
            data["modality"] = data["modality"].value
        if isinstance(data.get("media_type"), MediaType):
            data["media_type"] = data["media_type"].value
        if isinstance(data.get("source_type"), SourceType):
            data["source_type"] = data["source_type"].value
        return data