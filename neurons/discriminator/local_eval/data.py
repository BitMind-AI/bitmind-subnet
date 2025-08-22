from typing import Tuple, Optional, Iterator
import contextlib
import tempfile
import tarfile
import os
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, scan_cache_dir


def discover_latest_image_config() -> Optional[str]:
    """Return the latest config name (partition) for the image dataset or None if unavailable."""
    try:
        configs = get_dataset_config_names('bitmind/bm-image-benchmarks')
        return sorted(configs)[-1] if configs else None
    except Exception:
        return None


def discover_all_image_configs() -> list:
    """Return all config names sorted ascending; empty list if unavailable."""
    try:
        configs = get_dataset_config_names('bitmind/bm-image-benchmarks')
        return sorted(configs) if configs else []
    except Exception:
        return []


def load_latest_image_dataset(streaming: bool = True) -> Tuple[Optional[object], Optional[str]]:
    """Load the newest working image dataset split, with fallback to older splits.

    Returns (dataset, config_name). Dataset is streaming iterable if streaming=True.
    """
    configs = discover_all_image_configs()
    if not configs:
        return None, None

    # Try from newest to oldest
    for config_name in reversed(configs):
        try:
            dataset = load_dataset(
                'bitmind/bm-image-benchmarks', config_name, split='train', streaming=streaming
            )
            # Probe a few samples to ensure they are loadable
            probe_count = 0
            for sample in dataset:
                # Basic validation: must have label and at least one media field
                if sample.get('label') is None:
                    raise ValueError("missing label in sample")
                # Accept either PIL object or a path/bytes; defer actual decoding to inference
                media_has_value = (
                    sample.get('media_image') is not None
                    or sample.get('image') is not None
                    or sample.get('filepath') is not None
                )
                if not media_has_value:
                    raise ValueError("missing image media field")
                probe_count += 1
                if probe_count >= 3:
                    break
            if probe_count == 0:
                # No samples produced; try older split
                continue
            return dataset, config_name
        except Exception:
            # Try older split
            continue

    return None, None


def discover_latest_video_config() -> Optional[str]:
    """Return the latest config name (partition) for the video dataset or None if unavailable."""
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names('bitmind/bm-video-benchmarks')
        return sorted(configs)[-1] if configs else None
    except Exception:
        return None


def load_latest_video_dataset(streaming: bool = True, split: str = 'train') -> Tuple[Optional[object], Optional[str]]:
    """Load the latest video dataset config and requested split.

    Returns a tuple of (dataset, config_name). Dataset is streaming iterable if streaming=True.
    """
    latest_config = discover_latest_video_config()
    if latest_config is None:
        return None, None
    try:
        dataset = load_dataset('bitmind/bm-video-benchmarks', latest_config, split=split, streaming=streaming)
        return dataset, latest_config
    except Exception:
        return None, None


@contextlib.contextmanager
def video_payload(temp_dir: Optional[str], dataset_config: str) -> Iterator[str]:
    """Context manager that downloads and extracts the TAR payload for the given
    video dataset config and yields the extraction directory path.

    If temp_dir is None, a temporary directory is created and cleaned up.
    """
    own_tmpdir = None
    extract_dir = None
    try:
        if temp_dir is None:
            own_tmpdir = tempfile.TemporaryDirectory()
            extract_dir = own_tmpdir.name
        else:
            os.makedirs(temp_dir, exist_ok=True)
            extract_dir = temp_dir

        tar_path = hf_hub_download(
            repo_id='bitmind/bm-video-benchmarks',
            filename=f'videos/{dataset_config}.tar.gz',
            repo_type='dataset'
        )

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)

        yield extract_dir
    finally:
        if own_tmpdir is not None:
            own_tmpdir.cleanup()


