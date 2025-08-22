from typing import Tuple, Optional, Iterator
import contextlib
import tempfile
import tarfile
import os
import shutil
import glob
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, scan_cache_dir, delete_cache_entries


def discover_latest_image_config() -> Optional[str]:
    """Return the latest config name (partition) for the image dataset or None if unavailable."""
    try:
        configs = get_dataset_config_names('bitmind/bm-image-benchmarks')
        return sorted(configs)[-1] if configs else None
    except Exception:
        return None


def load_latest_image_dataset(streaming: bool = True) -> Tuple[Optional[object], Optional[str]]:
    """Load the latest image dataset config's train split.

    Returns a tuple of (dataset, config_name). Dataset is a streaming iterable if streaming=True.
    """
    latest_config = discover_latest_image_config()
    if latest_config is None:
        return None, None
    try:
        dataset = load_dataset('bitmind/bm-image-benchmarks', latest_config, split='train', streaming=streaming)
        return dataset, latest_config
    except Exception:
        return None, None


def prune_image_dataset_cache() -> None:
    """Best-effort removal of cached bm-image-benchmarks data from datasets cache.

    This removes local cache directories for the dataset so the next
    non-streaming load only keeps the latest split.
    """
    try:
        try:
            from datasets.utils import logging as ds_logging  # noqa: F401
            from datasets.config import HF_DATASETS_CACHE
            cache_dir = HF_DATASETS_CACHE
        except Exception:
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

        if not os.path.isdir(cache_dir):
            return

        # Remove directories that clearly belong to the dataset
        patterns = [
            os.path.join(cache_dir, "bitmind___bm-image-benchmarks*"),
            os.path.join(cache_dir, "*bitmind___bm-image-benchmarks*"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass
    except Exception:
        pass


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


def prune_old_video_payloads(current_config: str) -> None:
    """Remove cached TAR payloads for bm-video-benchmarks configs other than current."""
    try:
        cache_info = scan_cache_dir()
        entries_to_delete = []
        for repo in cache_info.repos:
            if repo.repo_id == 'bitmind/bm-video-benchmarks' and repo.repo_type == 'dataset':
                for revision in repo.revisions:
                    for file_ref in revision.files:
                        # file_ref.path like 'videos/<config>.tar.gz'
                        if file_ref.path.startswith('videos/') and not file_ref.path.endswith(f'{current_config}.tar.gz'):
                            entries_to_delete.append(file_ref)
        if entries_to_delete:
            delete_cache_entries(*entries_to_delete)
    except Exception:
        pass


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


