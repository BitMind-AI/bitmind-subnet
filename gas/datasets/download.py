import asyncio
import base64
import hashlib
import json
import random
import os
import traceback
import tempfile
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator, Tuple
from zipfile import ZipFile
import tarfile

import bittensor as bt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


import requests
from gas.types import Modality, MediaType, DatasetConfig


def download_and_extract(
    dataset: DatasetConfig,
    images_per_parquet: int = 100,
    videos_per_zip: int = 50,
    parquet_per_dataset: int = 5,
    zips_per_dataset: int = 2,
    temp_dir: Optional[str] = None,
) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
    """
    Download datasets and yield extracted media as a generator.

    This approach uses temp files to avoid storing large zip/parquet files
    locally while ensuring diverse media availability.

    Args:
        datasets: List of DatasetConfig objects
        exclude_tags: Tags to exclude from download

    Yields:
        Tuples of (media_bytes, metadata_dict)
    """
    try:
        # temp dir for zip/parquet
        temp_dir_root = None
        if temp_dir is not None:
            try:
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            temp_dir_root = Path(tempfile.mkdtemp(dir=temp_dir))
        else:
            temp_dir_root = Path(tempfile.mkdtemp())

        try:
            filenames = _list_remote_dataset_files(dataset.path, dataset.source_format)
            if not filenames:
                bt.logging.warning(f"No files found for {dataset.path}")
                return None, None

            remote_paths = _get_download_urls(dataset.path, filenames)
            n_files = (
                parquet_per_dataset
                if dataset.modality == Modality.IMAGE
                else zips_per_dataset
            )
            to_download = _select_files_to_download(remote_paths, n_files)

            bt.logging.debug(
                f"Downloading {len(to_download)} files from {dataset.path}"
            )
            downloaded_files = _download_files(to_download, temp_dir_root)

            for source_file in downloaded_files:
                if dataset.modality == Modality.IMAGE:
                    for media_bytes, metadata in yield_images_from_parquet(
                        source_file, images_per_parquet, dataset
                    ):
                        yield media_bytes, metadata
                elif dataset.modality == Modality.VIDEO:
                    file_name_lower = str(source_file.name).lower()
                    if _is_zip_file(file_name_lower):
                        for media_bytes, metadata in yield_videos_from_zip(
                            source_file, videos_per_zip, dataset
                        ):
                            yield media_bytes, metadata
                    elif _is_tar_file(file_name_lower):
                        for media_bytes, metadata in yield_videos_from_tar(
                            source_file, videos_per_zip, dataset
                        ):
                            yield media_bytes, metadata

        finally:
            if temp_dir_root.exists():
                shutil.rmtree(temp_dir_root)

    except Exception as e:
        bt.logging.error(f"Error processing {dataset.path}: {e}")


def yield_videos_from_zip(
    zip_path: Path,
    num_videos: int,
    dataset: DatasetConfig,
    file_extensions: set = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"},
    include_checksums: bool = True,
) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
    """
    Extract random videos from a zip file as a generator.

    Args:
        zip_path: Path to the zip file
        num_videos: Number of videos to extract
        file_extensions: Set of valid video file extensions
        include_checksums: Whether to calculate and include file checksums in metadata

    Yields:
        Tuples of (video_bytes, metadata_dict)
    """
    try:
        with ZipFile(zip_path) as zip_file:
            video_files = [
                f
                for f in zip_file.namelist()
                if any(f.lower().endswith(ext) for ext in file_extensions)
                and "MACOSX" not in f
            ]
            if not video_files:
                bt.logging.warning(f"No video files found in {zip_path}")
                return

            bt.logging.debug(f"{len(video_files)} video files found in {zip_path}")
            selected_videos = random.sample(
                video_files, min(num_videos, len(video_files))
            )

            bt.logging.debug(
                f"Extracting {len(selected_videos)} randomly sampled video files from {zip_path}"
            )
            for video in selected_videos:
                try:
                    # Extract video bytes
                    with zip_file.open(video) as source:
                        video_bytes = source.read()

                    video_info = zip_file.getinfo(video)
                    metadata = {
                        "dataset": Path(zip_path).parent.name,
                        "source_zip": str(zip_path),
                        "path_in_zip": video,
                        "extraction_date": datetime.now().isoformat(),
                        "file_size": len(video_bytes),
                        "dataset_path": dataset.path,
                        "dataset_tags": dataset.tags,
                        "dataset_priority": dataset.priority,
                        "modality": dataset.modality,
                        "media_type": dataset.media_type,
                        "zip_metadata": {
                            "compress_size": video_info.compress_size,
                            "file_size": video_info.file_size,
                            "compress_type": video_info.compress_type,
                            "date_time": datetime.strftime(
                                datetime(*video_info.date_time), "%Y-%m-%d %H:%M:%S"
                            ),
                        },
                    }

                    if include_checksums:
                        metadata["checksums"] = {
                            "md5": hashlib.md5(video_bytes).hexdigest(),
                            "sha256": hashlib.sha256(video_bytes).hexdigest(),
                        }

                    yield video_bytes, metadata

                except Exception as e:
                    bt.logging.warning(f"Error extracting {video}: {e}")
                    continue

    except Exception as e:
        bt.logging.warning(f"Error processing zip file {zip_path}: {e}")


def yield_videos_from_tar(
    tar_path: Path,
    num_videos: int,
    dataset: DatasetConfig,
    file_extensions: set = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"},
    include_checksums: bool = True,
) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
    """
    Extract random videos from a tar archive (.tar, .tar.gz, .tgz) as a generator.

    Args:
        tar_path: Path to the tar archive
        num_videos: Number of videos to extract
        file_extensions: Set of valid video file extensions
        include_checksums: Whether to calculate and include file checksums in metadata

    Yields:
        Tuples of (video_bytes, metadata_dict)
    """
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            members = [
                m
                for m in tf.getmembers()
                if m.isreg()
                and any(m.name.lower().endswith(ext) for ext in file_extensions)
                and "MACOSX" not in m.name
            ]

            if not members:
                bt.logging.warning(f"No video files found in {tar_path}")
                return

            bt.logging.debug(f"{len(members)} video files found in {tar_path}")
            selected_members = random.sample(members, min(num_videos, len(members)))

            bt.logging.debug(
                f"Extracting {len(selected_members)} randomly sampled video files from {tar_path}"
            )

            for member in selected_members:
                try:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    video_bytes = extracted.read()

                    metadata = {
                        "dataset": Path(tar_path).parent.name,
                        "source_tar": str(tar_path),
                        "path_in_tar": member.name,
                        "extraction_date": datetime.now().isoformat(),
                        "file_size": len(video_bytes),
                        "dataset_path": dataset.path,
                        "dataset_tags": dataset.tags,
                        "dataset_priority": dataset.priority,
                        "modality": dataset.modality,
                        "media_type": dataset.media_type,
                        "tar_metadata": {
                            "size": member.size,
                            "mtime": datetime.fromtimestamp(member.mtime).isoformat()
                            if getattr(member, "mtime", None)
                            else None,
                            "uid": getattr(member, "uid", None),
                            "gid": getattr(member, "gid", None),
                            "type": chr(member.type) if isinstance(member.type, int) else str(member.type),
                        },
                    }

                    if include_checksums:
                        metadata["checksums"] = {
                            "md5": hashlib.md5(video_bytes).hexdigest(),
                            "sha256": hashlib.sha256(video_bytes).hexdigest(),
                        }

                    yield video_bytes, metadata

                except Exception as e:
                    bt.logging.warning(f"Error extracting {member.name}: {e}")
                    continue

    except Exception as e:
        bt.logging.warning(f"Error processing tar file {tar_path}: {e}")


def yield_images_from_parquet(
    parquet_path: Path,
    num_images: int,
    dataset: DatasetConfig,
    seed: Optional[int] = None,
) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
    """
    Extract random images from a parquet file as a generator.

    Args:
        parquet_path: Path to the parquet file
        num_images: Number of images to extract
        seed: Random seed for sampling

    Yields:
        Tuples of (PIL_Image, metadata_dict)
    """
    # read parquet file, sample random image rows
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    sample_df = df.sample(n=min(num_images, len(df)), random_state=seed)
    image_col = next((col for col in sample_df.columns if "image" in col.lower()), None)
    metadata_cols = [c for c in sample_df.columns if c != image_col]

    if not image_col:
        bt.logging.warning(f"No image column found in {parquet_path}")
        return

    for idx, row in sample_df.iterrows():
        try:
            img_data = row[image_col]
            if isinstance(img_data, dict):
                key = next(
                    (
                        k
                        for k in img_data
                        if "bytes" in k.lower() or "image" in k.lower()
                    ),
                    None,
                )
                img_data = img_data[key]

            try:
                img = Image.open(BytesIO(img_data))
            except Exception:
                img_data = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_data))

            metadata = {
                "dataset": Path(parquet_path).parent.name,
                "source_parquet": str(parquet_path),
                "original_index": str(idx),
                "image_format": img.format.lower() if img.format else "png",
                "image_size": img.size,
                "image_mode": img.mode,
                "dataset_path": dataset.path,
                "dataset_tags": dataset.tags,
                "dataset_priority": dataset.priority,
                "modality": dataset.modality,
                "media_type": dataset.media_type,
            }

            for col in metadata_cols:
                # Convert any non-serializable types to strings
                try:
                    json.dumps({col: row[col]})
                    metadata[col] = row[col]
                except (TypeError, OverflowError):
                    metadata[col] = str(row[col])

            # Yield PIL Image directly instead of converting to bytes
            yield img, metadata

        except Exception as e:
            bt.logging.warning(f"Failed to extract image {idx}: {e}")
            continue


def _select_files_to_download(urls: List[str], count: int) -> List[str]:
    """Select random files to download"""
    return random.sample(urls, min(count, len(urls)))


def _list_remote_dataset_files(
    dataset_path: str, source_format: str = ".parquet"
) -> List[str]:
    """List available files in a dataset, filtered by source_format.

    Supports single extensions (e.g., .parquet, .zip) and tar variants (.tar, .tar.gz, .tgz).
    """
    if not source_format.startswith("."):
        source_format = "." + source_format

    if source_format == ".tar":
        extensions = [".tar", ".tar.gz", ".tgz"]
        return list_hf_files(repo_id=dataset_path, extension=extensions)

    return list_hf_files(repo_id=dataset_path, extension=source_format)


def _get_download_urls(dataset_path: str, filenames: List[str]) -> List[str]:
    """Get Hugging Face download URLs for data files"""
    return [
        f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{f}"
        for f in filenames
    ]


def _download_files(urls: List[str], output_dir: Path) -> List[Path]:
    """Download a subset of a remote dataset's compressed files"""
    return download_files_sync(urls, output_dir)


def list_hf_files(repo_id, repo_type="dataset", extension=None):
    """List files from a Hugging Face repository.

    Args:
        repo_id: Repository ID
        repo_type: Type of repository ('dataset', 'model', etc.)
        extension: Filter files by extension

    Returns:
        List of files in the repository
    """
    files = []
    try:
        import huggingface_hub as hf_hub

        files = list(hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        if extension:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]
    except Exception as e:
        bt.logging.error(f"Failed to list files of type {extension} in {repo_id}: {e}")
    return files


def _is_zip_file(filename_lower: str) -> bool:
    """Return True if filename looks like a zip archive."""
    return filename_lower.endswith(".zip")


def _is_tar_file(filename_lower: str) -> bool:
    """Return True if filename looks like a tar archive (.tar, .tar.gz, .tgz)."""
    return filename_lower.endswith(".tar") or filename_lower.endswith(".tar.gz") or filename_lower.endswith(".tgz")


def download_files_sync(
    urls: List[str], output_dir: Path, chunk_size: int = 8192
) -> List[Path]:
    """Download multiple files synchronously.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time

    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for url in urls:
        try:
            downloaded_file = download_single_file_sync(url, output_dir, chunk_size)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
        except Exception as e:
            bt.logging.error(f"Error downloading {url}: {e}")

    return downloaded_files


def download_single_file_sync(
    url: str, output_dir: Path, chunk_size: int
) -> Optional[Path]:
    """Download a single file synchronously.

    Args:
        url: URL to download
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time

    Returns:
        Path to the downloaded file, or None if failed
    """
    try:
        bt.logging.info(f"Downloading {url}")

        response = requests.get(url, stream=True, timeout=3600)
        if response.status_code != 200:
            bt.logging.error(f"Failed to download {url}: Status {response.status_code}")
            return None

        filename = os.path.basename(url)
        filepath = output_dir / filename

        bt.logging.trace(f"Writing to {filepath}")

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        return filepath

    except Exception as e:
        bt.logging.error(f"Error downloading {url}: {str(e)}")
        bt.logging.error(traceback.format_exc())
        return None
