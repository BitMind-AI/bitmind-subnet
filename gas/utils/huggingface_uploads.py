import os
import time
import json
import tempfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

import bittensor as bt
import pandas as pd
from huggingface_hub import HfApi, create_repo, CommitOperationAdd, hf_hub_download

from gas.utils.model_zips import calculate_sha256


def get_current_time_split() -> str:
    """
    Uses ISO week format: train_2025_W03, train_2025_W04, etc.

    Note: HuggingFace split names must match regex ^\w+(\.\w+)*$
    (word characters and dots only), so we use underscores instead of hyphens.

    Returns:
        str: Time-based split name (e.g., "train_2025_W03")
    """
    now = datetime.now(timezone.utc)
    year = now.year
    iso_week = now.isocalendar()[1]  # (1-53)
    return f"train_{year}_W{iso_week:02d}"


def upload_images_to_hf(
    media_entries: List[Any], hf_token: str, dataset_repo: str
) -> List[str]:
    """Upload images and metadata to hf"""
    return upload_media_to_hf(media_entries, hf_token, dataset_repo, "image")


def upload_videos_to_hf(
    media_entries: List[Any], hf_token: str, dataset_repo: str, videos_per_archive: int
) -> List[str]:
    """Upload video .tar.gz files and metadata to hf"""
    return upload_media_to_hf(
        media_entries,
        hf_token,
        dataset_repo,
        "video",
        videos_per_archive=videos_per_archive,
    )


def upload_media_to_hf(
    media_entries: List[Any], hf_token: str, dataset_repo: str, modality: str, **kwargs
) -> List[str]:
    """
    Common upload logic for media files to HuggingFace dataset repository.

    Returns:
        List of successfully uploaded media IDs
    """
    try:
        hf_api = HfApi(token=hf_token)

        bt.logging.info(
            f"Preparing to upload {len(media_entries)} {modality} files to {dataset_repo}"
        )
        try:
            create_repo(
                repo_id=dataset_repo, repo_type="dataset", exist_ok=True, token=hf_token
            )
        except Exception as e:
            if (
                "429" in str(e)
                or "rate" in str(e).lower()
                or "too many requests" in str(e).lower()
            ):
                bt.logging.warning(
                    "Hit HuggingFace rate limit during repo creation. Stopping upload and will retry next cycle."
                )
                return []
            raise

        upload_files, metadata_entries, successfully_processed_ids = (
            _process_media_entries(media_entries, modality)
        )

        if not upload_files:
            bt.logging.warning(f"No valid {modality} files to upload in batch")
            return []

        # prep upload operations
        if modality == "image":
            operations = _prepare_image_operations(upload_files, metadata_entries)
        elif modality == "video":
            operations, temp_files = _prepare_video_operations(
                upload_files, metadata_entries, kwargs.get("videos_per_archive", 25)
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        if not operations:
            return []

        # upload
        commit_info = hf_api.create_commit(
            repo_id=dataset_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload batch of {len(upload_files)} {modality} files and metadata",
        )

        bt.logging.info(
            f"Successfully uploaded {len(upload_files)} {modality} files to {dataset_repo} (commit: {commit_info.oid[:8]})"
        )

        # Cleanup
        if modality == "video" and "temp_files" in locals():
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception as cleanup_error:
                    bt.logging.warning(
                        f"Failed to cleanup temp file {temp_file}: {cleanup_error}"
                    )

        return successfully_processed_ids

    except Exception as e:
        if (
            "429" in str(e)
            or "rate" in str(e).lower()
            or "too many requests" in str(e).lower()
        ):
            bt.logging.warning(
                f"Hit HuggingFace rate limit during {modality} upload. Stopping upload and will retry next cycle."
            )
            return []
        bt.logging.error(f"Failed to upload {modality} files: {e}")
        return []


def _process_media_entries(
    media_entries: List[Any], modality: str
) -> Tuple[List[Tuple], List[Dict], List[str]]:
    """Process media entries into upload files, metadata, and success IDs."""
    upload_files = []
    metadata_entries = []
    successfully_processed_ids = []

    for media_entry in media_entries:
        media_path = Path(media_entry.file_path)
        if not media_path.exists():
            bt.logging.warning(f"{modality.title()} file not found: {media_path}")
            continue

        file_extension = media_path.suffix
        timestamp = int(time.time())
        filename = f"{timestamp}_{media_entry.id[:8]}{file_extension}"
        repo_path = f"{modality}s/{filename}"  # images/ or videos/

        upload_files.append((media_path, repo_path))

        with open(media_path, "rb") as f:
            content = f.read()
            media_hash = calculate_sha256(content)[:16]

        prompt_content = getattr(media_entry, "prompt_content", "") or ""

        metadata_entries.append(
            {
                "filename": filename,
                "media_id": media_entry.id,
                "generator_hotkey": getattr(media_entry, "hotkey", "validator"),
                "generator_uid": getattr(media_entry, "uid", 0),
                "prompt_id": media_entry.prompt_id,
                "prompt_content": prompt_content,
                "model_name": media_entry.model_name,
                "media_hash": media_hash,
                "timestamp": str(timestamp),
                "media_type": file_extension,
                "modality": media_entry.modality.value,
                "source_type": media_entry.source_type.value,
                "verified": getattr(media_entry, "verified", None),
            }
        )

        successfully_processed_ids.append(media_entry.id)

    return upload_files, metadata_entries, successfully_processed_ids


def _prepare_image_operations(
    upload_files: List[Tuple], metadata_entries: List[Dict]
) -> List[CommitOperationAdd]:
    """Prepare operations for chunked Parquet with embedded images"""
    operations = []
    images_per_chunk = 25
    timestamp = int(time.time())
    time_split = get_current_time_split()  # e.g., "train-2025-W03"

    bt.logging.info(f"Uploading images to time-based split: {time_split}")

    for chunk_idx in range(0, len(upload_files), images_per_chunk):
        chunk_files = upload_files[chunk_idx : chunk_idx + images_per_chunk]
        chunk_metadata = metadata_entries[chunk_idx : chunk_idx + images_per_chunk]

        # Create chunk entries with embedded image bytes
        chunk_entries = []
        for (local_path, repo_path), metadata in zip(chunk_files, chunk_metadata):
            try:
                with open(local_path, "rb") as f:
                    image_bytes = f.read()

                entry = metadata.copy()
                entry["image"] = (
                    image_bytes  # HF will auto-detect bytes and convert to PIL
                )
                entry["file_name"] = metadata["filename"]
                entry["file_path"] = repo_path
                entry["time_split"] = time_split
                entry["upload_timestamp"] = timestamp

                chunk_entries.append(entry)

            except Exception as e:
                bt.logging.error(f"Failed to process image {local_path}: {e}")
                continue

        if not chunk_entries:
            bt.logging.warning(f"No valid images in chunk {chunk_idx}")
            continue

        # Create time-organized Parquet chunk with standard HuggingFace naming
        total_chunks = (len(upload_files) - 1) // images_per_chunk + 1
        chunk_number = chunk_idx // images_per_chunk

        # Organize by time split: data/{time_split}/{chunk_filename}
        chunk_filename = f"data/{time_split}/{time_split}_{timestamp:010d}_{chunk_number:05d}_of_{total_chunks:05d}.parquet"

        df = pd.DataFrame(chunk_entries)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            temp_parquet_path = temp_file.name
            df.to_parquet(temp_parquet_path, index=False)

        operations.append(
            CommitOperationAdd(
                path_in_repo=chunk_filename, path_or_fileobj=temp_parquet_path
            )
        )

        bt.logging.info(
            f"Created self-contained image chunk {chunk_filename} with {len(chunk_entries)} images + metadata"
        )

    return operations


def _prepare_video_operations(
    upload_files: List[Tuple], metadata_entries: List[Dict], videos_per_archive: int
) -> Tuple[List[CommitOperationAdd], List[str]]:
    """Prepare operations for video archive upload with time-based splits."""
    archive_uploads = create_video_archives(
        upload_files, metadata_entries, videos_per_archive
    )

    if not archive_uploads:
        bt.logging.warning("No video archives created")
        return [], []

    operations = []
    temp_files = []
    time_split = get_current_time_split()  # e.g., "train-2025-W03"

    bt.logging.info(f"Uploading videos to time-based split: {time_split}")

    for archive_path, archive_metadata_list in archive_uploads:
        temp_files.append(archive_path)
        for entry in archive_metadata_list:
            entry["time_split"] = time_split
            entry["upload_timestamp"] = int(time.time())

        # Organize archives by time split: archives/{time_split}/{archive_name}
        archive_name = Path(archive_path).name
        archive_filename = f"archives/{time_split}/{archive_name}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=archive_filename, path_or_fileobj=str(archive_path)
            )
        )

        # Organize metadata by time split: metadata/{time_split}/{metadata_name}
        metadata_name = f"{Path(archive_path).stem}_metadata.json"
        metadata_filename = f"metadata/{time_split}/{metadata_name}"
        metadata_content = json.dumps(archive_metadata_list, indent=2)
        operations.append(
            CommitOperationAdd(
                path_in_repo=metadata_filename,
                path_or_fileobj=metadata_content.encode(),
            )
        )

        bt.logging.info(f"Created time-organized video archive {archive_filename}")

    return operations, temp_files


def create_video_archives(
    upload_files: List[Tuple[str, str]],
    metadata_entries: List[Dict[str, Any]],
    videos_per_archive: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Create tar.gz archives containing videos with their metadata.

    Args:
        upload_files: List of (local_path, repo_path) tuples for videos
        metadata_entries: List of metadata dictionaries corresponding to videos
        videos_per_archive: Maximum number of videos per archive (default: 25)

    Returns:
        List of (archive_path, archive_metadata_list) tuples

    Raises:
        ValueError: If videos_per_archive is invalid
        Exception: If archive creation fails
    """
    if not upload_files:
        bt.logging.warning("No upload files provided for video archiving")
        return []

    if videos_per_archive is None or videos_per_archive <= 0:
        bt.logging.warning(
            f"Invalid videos_per_archive: {videos_per_archive}. Using default of 25."
        )
        videos_per_archive = 25

    if len(upload_files) != len(metadata_entries):
        raise ValueError(
            f"Mismatch between upload_files ({len(upload_files)}) and metadata_entries ({len(metadata_entries)})"
        )

    archives = []
    temp_files_to_cleanup = []
    timestamp = int(time.time())

    bt.logging.info(
        f"Creating video archives with max {videos_per_archive} videos per archive"
    )

    try:
        for i in range(0, len(upload_files), videos_per_archive):
            batch_files = upload_files[i : i + videos_per_archive]
            batch_metadata = metadata_entries[i : i + videos_per_archive]

            archive_fd, archive_path = tempfile.mkstemp(
                suffix=f"_video_batch_{i//videos_per_archive + 1}_{timestamp}.tar.gz"
            )
            temp_files_to_cleanup.append(archive_path)

            os.close(
                archive_fd
            )  # Close the file descriptor since tarfile will handle the file

            # Create tar.gz archive
            with tarfile.open(archive_path, "w:gz") as tar:
                archive_metadata_list = []

                for (local_path, repo_path), metadata in zip(
                    batch_files, batch_metadata
                ):
                    local_path = Path(local_path)
                    if not local_path.exists():
                        bt.logging.warning(f"Skipping missing video file: {local_path}")
                        continue

                    tar.add(str(local_path), arcname=repo_path)

                    archive_metadata = metadata.copy()
                    archive_metadata["archive_filename"] = Path(archive_path).name
                    archive_metadata["video_path_in_archive"] = repo_path
                    archive_metadata_list.append(archive_metadata)

                    bt.logging.debug(f"Added {local_path} to archive as {repo_path}")

                if archive_metadata_list:  # Only keep non-empty archives
                    archives.append((archive_path, archive_metadata_list))
                    temp_files_to_cleanup.remove(
                        archive_path
                    )  # Don't cleanup successful archives
                    bt.logging.info(
                        f"Created archive {Path(archive_path).name} with {len(archive_metadata_list)} videos"
                    )
                else:
                    bt.logging.warning("Skipped empty archive (no valid videos found)")

    except Exception as e:
        bt.logging.error(f"Error creating video archives: {e}")
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise

    if not archives:
        bt.logging.warning("No video archives created")
    else:
        bt.logging.info(f"Successfully created {len(archives)} video archives")

    return archives
