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
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image

from gas.utils.model_zips import calculate_sha256


def get_current_time_split() -> str:
    """
    Uses ISO week format for subset/config names: 2025W03, 2025W04, etc.

    This creates weekly dataset subsets/configs, not splits.
    Users load them like: load_dataset('repo', '2025W36')

    Note: HF split names must match ^\w+(\.\w+)*$ (no hyphens allowed)

    Returns:
        str: Weekly subset/config name (e.g., "2025W03")
    """
    now = datetime.now(timezone.utc)
    year = now.year
    iso_week = now.isocalendar()[1]  # (1-53)
    return f"{year}W{iso_week:02d}"


def upload_images_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    validator_hotkey: str = None,
) -> List[str]:
    """Upload images and metadata to hf"""
    return upload_media_to_hf(
        media_entries,
        hf_token,
        dataset_repo,
        "image",
        validator_hotkey=validator_hotkey,
    )


def upload_videos_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    videos_per_archive: int,
    validator_hotkey: str = None,
) -> List[str]:
    """Upload video .tar.gz files and metadata to hf"""
    return upload_media_to_hf(
        media_entries,
        hf_token,
        dataset_repo,
        "video",
        videos_per_archive=videos_per_archive,
        validator_hotkey=validator_hotkey,
    )


def upload_media_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    modality: str,
    **kwargs
) -> List[str]:
    """
    Core hf dataset upload logic

    Returns:
        List of successfully uploaded media IDs
    """
    try:
        hf_api = HfApi(token=hf_token)

        bt.logging.info(f"Preparing to upload {len(media_entries)} {modality} files to {dataset_repo}")
        try:
            create_repo(
                repo_id=dataset_repo,
                repo_type="dataset",
                exist_ok=True,
                token=hf_token
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

        if modality == "image":
            dataset, config_name = _prepare_image_operations(
                upload_files, metadata_entries
            )
            if not dataset:
                return []

            try:
                create_repo(
                    dataset_repo, 
                    repo_type="dataset", 
                    exist_ok=True, 
                    token=hf_token
                )
            except:
                pass  # Repo exists

            operations, temp_files = _prepare_shard_operations(
                dataset,
                config_name,
                kwargs.get("validator_hotkey")
            )

            if operations:
                commit_info = hf_api.create_commit(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Add new image shard to {config_name}",
                )
                bt.logging.info(
                    f"Successfully added {len(dataset)} images as new shard to {dataset_repo} "
                    f"config {config_name} (commit: {commit_info.oid[:8]})"
                )

                # Cleanup temp files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        elif modality == "video":
            dataset, config_name, archive_operations, temp_files, metadata_operations, metadata_temp_files = (
                _prepare_video_dataset_and_archives(
                    upload_files,
                    metadata_entries,
                    kwargs.get("videos_per_archive", 25),
                    kwargs.get("validator_hotkey"),
                )
            )
            if not dataset:
                return []

            try:
                create_repo(
                    dataset_repo, repo_type="dataset", exist_ok=True, token=hf_token
                )
            except:
                pass  # Repo exists

            # combine the metadata and archive ops here (metadata operations are created per-archive)
            all_operations = metadata_operations + archive_operations
            all_temp_files = metadata_temp_files + temp_files

            if all_operations:
                commit_info = hf_api.create_commit(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    operations=all_operations,
                    commit_message=f"Add new video shard and archives to {config_name}",
                )
                bt.logging.info(
                    f"Successfully added video metadata as {len(metadata_operations)} separate shards (1:1 with archives) and {len(upload_files)} "
                    f"archives to {dataset_repo} config {config_name} (commit: {commit_info.oid[:8]})"
                )

                # Cleanup temp files
                for temp_file in all_temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        return successfully_processed_ids

    except Exception as e:
        if (
            "429" in str(e)
            or "rate" in str(e).lower()
            or "too many requests" in str(e).lower()
        ):
            bt.logging.warning(
                f"Hit HuggingFace rate limit during {modality} upload. "
                "Stopping upload and will retry next cycle."
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
        repo_path = f"{modality}s/{filename}"

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
                "media_type": media_entry.media_type.value,
                "modality": media_entry.modality.value,
                "source_type": media_entry.source_type.value,
                "verified": getattr(media_entry, "verified", None),
            }
        )

        successfully_processed_ids.append(media_entry.id)

    return upload_files, metadata_entries, successfully_processed_ids


def _prepare_image_operations(
    upload_files: List[Tuple], metadata_entries: List[Dict]
) -> tuple:
    """Create weekly config using push_to_hub approach"""
    time_split = get_current_time_split()  # e.g., "2025W38"

    bt.logging.info(f"Creating weekly config: {time_split}")
    dataset_dict = {
        "media_hash": [],
        "model_name": [],
        "generator_hotkey": [],
        "generator_uid": [],
        "prompt_str": [],
        "timestamp": [],
        "media_type": [],
        "image": [],  # PIL Images
    }

    for (local_path, repo_path), metadata in zip(upload_files, metadata_entries):
        try:
            img = Image.open(local_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Force the image to be fully loaded into memory to break file path reference
            img.load()
            img = img.copy()

            dataset_dict["media_hash"].append(metadata["media_hash"])
            dataset_dict["model_name"].append(metadata["model_name"])
            dataset_dict["generator_hotkey"].append(
                metadata.get("generator_hotkey", "validator")
            )
            dataset_dict["generator_uid"].append(metadata.get("generator_uid", 0))
            dataset_dict["prompt_str"].append(metadata.get("prompt_content", ""))
            dataset_dict["timestamp"].append(int(metadata["timestamp"]))
            dataset_dict["media_type"].append(metadata["media_type"])
            dataset_dict["image"].append(img)

        except Exception as e:
            bt.logging.error(f"Failed to process image {local_path}: {e}")
            continue

    if not dataset_dict["image"]:
        bt.logging.warning("No valid image data to upload")
        return None, None

    # Create Dataset with proper features (like working code)
    features = Features(
        {
            "media_hash": Value("string"),
            "model_name": Value("string"),
            "generator_hotkey": Value("string"),
            "generator_uid": Value("int64"),
            "prompt_str": Value("string"),
            "timestamp": Value("int64"),
            "media_type": Value("string"),
            "image": HFImage(),
        }
    )

    dataset = Dataset.from_dict(dataset_dict, features=features)
    bt.logging.info(
        f"Created weekly dataset with {len(dataset_dict['image'])} images for config {time_split}"
    )

    return dataset, time_split


def _prepare_shard_operations(
    dataset: Dataset, config_name: str, validator_hotkey: str = None, archive_suffix: str = ""
) -> tuple:
    """Create shard operations for any dataset (adds new shards instead of overwriting)"""
    import tempfile
    from huggingface_hub import CommitOperationAdd

    # Create unique shard filename with validator hotkey + timestamp
    timestamp = int(time.time())
    split_name = f"data_{config_name}"  # e.g., "data_2025W38"

    if validator_hotkey:
        hotkey_prefix = (
            validator_hotkey[:8] if len(validator_hotkey) >= 8 else validator_hotkey
        )
        shard_filename = f"{split_name}/shard-{hotkey_prefix}-{timestamp}{archive_suffix}.parquet"
        bt.logging.info(f"Creating shard with validator hotkey prefix: {shard_filename}")
    else:
        shard_filename = f"{split_name}/shard-{timestamp}{archive_suffix}.parquet"
        bt.logging.warning(f"No validator hotkey provided for shard: {shard_filename}")

    # Create temporary parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        temp_path = tmp_file.name

    # Save dataset as parquet
    dataset.to_parquet(temp_path)

    # Create commit operation
    operation = CommitOperationAdd(
        path_in_repo=shard_filename, path_or_fileobj=temp_path
    )

    bt.logging.info(f"Created shard: {shard_filename}")

    return [operation], [temp_path]


def _prepare_video_dataset_and_archives(
    upload_files: List[Tuple],
    metadata_entries: List[Dict],
    videos_per_archive: int,
    validator_hotkey: str = None,
) -> tuple:
    """Create video metadata dataset with configs AND archive operations together"""
    time_split = get_current_time_split()  # e.g., "2025W38"

    bt.logging.info(f"Creating video dataset and archives for config: {time_split}")
    archive_uploads = create_video_archives(
        upload_files, metadata_entries, videos_per_archive, validator_hotkey
    )

    if not archive_uploads:
        bt.logging.warning("No video archives created")
        return None, None, [], []

    operations = []
    temp_files = []
    all_metadata_operations = []
    all_metadata_temp_files = []

    # Process each archive and its metadata separately (1:1 mapping)
    for archive_index, (archive_path, archive_metadata_list) in enumerate(archive_uploads):
        temp_files.append(archive_path)
        archive_name = Path(archive_path).name

        # Add archive operation
        archive_filename = f"archives/{time_split}/{archive_name}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=archive_filename, path_or_fileobj=str(archive_path)
            )
        )

        # Create separate dataset for this archive's metadata
        archive_dataset_dict = {
            "media_hash": [],
            "model_name": [],
            "generator_hotkey": [],
            "generator_uid": [],
            "prompt_str": [],
            "timestamp": [],
            "media_type": [],
            "archive_filename": [],
            "video_path_in_archive": [],
        }

        # Add this archive's metadata to its own dataset
        for metadata in archive_metadata_list:
            archive_dataset_dict["media_hash"].append(metadata["media_hash"])
            archive_dataset_dict["model_name"].append(metadata["model_name"])
            archive_dataset_dict["generator_hotkey"].append(
                metadata.get("generator_hotkey", "validator")
            )
            archive_dataset_dict["generator_uid"].append(metadata.get("generator_uid", 0))
            archive_dataset_dict["prompt_str"].append(metadata.get("prompt_content", ""))
            archive_dataset_dict["timestamp"].append(int(metadata["timestamp"]))
            archive_dataset_dict["media_type"].append(metadata["media_type"])
            archive_dataset_dict["archive_filename"].append(archive_name)  # Just filename
            archive_dataset_dict["video_path_in_archive"].append(
                metadata["video_path_in_archive"]
            )

        if archive_dataset_dict["media_hash"]:  # Only create dataset if has data
            features = Features(
                {
                    "media_hash": Value("string"),
                    "model_name": Value("string"),
                    "generator_hotkey": Value("string"),
                    "generator_uid": Value("int64"),
                    "prompt_str": Value("string"),
                    "timestamp": Value("int64"),
                    "media_type": Value("string"),
                    "archive_filename": Value("string"),
                    "video_path_in_archive": Value("string"),
                }
            )

            archive_dataset = Dataset.from_dict(archive_dataset_dict, features=features)

            # Create dedicated parquet shard for this archive
            archive_metadata_operations, archive_metadata_temp_files = _prepare_shard_operations(
                archive_dataset, time_split, validator_hotkey, archive_suffix=f"_archive{archive_index:03d}"
            )

            all_metadata_operations.extend(archive_metadata_operations)
            all_metadata_temp_files.extend(archive_metadata_temp_files)

            bt.logging.info(
                f"Created 1:1 metadata dataset for archive {archive_name} with {len(archive_dataset_dict['media_hash'])} entries"
            )

    if not all_metadata_operations:
        bt.logging.warning("No valid video metadata to upload")
        return None, None, [], [], [], []

    # Return combined dataset info (even though we created separate shards)
    bt.logging.info(
        f"Created {len(archive_uploads)} separate metadata shards (1:1 with archives) for config {time_split}"
    )

    # Create a dummy combined dataset for return compatibility
    combined_dataset_dict = {"dummy": ["placeholder"]}
    features = Features({"dummy": Value("string")})
    dummy_dataset = Dataset.from_dict(combined_dataset_dict, features=features)

    return dummy_dataset, time_split, operations, temp_files, all_metadata_operations, all_metadata_temp_files


def create_video_archives(
    upload_files: List[Tuple[str, str]],
    metadata_entries: List[Dict[str, Any]],
    videos_per_archive: int,
    validator_hotkey: str = None,
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

    # Use single timestamp for entire upload batch (we need consistency with parquet metadata for hf autodetect to work)
    base_timestamp = int(time.time())

    bt.logging.info(f"Creating video archives with max {videos_per_archive} videos per archive")

    try:
        for i in range(0, len(upload_files), videos_per_archive):
            batch_files = upload_files[i : i + videos_per_archive]
            batch_metadata = metadata_entries[i : i + videos_per_archive]

            batch_index = i // videos_per_archive
            timestamp = base_timestamp + batch_index

            if validator_hotkey:
                # first 8 chars of hotkey + timestamp as file basenames
                hotkey_prefix = (
                    validator_hotkey[:8]
                    if len(validator_hotkey) >= 8
                    else validator_hotkey
                )
                archive_filename = f"{hotkey_prefix}_{timestamp}.tar.gz"
                bt.logging.info(
                    f"Creating video archive with validator hotkey prefix: {archive_filename}"
                )
            else:
                # Fallback to batch naming if no hotkey available
                archive_filename = f"validator_{i//videos_per_archive + 1}_{timestamp}.tar.gz"
                bt.logging.warning(f"No validator hotkey provided, using fallback naming: {archive_filename}")

            # Create tar.gz archive
            temp_dir = tempfile.gettempdir()
            archive_path = os.path.join(temp_dir, archive_filename)
            temp_files_to_cleanup.append(archive_path)

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
