import base64
import hashlib
import json
import logging
import mimetypes
import os
import random
import warnings
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from zipfile import ZipFile

from PIL import Image
import pyarrow.parquet as pq
import bittensor as bt


def extract_videos_from_zip(
    zip_path: Path,
    dest_dir: Path,
    num_videos: int,
    file_extensions: Set[str] = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'},
    include_checksums: bool = True
) -> List[Tuple[str, str]]:
    """
    Extract random videos and their metadata from a zip file and save them to disk.
q
    Args:
        zip_path: Path to the zip file
        dest_dir: Directory to save videos and metadata
        num_videos: Number of videos to extract
        file_extensions: Set of valid video file extensions
        include_checksums: Whether to calculate and include file checksums in metadata

    Returns:
        List of tuples containing (video_path, metadata_path)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    try:
        with ZipFile(zip_path) as zip_file:
            video_files = [
                f for f in zip_file.namelist()
                if any(f.lower().endswith(ext) for ext in file_extensions)
            ]
            if not video_files:
                bt.logging.warning(f"No video files found in {zip_path}")
                return extracted_files

            bt.logging.info(f"{len(video_files)} video files found in {zip_path}")
            selected_videos = random.sample(
                video_files,
                min(num_videos, len(video_files))
            )

            bt.logging.info(f"Extracting {len(selected_videos)} randomly sampled video files from {zip_path}")
            for idx, video in enumerate(selected_videos):
                if 'MACOSX' in video:
                    continue
                try:
                    # extract video and get metadata
                    video_path = dest_dir /  Path(video).name
                    with zip_file.open(video) as source:
                        with open(video_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

                    video_info = zip_file.getinfo(video)
                    metadata = {
                        'source_zip': str(zip_path),
                        'path_in_zip': video,
                        'extraction_date': datetime.now().isoformat(),
                        'file_size': os.path.getsize(video_path),
                        'zip_metadata': {
                            'compress_size': video_info.compress_size,
                            'file_size': video_info.file_size,
                            'compress_type': video_info.compress_type,
                            'date_time': datetime.strftime(
                                datetime(*video_info.date_time),
                                '%Y-%m-%d %H:%M:%S'
                            ),
                        }
                    }

                    if include_checksums:
                        with open(video_path, 'rb') as f:
                            file_data = f.read()
                            metadata['checksums'] = {
                                'md5': hashlib.md5(file_data).hexdigest(),
                                'sha256': hashlib.sha256(file_data).hexdigest()
                            }

                    metadata_filename = f"{video_path.stem}.json"
                    metadata_path = dest_dir / metadata_filename

                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    extracted_files.append((str(video_path), str(metadata_path)))
                    logging.info(f"Extracted {Path(video).name} from {zip_path}")

                except Exception as e:
                    bt.logging.warning(f"Error extracting {video}: {e}")
                    continue

    except Exception as e:
        bt.logging.warning(f"Error processing zip file {zip_path}: {e}")

    return extracted_files


def extract_images_from_parquet(
    parquet_path: Path,
    dest_dir: Path,
    num_images: int,
    seed: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Extract random images and their metadata from a parquet file and save them to disk.

    Args:
        parquet_path: Path to the parquet file
        dest_dir: Directory to save images and metadata
        num_images: Number of images to extract
        columns: Specific columns to include in metadata
        seed: Random seed for sampling

    Returns:
        List of tuples containing (image_path, metadata_path)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # read parquet file, sample random image rows
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    sample_df = df.sample(n=min(num_images, len(df)), random_state=seed)
    image_col = next((col for col in sample_df.columns if 'image' in col.lower()), None)
    metadata_cols = [c for c in sample_df.columns if c != image_col]

    saved_files = []
    parquet_prefix = parquet_path.stem
    for idx, row in sample_df.iterrows():
        try:
            img_data = row[image_col]
            if isinstance(img_data, dict):
                key = next((k for k in img_data if 'bytes' in k.lower() or 'image' in k.lower()), None)
                img_data = img_data[key]

            try:
                img = Image.open(BytesIO(img_data))
            except Exception as e:
                img_data = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_data))

            base_filename = f"{parquet_prefix}__image_{idx}"
            image_format = img.format.lower() if img.format else 'png'
            img_filename = f"{base_filename}.{image_format}"
            img_path = dest_dir / img_filename
            img.save(img_path)

            metadata = {
                'source_parquet': str(parquet_path),
                'original_index': str(idx),
                'image_format': image_format,
                'image_size': img.size,
                'image_mode': img.mode
            }

            for col in metadata_cols:
                # Convert any non-serializable types to strings
                try:
                    json.dumps({col: row[col]})
                    metadata[col] = row[col]
                except (TypeError, OverflowError):
                    metadata[col] = str(row[col])
    
            metadata_filename = f"{base_filename}.json"
            metadata_path = dest_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
    
            saved_files.append(str(img_path))

        except Exception as e:
            bt.logging.warning(f"Failed to extract/save image {idx}: {e}")
            continue

    return saved_files