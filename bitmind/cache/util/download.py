import os
import traceback
from pathlib import Path
from typing import List, Union, Optional

import asyncio
import aiohttp
import bittensor as bt
import huggingface_hub as hf_hub
from requests.exceptions import RequestException


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
        files = list(hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        if extension:
            files = [f for f in files if f.endswith(extension)]
    except Exception as e:
        bt.logging.error(f"Failed to list files of type {extension} in {repo_id}: {e}")
    return files


async def download_files(
    urls: List[str], output_dir: Union[str, Path], chunk_size: int = 8192
) -> List[Path]:
    """Download multiple files asynchronously.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time

    Returns:
        List of successfully downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    timeout = aiohttp.ClientTimeout(
        total=3600,
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Create download tasks for each URL
        for url in urls:
            download_tasks.append(
                download_single_file(session, url, output_dir, chunk_size)
            )

        # Run all downloads concurrently and gather results
        downloaded_files = await asyncio.gather(*download_tasks, return_exceptions=True)

    # Filter out exceptions and return only successful downloads
    return [f for f in downloaded_files if isinstance(f, Path)]


async def download_single_file(
    session: aiohttp.ClientSession, url: str, output_dir: Path, chunk_size: int
) -> Path:
    """Download a single file asynchronously.

    Args:
        session: aiohttp ClientSession to use for requests
        url: URL to download
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time

    Returns:
        Path to the downloaded file
    """
    try:
        bt.logging.info(f"Downloading {url}")

        async with session.get(url) as response:
            if response.status != 200:
                bt.logging.error(f"Failed to download {url}: Status {response.status}")
                raise Exception(f"HTTP error {response.status}")

            filename = os.path.basename(url)
            filepath = output_dir / filename

            bt.logging.info(f"Writing to {filepath}")

            # Use async file I/O to write the file
            with open(filepath, "wb") as f:
                # Download and write in chunks
                async for chunk in response.content.iter_chunked(chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

            return filepath

    except Exception as e:
        bt.logging.error(f"Error downloading {url}: {str(e)}")
        bt.logging.error(traceback.format_exc())
        raise


def openvid1m_err_handler(
    base_zip_url: str,
    output_path: Path,
    part_index: int,
    chunk_size: int = 8192,
    timeout: int = 300,
) -> Optional[Path]:
    """Synchronous error handler for OpenVid1M downloads that handles split files.

    Args:
        base_zip_url: Base URL for the zip parts
        output_path: Directory to save files
        part_index: Index of the part to download
        chunk_size: Size of download chunks
        timeout: Download timeout in seconds

    Returns:
        Path to combined file if successful, None otherwise
    """
    part_urls = [
        f"{base_zip_url}{part_index}_partaa",
        f"{base_zip_url}{part_index}_partab",
    ]
    error_log_path = output_path / "download_log.txt"
    downloaded_parts = []

    # Download each part
    for part_url in part_urls:
        part_file_path = output_path / Path(part_url).name

        if part_file_path.exists():
            bt.logging.warning(f"File {part_file_path} exists.")
            downloaded_parts.append(part_file_path)
            continue

        try:
            response = requests.get(part_url, stream=True, timeout=timeout)
            if response.status_code != 200:
                raise RequestException(
                    f"HTTP {response.status_code}: {response.reason}"
                )

            with open(part_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

            bt.logging.info(f"File {part_url} saved to {part_file_path}")
            downloaded_parts.append(part_file_path)

        except Exception as e:
            error_message = f"File {part_url} download failed: {str(e)}\n"
            bt.logging.error(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            return None

    if len(downloaded_parts) == len(part_urls):
        try:
            combined_file = output_path / f"OpenVid_part{part_index}.zip"
            combined_data = bytearray()
            for part_path in downloaded_parts:
                with open(part_path, "rb") as part_file:
                    combined_data.extend(part_file.read())

            with open(combined_file, "wb") as out_file:
                out_file.write(combined_data)

            for part_path in downloaded_parts:
                part_path.unlink()

            bt.logging.info(f"Successfully combined parts into {combined_file}")
            return combined_file

        except Exception as e:
            error_message = (
                f"Failed to combine parts for index {part_index}: {str(e)}\n"
            )
            bt.logging.error(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            return None

    return None
