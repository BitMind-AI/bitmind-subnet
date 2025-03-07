import requests
import os
from pathlib import Path
from requests.exceptions import RequestException
from typing import List, Union, Dict, Optional

import bittensor as bt
import huggingface_hub as hf_hub


def download_files(
    urls: List[str],
    output_dir: Union[str, Path],
    chunk_size: int = 8192
) -> List[Path]:
    """
    Downloads multiple files synchronously.
    
    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time
    
    Returns:
        List of successfully downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for url in urls:
        try:
            bt.logging.info(f'Downloading {url}')
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                bt.logging.error(f'Failed to download {url}: Status {response.status_code}')
                continue

            filename = os.path.basename(url)
            filepath = output_dir / filename

            bt.logging.info(f'Writing to {filepath}')
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

            downloaded_files.append(filepath)
            bt.logging.info(f'Successfully downloaded {filename}')

        except Exception as e:
            bt.logging.error(f'Error downloading {url}: {str(e)}')
            continue

    return downloaded_files


def list_hf_files(repo_id, repo_type='dataset', extension=None):
    files = []
    try:
        files = list(hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        if extension:
            files = [f for f in files if f.endswith(extension)]
    except Exception as e:
        bt.logging.error(f"Failed to list files of type {extension} in {repo_id}: {e}")
    return files


def openvid1m_err_handler(
    base_zip_url: str,
    output_path: Path,
    part_index: int,
    chunk_size: int = 8192,
    timeout: int = 300
) -> Optional[Path]:
    """
    Synchronous error handler for OpenVid1M downloads that handles split files.
    
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
        f"{base_zip_url}{part_index}_partab"
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
                
            with open(part_file_path, 'wb') as f:
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
                with open(part_path, 'rb') as part_file:
                    combined_data.extend(part_file.read())
                    
            with open(combined_file, 'wb') as out_file:
                out_file.write(combined_data)
                
            for part_path in downloaded_parts:
                part_path.unlink()
                
            bt.logging.info(f"Successfully combined parts into {combined_file}")
            return combined_file
            
        except Exception as e:
            error_message = f"Failed to combine parts for index {part_index}: {str(e)}\n"
            bt.logging.error(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            return None
    
    return None
