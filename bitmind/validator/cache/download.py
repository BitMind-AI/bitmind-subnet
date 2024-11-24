from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional, Union
import aiohttp
import aiofiles
import asyncio

import bittensor as bt


class FileType(Enum):
    PARQUET = auto()
    ZIP = auto()


def get_integrity_check(file_type: FileType) -> Callable[[Path], bool]:
    """Returns the appropriate validation function for the file type."""
    if file_type == FileType.PARQUET:
        return is_parquet_complete
    elif file_type == FileType.ZIP:
        return is_zip_complete
    raise ValueError(f"Unsupported file type: {file_type}")


async def download_file(
    url: str,
    destination: Path,
    file_type: FileType,
    timeout: int = 300,
    chunk_size: int = 8192,
    error_handler: Optional[Callable] = None
) -> Optional[Path]:
    """
    Asynchronously download a file from a URL.
    
    Args:
        url: URL of the file
        destination: Path to save the file
        file_type: Type of file (PARQUET or ZIP)
        timeout: Download timeout in seconds
        chunk_size: Size of chunks to download at a time
        error_handler: Optional function to handle download errors
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    file_path = destination / Path(url).name
    if file_path.exists():
        bt.logging.debug(f"File already exists: {file_path}")
        return file_path

    temp_path = file_path.with_suffix('.temp')
    integrity_check = get_integrity_check(file_type)
    
    try:
        timeout_client = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_client) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    bt.logging.error(
                        f"Failed to download {url}: HTTP {response.status}"
                    )
                    return None
                
                try:
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                            
                    if not await integrity_check(temp_path):
                        bt.logging.error(f"Downloaded file {url} is corrupted")
                        temp_path.unlink()
                        if error_handler:
                            return error_handler(url, destination)
                        return None
                        
                    # Rename temp file to final name
                    temp_path.rename(file_path)
                    bt.logging.info(f"Successfully downloaded {url} to {file_path}")
                    return file_path
                    
                except Exception as e:
                    bt.logging.error(f"Error downloading {url}: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
                    if error_handler:
                        return error_handler(url, destination)
                    return None
                    
    except Exception as e:
        bt.logging.error(f"Connection error downloading {url}: {e}")
        if error_handler:
            return error_handler(url, destination)
        return None


async def download_files(
    urls: List[str],
    destination: Path,
    file_type: FileType,
    max_concurrent: int = 5,
    timeout: int = 300,
    error_handler: Optional[Callable] = None
) -> List[Path]:
    """
    Asynchronously download multiple files.
    
    Args:
        urls: List of URLs to download
        destination: Directory to save files
        file_type: Type of files to download
        max_concurrent: Maximum number of concurrent downloads
        timeout: Download timeout per file in seconds
        error_handler: Optional function to handle download errors
        
    Returns:
        List of successfully downloaded file paths
    """
    sem = asyncio.Semaphore(max_concurrent)
    
    async def download_with_semaphore(url: str) -> Optional[Path]:
        async with sem:
            return await download_file(
                url, 
                destination,
                file_type,
                timeout=timeout,
                error_handler=error_handler
            )
            
    tasks = [download_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed downloads (None results)
    return [r for r in results if r is not None]


async def openvid1m_err_handler(
    base_zip_url: str,
    output_path: Path,
    part_index: int,
    chunk_size: int = 8192,
    timeout: int = 300
) -> Optional[Path]:
    """
    Async error handler for OpenVid1M downloads that handles split files.
    
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
            timeout_client = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_client) as session:
                async with session.get(part_url) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(
                            f"HTTP {response.status}: {response.reason}"
                        )
                        
                    async with aiofiles.open(part_file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                            
                    bt.logging.info(f"File {part_url} saved to {part_file_path}")
                    downloaded_parts.append(part_file_path)
                    
        except Exception as e:
            error_message = f"File {part_url} download failed: {str(e)}\n"
            bt.logging.error(error_message)
            async with aiofiles.open(error_log_path, "a") as error_log_file:
                await error_log_file.write(error_message)
            return None

    # If we got both parts, combine them
    if len(downloaded_parts) == len(part_urls):
        try:
            combined_file = output_path / f"OpenVid_part{part_index}.zip"
            
            # Read all parts and combine
            combined_data = bytearray()
            for part_path in downloaded_parts:
                async with aiofiles.open(part_path, 'rb') as part_file:
                    combined_data.extend(await part_file.read())
                    
            # Write combined file
            async with aiofiles.open(combined_file, 'wb') as out_file:
                await out_file.write(combined_data)
                
            # Clean up part files
            for part_path in downloaded_parts:
                part_path.unlink()
                
            bt.logging.info(f"Successfully combined parts into {combined_file}")
            return combined_file
            
        except Exception as e:
            error_message = f"Failed to combine parts for index {part_index}: {str(e)}\n"
            bt.logging.error(error_message)
            async with aiofiles.open(error_log_path, "a") as error_log_file:
                await error_log_file.write(error_message)
            return None
    
    return None

"""
data_folder = output_path / "data" / "train"
data_folder.mkdir(parents=True, exist_ok=True)
data_urls = [
    "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
    "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
]
for data_url in data_urls:
    data_path = data_folder / Path(data_url).name
    command = ["wget", "-O", str(data_path), data_url]
    subprocess.run(command, check=True)
"""
