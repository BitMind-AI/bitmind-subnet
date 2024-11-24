from pathlib import Path
from typing import Union
from zipfile import ZipFile, BadZipFile
import asyncio
import pyarrow.parquet as pq
import bittensor as bt


def get_most_recent_update_time(directory: Path) -> float:
    """Get the most recent modification time of any file in directory."""
    try:
        mtimes = [f.stat().st_mtime for f in directory.iterdir()]
        return max(mtimes) if mtimes else 0
    except Exception as e:
        bt.logging.error(f"Error getting modification times: {e}")
        return 0


async def is_zip_complete(zip_path: Union[str, Path]) -> bool:
    """
    Asynchronously check if a zip file is complete and valid.
    
    Args:
        zip_path: Path to zip file
        
    Returns:
        bool: True if zip is complete and valid, False otherwise
    """
    async def _check_zip() -> bool:
        try:
            with ZipFile(zip_path) as zf:
                zf.testzip()
                return True
        except (BadZipFile, Exception) as e:
            bt.logging.error(f"Zip file {zip_path} is incomplete or corrupted: {e}")
            return False
            
    return await asyncio.to_thread(_check_zip)


async def is_parquet_complete(path: Path) -> bool:
    """
    Asynchronously verify if a parquet file is complete and not corrupted.
    
    Args:
        path: Path to the parquet file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    async def _check_parquet() -> bool:
        try:
            with open(path, 'rb') as f:
                pq.read_metadata(f)
            return True
        except Exception as e:
            bt.logging.error(f"Parquet file {path} is incomplete or corrupted: {e}")
            return False
            
    return await asyncio.to_thread(_check_parquet)


