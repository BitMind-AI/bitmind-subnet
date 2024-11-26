from pathlib import Path
from typing import Union, Callable
from zipfile import ZipFile, BadZipFile
from enum import Enum, auto
import asyncio
import pyarrow.parquet as pq
import bittensor as bt


def seconds_to_str(seconds):
    seconds = int(float(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_most_recent_update_time(directory: Path) -> float:
    """Get the most recent modification time of any file in directory."""
    try:
        mtimes = [f.stat().st_mtime for f in directory.iterdir()]
        return max(mtimes) if mtimes else 0
    except Exception as e:
        bt.logging.error(f"Error getting modification times: {e}")
        return 0


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


def is_zip_complete(zip_path: Union[str, Path], testzip=False) -> bool:
    """
    Args:
        zip_path: Path to zip file
        testzip: More thorough, less efficient
    Returns:
        bool: True if zip is valid, False otherwise
    """
    try:
        with ZipFile(zip_path) as zf:
            if testzip:
                zf.testzip()
            else:
                zf.namelist()
            return True
    except (BadZipFile, Exception) as e:
        bt.logging.error(f"Zip file {zip_path} is invalid: {e}")
        return False
        

def is_parquet_complete(path: Path) -> bool:
    """    
    Args:
        path: Path to the parquet file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with open(path, 'rb') as f:
            pq.read_metadata(f)
        return True
    except Exception as e:
        bt.logging.error(f"Parquet file {path} is incomplete or corrupted: {e}")
        return False

