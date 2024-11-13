from typing import Optional, List, Tuple
from datasets import Dataset
import av
import numpy as np
from io import BytesIO
import requests
import tempfile
import os
from pathlib import Path

from .base_dataset import BaseDataset


def download_video(url: str) -> Optional[str]:
    """Download a video from a URL to a temporary file.
    
    Args:
        url (str): URL of the video to download
        
    Returns:
        str: Path to temporary file containing the video, or None if download failed
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create a temporary file with .mp4 extension
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp.close()
            return tmp.name
        return None
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


class VideoDataset(BaseDataset):
    def __init__(
        self,
        huggingface_dataset_path: Optional[str] = None,
        huggingface_dataset_split: str = 'train',
        huggingface_dataset_name: Optional[str] = None,
        huggingface_dataset: Optional[Dataset] = None,
        download_mode: Optional[str] = None,
        max_frames: int = 32,  # Maximum number of frames to extract
        frame_interval: int = 1  # Interval between extracted frames
    ):
        """Initialize the VideoDataset.
        
        Args:
            huggingface_dataset_path (str, optional): Path to the Hugging Face dataset.
            huggingface_dataset_split (str): Dataset split to load. Defaults to 'train'.
            huggingface_dataset_name (str, optional): Name of the specific Hugging Face dataset subset.
            huggingface_dataset (Dataset, optional): Pre-loaded Hugging Face dataset instance.
            download_mode (str, optional): Download mode for the dataset.
            max_frames (int): Maximum number of frames to extract from each video.
            frame_interval (int): Number of frames to skip between extracted frames.
        """
        # Call parent class initialization
        super().__init__(
            huggingface_dataset_path=huggingface_dataset_path,
            huggingface_dataset_split=huggingface_dataset_split,
            huggingface_dataset_name=huggingface_dataset_name,
            huggingface_dataset=huggingface_dataset,
            download_mode=download_mode
        )
        
        # Load the dataset if not already loaded by parent class
        if self.dataset is None:
            self.dataset = load_huggingface_dataset(
                self.huggingface_dataset_path,
                self.huggingface_dataset_split,
                self.huggingface_dataset_name,
                download_mode
            )
        
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self._temp_files = set()  # Track temporary files for cleanup
        
    def __getitem__(self, index: int) -> dict:
        """Get video frames and metadata from the dataset.
        
        Args:
            index (int): Index of the video in the dataset.
            
        Returns:
            dict: Dictionary containing:
                - 'frames': numpy array of shape (num_frames, height, width, 3)
                - 'id': str identifier for the video
                - 'source': dataset source
                - 'fps': original video fps
                - 'num_frames': total number of frames extracted
        """
        sample = self.dataset[int(index)]
        video_path = None
        video_id = str(index)
        
        try:
            # Handle different video source formats
            if 'video' in sample:
                if isinstance(sample['video'], bytes):
                    # Handle bytes data
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tmp.write(sample['video'])
                    tmp.close()
                    video_path = tmp.name
                    self._temp_files.add(video_path)
                elif isinstance(sample['video'], str) and os.path.exists(sample['video']):
                    video_path = sample['video']
            elif 'video_url' in sample:
                video_path = download_video(sample['video_url'])
                if video_path:
                    self._temp_files.add(video_path)
                video_id = sample['video_url']
            elif 'url' in sample:
                video_path = download_video(sample['url'])
                if video_path:
                    self._temp_files.add(video_path)
                video_id = sample['url']
                
            # Extract video ID from metadata if available
            if 'name' in sample:
                video_id = sample['name']
            elif 'filename' in sample:
                video_id = sample['filename']
            
            if video_path is None or not os.path.exists(video_path):
                raise FileNotFoundError("Video file not found or download failed")
            
            # Extract frames using PyAV
            frames = []
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                
                for i, frame in enumerate(container.decode(video=0)):
                    if i % self.frame_interval == 0 and len(frames) < self.max_frames:
                        # Convert to RGB numpy array
                        img = frame.to_ndarray(format='rgb24')
                        frames.append(img)
                    elif len(frames) >= self.max_frames:
                        break
            
            frames = np.stack(frames) if frames else np.array([])
            
            return {
                'frames': frames,
                'id': video_id,
                'source': self.huggingface_dataset_path,
                'fps': fps,
                'num_frames': len(frames)
            }
            
        except Exception as e:
            print(f"Error loading video at index {index}: {e}")
            return {
                'frames': np.array([]),
                'id': video_id,
                'source': self.huggingface_dataset_path,
                'fps': 0,
                'num_frames': 0
            }
            
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)
    
    def cleanup(self):
        """Clean up any temporary files created during video loading."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Error deleting temporary file {temp_file}: {e}")
        self._temp_files.clear()
        
    def __del__(self):
        """Ensure cleanup of temporary files when the dataset is destroyed."""
        self.cleanup()