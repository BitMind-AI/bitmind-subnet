import os
import requests
import time
from abc import ABC, abstractmethod
from PIL import Image
from io import BytesIO
import bittensor as bt
import json

from bitmind.types import Modality


class BaseScraper(ABC):
    """
    Base class for image scrapers with common functionality
    
    Parameters:
    -----------
    min_width : int
        Minimum width for downloaded images (default: 128)
    min_height : int
        Minimum height for downloaded images (default: 128)
    """
    
    def __init__(self, min_width=128, min_height=128, media_type=None):
        self.min_width = min_width
        self.min_height = min_height
        self.media_type = media_type.value     # for reference when determining save location
        self.modality = Modality.IMAGE.value   # does not support video yet
    
    def _check_image_size(self, url):
        """
        Check if image meets minimum size requirements
        
        Parameters:
        -----------
        url : str
            Image URL to check
            
        Returns:
        --------
        tuple
            (meets_requirements: bool, width: int, height: int)
        """
        try:
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                meets_requirements = width >= self.min_width and height >= self.min_height
                return meets_requirements, width, height
            else:
                bt.logging.warning(f"Failed to fetch image for size check: {url}")
                return False, 0, 0
        except Exception as e:
            bt.logging.warning(f"Error checking image size for {url}: {str(e)}")
            return False, 0, 0
    
    def _get_file_extension(self, url):
        """Get appropriate file extension based on content type"""
        extension = '.jpg'  # Default
        try:
            response = requests.head(url, timeout=10)
            content_type = response.headers.get('content-type', '')
            
            if 'image/jpeg' in content_type:
                extension = '.jpg'
            elif 'image/png' in content_type:
                extension = '.png'
            elif 'image/gif' in content_type:
                extension = '.gif'
            elif 'image/webp' in content_type:
                extension = '.webp'
        except Exception as e:
            bt.logging.info(f"Error fetching headers for {url}: {str(e)}")
        
        return extension
    
    def _download_single_image(self, url, file_path, width=None, height=None):
        """Download a single image to the specified path"""
        try:
            image_response = requests.get(url, timeout=10)
            if image_response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(image_response.content)
                size_info = f" ({width}x{height})" if width and height else ""
                bt.logging.trace(f"Downloaded {os.path.basename(file_path)}{size_info}")
                return True
            else:
                bt.logging.error(f"Failed to download {url}, status code: {image_response.status_code}")
                return False
        except Exception as e:
            bt.logging.error(f"Error downloading image from {url}: {str(e)}")
            return False

    @abstractmethod
    def get_image_urls(self, queries, limit=5):
        """
        Abstract method to get image URLs - must be implemented by subclasses
        
        Parameters:
        -----------
        queries : str or list
            Search query or list of queries
        limit : int
            Maximum number of images per query
            
        Returns:
        --------
        dict
            Dictionary with query keys and lists of image data
        """
        pass
    
    def download_images(self, queries=None, urls=None, source_image_paths=None, output_dir="downloads", limit=5):
        """
        Download images based on queries with size constraints
        
        Parameters:
        -----------
        queries : str or list
            Search query or list of queries (set if not using source_images or urls)
        urls: str or list
            Pre-fetched image urls (set if not using source_images or queries)
        source_image_paths: str or list
            Local path to image(s) with which to perform reverse image search
        output_dir : str
            Directory to save images
        limit : int
            Maximum number of images to download per query
        pre_fetched_urls : dict, optional
            Pre-fetched image URLs to use instead of searching
            
        Returns:
        --------
        dict
            Dictionary with query keys and lists of downloaded image info
        """
        if sum(x is not None for x in [queries, urls, source_image_paths]) != 1:
            raise ValueError("Either queries, urls, or source_image must be provided (mutually exclusive)")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if urls is None:
            # Get more URLs than needed to account for size filtering
            image_urls = self.get_image_urls(
                queries=queries,
                source_image_paths=source_image_paths,
                limit=limit * 3
            )
        
        results = {}
        
        for query_key in image_urls:
            downloaded_count = 0
            downloaded_images = []
            
            for img_data in image_urls[query_key]:
                if downloaded_count >= limit:
                    break
                
                url = img_data['url']
                
                meets_size_req, width, height = self._check_image_size(url)
                
                if not meets_size_req:
                    bt.logging.trace(f"Skipping image {url} - size {width}x{height} below minimum {self.min_width}x{self.min_height}")
                    continue
                
                extension = self._get_file_extension(url)
                file_name = f"{time.time()}_{width}x{height}{extension}"
                file_path = os.path.join(output_dir, file_name)
                
                if self._download_single_image(url, file_path, width, height):
                    downloaded_count += 1
                    metadata = {
                        'url': url,
                        'file_path': file_path,
                        'width': width,
                        'height': height,
                        'query': img_data.get('query', query_key),
                        'source_url': img_data.get('source_url'),
                        'title': img_data.get('title', ''),
                        'source': img_data.get('source', 'unknown')
                    }
                    downloaded_images.append(metadata)
                    with open(file_path.replace(extension, '.json'), 'w') as fout:
                        json.dump(metadata, fout)
            
            results[query_key] = downloaded_images
            bt.logging.debug(f"Downloaded {downloaded_count} images meeting size requirements to {output_dir}")
        
        return results