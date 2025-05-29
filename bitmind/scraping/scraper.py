import os

import bittensor as bt

from bitmind.types import CacheConfig, Modality, MediaType


class MultiSiteScraper:
    """
    Combines multiple scrapers to get diverse image sources
    """
    
    def __init__(self, scrapers, output_dir):
        """
        Initialize with a list of scraper instances
        
        Parameters:
        -----------
        scrapers : list
            List of scraper instances (GoogleScraper, etc.)
        output_dir: str
            Root directory to write media to
            Validator sets this to self.config.cache_dir (~/.cache/sn34 by default)
            Appropriate subdirectory is determined by the scraper's `media_type` and `modality` attrs
        """
        self.scrapers = scrapers
        self.output_dir = output_dir
    
    def download_images(self, queries, limit_per_scraper=5):
        """
        Download images from all scrapers
        
        Parameters:
        -----------
        queries : str or list
            Search query or list of queries
        limit_per_scraper : int
            Maximum number of images per scraper per query
            
        Returns:
        --------
        dict
            Combined results from all scrapers
        """
        all_results = {}
        
        for scraper in self.scrapers:
            scraper_name = scraper.__class__.__name__
            cache_dir = CacheConfig(
                base_dir=self.output_dir, modality=scraper.modality, media_type=scraper.media_type
            ).get_path()
            scraper_dir = cache_dir / scraper_name
                
            bt.logging.debug(f"Starting {scraper_name} scraper...")
            try:
                results = scraper.download_images(queries, scraper_dir, limit_per_scraper)
                all_results[scraper_name] = results
                bt.logging.debug(f"Completed {scraper_name} scraper")
            except Exception as e:
                bt.logging.error(f"Error with {scraper_name}: {str(e)}")
                all_results[scraper_name] = {}

        return all_results