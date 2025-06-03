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

    def download_images(
        self, queries=None, source_image_paths=None, limit_per_scraper=5
    ):
        """
        Download images from all scrapers

        Parameters:
        -----------
        queries : str or list
            Search query or list of queries  (mutually exclusive with `source_image_paths`)
        source_image_paths: str or list
            Image path or list of image paths with which to perform reveres image search
            (mutually exclusive with `queries`)
        limit_per_scraper : int
            Maximum number of images per scraper per query

        Returns:
        --------
        dict
            Combined results from all scrapers
        """
        if sum(x is not None for x in [queries, source_image_paths]) != 1:
            raise ValueError(
                "Either queries, urls, or source_image must be provided (mutually exclusive)"
            )

        all_results = {}

        for scraper in self.scrapers:
            scraper_name = scraper.__class__.__name__
            cache_dir = CacheConfig(
                base_dir=self.output_dir,
                modality=scraper.modality,
                media_type=scraper.media_type,
            ).get_path()
            scraper_dir = cache_dir / scraper_name

            bt.logging.debug(f"Starting {scraper_name} scraper...")
            try:
                results = scraper.download_images(
                    queries=queries,
                    source_image_paths=source_image_paths,
                    output_dir=scraper_dir,
                    limit=limit_per_scraper,
                )
                all_results[scraper_name] = results
                bt.logging.debug(f"Completed {scraper_name} scraper")
            except Exception as e:
                bt.logging.error(f"Error with {scraper_name}: {str(e)}")
                all_results[scraper_name] = {}

        return all_results
