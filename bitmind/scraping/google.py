import os
import random
import time
import urllib.parse
from urllib.parse import quote_plus

import bittensor as bt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from bitmind.scraping.base import BaseScraper


class GoogleScraper(BaseScraper):
    """
    Google Images scraper using Selenium with reverse image search support
    """

    def __init__(
        self,
        user_agent=None,
        scroll_delay=500,
        headless=True,
        tbs=None,
        max_year=2017,
        min_width=128,
        min_height=128,
        media_type=None,
    ):
        super().__init__(min_width, min_height, media_type)

        self.user_agents = user_agent
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (X11; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            ]

        self.scroll_delay = scroll_delay
        self.headless = headless

        if tbs is None:
            tbs = {}

        if max_year is not None:
            current_year = time.localtime().tm_year
            if max_year < current_year:
                tbs["cdr"] = "1"  # enable custom date range
                tbs["cd_max"] = f"12/31/{max_year}"

        if min_width is not None and min_height is not None:
            if min_width >= 400 and min_height >= 300:
                tbs["isz"] = "l"
            elif min_width >= 128:
                tbs["isz"] = "m"

        self.tbs = self._parse_request_parameters(tbs)

    def get_image_urls(self, queries=None, source_image_paths=None, limit=5):
        """
        Get image URLs from Google Images using either text queries or reverse image search.

        Parameters:
        -----------
        queries : str or list, optional
            Search query or list of queries (mutually exclusive with source_image_paths)
        source_image_paths : str or list, optional
            Path(s) to image(s) to use for reverse image search (mutually exclusive with queries)
        limit : int, default=5
            Maximum number of images to return per query

        Returns:
        --------
        dict
            Dictionary mapping query keys to lists of image data dictionaries

        Raises:
        -------
        ValueError
            If neither or both queries and source_image_paths are provided
        """
        if sum(x is not None for x in [queries, source_image_paths]) != 1:
            raise ValueError(
                "Either queries or source_image must be provided (mutually exclusive)"
            )

        if queries is not None:
            return self.image_search(queries, limit)
        elif source_image_paths is not None:
            raise NotImplementedError

    def image_search(self, queries, limit=5):
        """
        Perform Google image search with text queries.

        Parameters:
        -----------
        queries : str or list
            Search query or list of queries to search for
        limit : int, default=5
            Maximum number of images to return per query

        Returns:
        --------
        dict
            Dictionary mapping query keys to lists of image data dictionaries
            Each image dict contains:
            - query: Original search query
            - url: Image URL
            - source: Always "google"
        """
        user_agent = (
            random.choice(self.user_agents)
            if isinstance(self.user_agents, list)
            else self.user_agents
        )
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={user_agent}")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            browser = webdriver.Chrome(options=chrome_options)
            results = {}

            if not isinstance(queries, list):
                queries = [queries]

            for query in queries:
                page_url = f"https://www.google.com/search?&safe=active&source=lnms&tbs={self.tbs}&tbm=isch&q={self._parse_request_queries(query)}"
                browser.get(page_url)

                # Scroll to load more images
                for _ in range(10):
                    browser.execute_script("window.scrollBy(0, window.innerHeight)")
                    time.sleep(self.scroll_delay / 1000)

                WebDriverWait(browser, 10).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "img"))
                )

                image_elements = (
                    browser.find_elements(By.TAG_NAME, "img")
                    + browser.find_elements(By.CSS_SELECTOR, "img[data-src]")
                    + browser.find_elements(By.CSS_SELECTOR, "img[src^='http']")
                )

                image_urls = []
                for img in image_elements:
                    for attr in ["src", "data-src"]:
                        src = img.get_attribute(attr)
                        if src and src.startswith("http") and "google" not in src:
                            image_urls.append(src)
                            break

                query_key = query.replace(" ", "")
                results[query_key] = [
                    {"query": query, "url": url, "source": "google"}
                    for url in image_urls[:limit]
                ]
            browser.quit()
            return results

        except Exception as e:
            bt.logging.error(f"Google scraper error: {str(e)}")
            if "browser" in locals() and browser:
                browser.quit()
            return {}

    def _parse_request_parameters(self, tbs):
        """Parse TBS parameters for Google search URL"""
        if not tbs:
            return ""
        param_parts = []
        for key, value in tbs.items():
            if value:
                param_parts.append(f"{key}:{value}")
        return urllib.parse.quote(",".join(param_parts))

    def _parse_request_queries(self, query):
        """Encode query for URL"""
        return quote_plus(query) if query else ""
