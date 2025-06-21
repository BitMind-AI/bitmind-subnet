import os
import random
import time
import urllib.parse
from urllib.parse import quote_plus
import json
import re

import bittensor as bt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

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
                tbs["isz"] = "l"  # large
            elif min_width >= 128:
                tbs["isz"] = "m"  # medium

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
        Perform Google image search with text queries and extract full-size image URLs.

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
            - url: Full-size image URL
            - source: Always "google"
            - thumbnail_url: Thumbnail URL (optional)
        """
        user_agent = (
            random.choice(self.user_agents)
            if isinstance(self.user_agents, list)
            else self.user_agents
        )

        browser = None
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={user_agent}")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )
            chrome_options.add_experimental_option("useAutomationExtension", False)

            browser = webdriver.Chrome(options=chrome_options)
            browser.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            results = {}

            if not isinstance(queries, list):
                queries = [queries]

            for query in queries:
                bt.logging.info(f"Searching for: {query}")
                page_url = f"https://www.google.com/search?&safe=active&source=lnms&tbs={self.tbs}&tbm=isch&q={self._parse_request_queries(query)}"
                browser.get(page_url)

                # Wait for page to load
                time.sleep(2)

                # Scroll to load more images
                for i in range(5):  # Reduced scrolling for faster execution
                    browser.execute_script("window.scrollBy(0, window.innerHeight)")
                    time.sleep(self.scroll_delay / 1000)

                # Method 1: Click on images to get full-size URLs
                image_urls = self._extract_full_size_urls_by_clicking(browser, limit)

                # Method 2: If clicking method fails, try extracting from page source
                if len(image_urls) < limit:
                    bt.logging.info("Trying alternative extraction method...")
                    additional_urls = self._extract_urls_from_page_source(
                        browser, limit - len(image_urls)
                    )
                    image_urls.extend(additional_urls)

                query_key = query.replace(" ", "")
                results[query_key] = [
                    {"query": query, "url": url, "source": "google"}
                    for url in image_urls[:limit]
                ]

                bt.logging.info(
                    f"Found {len(results[query_key])} images for query: {query}"
                )

            return results

        except Exception as e:
            bt.logging.error(f"Google scraper error: {str(e)}")
            return {}
        finally:
            if browser:
                browser.quit()

    def _extract_full_size_urls_by_clicking(self, browser, limit):
        """Extract full-size image URLs by clicking on thumbnails"""
        image_urls = []

        try:
            # Find clickable image containers
            image_containers = browser.find_elements(By.CSS_SELECTOR, "[data-ri]")

            for i, container in enumerate(
                image_containers[: limit * 2]
            ):  # Get more than needed
                if len(image_urls) >= limit:
                    break

                try:
                    # Scroll element into view
                    browser.execute_script(
                        "arguments[0].scrollIntoView(true);", container
                    )
                    time.sleep(0.5)

                    # Click on the image
                    ActionChains(browser).move_to_element(container).click().perform()
                    time.sleep(1)

                    # Look for the full-size image in the preview pane
                    full_size_img = None

                    # Try different selectors for the full-size image
                    selectors = [
                        "img[src*='https://'][style*='max-width']",
                        ".irc_mi img",
                        ".v4dQwb img",
                        "img[jsname]",
                    ]

                    for selector in selectors:
                        try:
                            elements = browser.find_elements(By.CSS_SELECTOR, selector)
                            for elem in elements:
                                src = elem.get_attribute("src")
                                if (
                                    src
                                    and src.startswith("http")
                                    and "google" not in src
                                    and "gstatic" not in src
                                    and len(src) > 50
                                ):  # Longer URLs are usually full-size
                                    full_size_img = src
                                    break
                            if full_size_img:
                                break
                        except:
                            continue

                    if full_size_img and full_size_img not in image_urls:
                        image_urls.append(full_size_img)
                        bt.logging.debug(
                            f"Found full-size image: {full_size_img[:100]}..."
                        )

                    # Close the preview if it opened
                    try:
                        close_button = browser.find_element(
                            By.CSS_SELECTOR, "[aria-label='Close']"
                        )
                        close_button.click()
                        time.sleep(0.5)
                    except:
                        pass

                except Exception as e:
                    bt.logging.debug(f"Error clicking image {i}: {str(e)}")
                    continue

        except Exception as e:
            bt.logging.error(f"Error in clicking method: {str(e)}")

        return image_urls

    def _extract_urls_from_page_source(self, browser, limit):
        """Extract image URLs from page source using regex"""
        image_urls = []

        try:
            page_source = browser.page_source

            # Look for image URLs in various patterns
            patterns = [
                r'"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
                r"'(https://[^']*\.(?:jpg|jpeg|png|gif|webp)[^']*)'",
                r'src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
                r'data-src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    if (
                        match not in image_urls
                        and "google" not in match
                        and "gstatic" not in match
                        and len(match) > 50
                    ):  # Filter out small/thumbnail URLs
                        image_urls.append(match)
                        if len(image_urls) >= limit:
                            break
                if len(image_urls) >= limit:
                    break

        except Exception as e:
            bt.logging.error(f"Error extracting from page source: {str(e)}")

        return image_urls[:limit]

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
