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
    Google Images scraper using Selenium
    """
    
    def __init__(
        self,
        user_agent=None,
        scroll_delay=500,
        headless=True,
        tbs=None,
        safe=False,
        max_year=None,
        min_width=128,
        min_height=128,
        use_google_size_filter=True
    ):
        super().__init__(min_width, min_height)
        
        if user_agent is None:
            user_agent = [
                'Mozilla/5.0 (X11; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            ]
            
        self.user_agent = random.choice(user_agent) if isinstance(user_agent, list) else user_agent
        self.scroll_delay = scroll_delay
        self.headless = headless
        self.use_google_size_filter = use_google_size_filter
        
        if tbs is None:
            tbs = {}
            
        if max_year is not None:
            current_year = time.localtime().tm_year
            if max_year < current_year:
                tbs["cdr"] = "1"  # enable custom date range
                tbs["cd_max"] = f"{max_year}/12/31"
        
        if use_google_size_filter and min_width >= 128 and min_height >= 128:
            if min_width >= 400 and min_height >= 300:
                tbs["isz"] = "l"  # Large images (400x300+)
            elif min_width >= 200 and min_height >= 200:
                tbs["isz"] = "m"  # Medium images (200x200+)
            else:
                tbs["isz"] = "ex"  # Exact size
                tbs["iszw"] = str(min_width)
                tbs["iszh"] = str(min_height)
                
        self.tbs = self._parse_request_parameters(tbs)
        self.safe = self._make_safe_query(safe)
    
    def get_image_urls(self, queries, limit=5):
        """Get image URLs from Google Images"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={self.user_agent}")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            browser = webdriver.Chrome(options=chrome_options)
            image_url_object = {}
            
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
                
                images = browser.find_elements(By.TAG_NAME, "img")
                image_urls = []
                
                for img in images:
                    src = img.get_attribute("src")
                    if src and src.startswith("http") and "google" not in src:
                        image_urls.append(src)
                
                query_key = query.replace(" ", "")
                image_url_object[query_key] = [
                    {"query": query, "url": url, "source": "google"} 
                    for url in image_urls[:limit]
                ]
            
            browser.quit()
            return image_url_object
            
        except Exception as e:
            bt.logging.error(f"Google scraper error: {str(e)}")
            if 'browser' in locals() and browser:
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
