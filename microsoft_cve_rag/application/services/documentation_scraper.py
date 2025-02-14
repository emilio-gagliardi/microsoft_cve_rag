import os
import re
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import html2text

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocMetadata:
    """Metadata for scraped documentation."""
    package_name: str
    version: Optional[str] = None
    last_updated: str = datetime.now().isoformat()
    platform: Optional[str] = None
    base_url: str = ""

class PackageDocScraper:
    """Generic documentation scraper for Python packages."""

    PLATFORMS = {
        "readthedocs.io": "ReadTheDocs",
        "readthedocs.org": "ReadTheDocs",
        "github.io": "GitHub Pages",
        "docs.python.org": "Python Docs",
    }

    def __init__(
        self,
        package_name: str,
        docs_url: str,
        output_dir: str = None,
        version: Optional[str] = None
    ):
        """Initialize the documentation scraper.
        
        Args:
            package_name: Name of the Python package
            docs_url: Base URL of the documentation
            output_dir: Directory to store documentation
            version: Package version to scrape (if available)
        """
        self.package_name = package_name
        self.base_url = docs_url.rstrip('/')
        self.version = version
        
        # Set up output directory using absolute path
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent  # Go up to microsoft_cve_rag
            output_dir = base_dir / "architect" / "docs"
        else:
            output_dir = Path(output_dir)
            
        self.output_dir = output_dir / package_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Documentation will be saved to: {self.output_dir}")
        
        # HTML to text converter for fallback
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0
        self.html_converter.protect_links = True  # Don't wrap links in <>
        self.html_converter.wrap_links = False    # Don't wrap links
        self.html_converter.unicode_snob = True   # Use Unicode
        self.html_converter.skip_internal_links = True  # Skip internal page anchors
        self.html_converter.inline_links = True   # Use inline links
        
        # Initialize crawler with optimized settings for docs
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            viewport_width=1280,
            viewport_height=900,
            verbose=True
        )
        
        try:
            self.crawler = AsyncWebCrawler(config=browser_config)
            self.run_config = CrawlerRunConfig(
                word_count_threshold=10,
                exclude_external_links=True,
                wait_until="networkidle"
            )
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {str(e)}")
            self.crawler = None
            self.run_config = None
        
        # Detect platform
        domain = self.base_url.split('/')[2]
        self.platform = next(
            (p for d, p in self.PLATFORMS.items() if d in domain),
            "Unknown"
        )
        
        # Initialize metadata
        self.metadata = DocMetadata(
            package_name=package_name,
            version=version,
            platform=self.platform,
            base_url=self.base_url
        )

    async def _scrape_page(self, url: str, retries: int = 2) -> Optional[str]:
        """Scrape a single documentation page with retries.
        
        Args:
            url: URL to scrape
            retries: Number of retry attempts
            
        Returns:
            Optional[str]: Markdown content if successful, None otherwise
        """
        if not self.crawler:
            logger.error("Crawler not initialized")
            return None

        for attempt in range(retries):
            try:
                logger.info(f"Attempt {attempt + 1} for {url}")
                async with self.crawler as crawler:
                    result = await crawler.arun(
                        url=url,
                        config=self.run_config
                    )
                    if result and result.markdown:
                        return result.markdown
                    elif result and result.cleaned_html:
                        # Try using cleaned HTML if markdown fails
                        return self.html_converter.handle(result.cleaned_html)
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1} for {url}: {str(e)}")
                if attempt < retries - 1:
                    continue
        return None

    def _sanitize_filename(self, url: str) -> str:
        """Generate filesystem-safe filename from URL."""
        path = url.replace(self.base_url, '').strip('/')
        if not path:
            return "index.md"
        return re.sub(r'[^a-zA-Z0-9-]', '_', path) + '.md'

    def _save_metadata(self):
        """Save metadata about the documentation."""
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=2)

    async def scrape_docs(self, urls: List[str]) -> Dict[str, bool]:
        """Scrape documentation pages and save as markdown.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dict[str, bool]: Map of URLs to success status
        """
        results = {}
        
        # Validate and normalize URLs
        normalized_urls = []
        for url in urls:
            if not url.startswith(self.base_url):
                url = f"{self.base_url}/{url.lstrip('/')}"
            normalized_urls.append(url)

        # Process URLs
        for url in normalized_urls:
            try:
                logger.info(f"Processing {url}")
                content = await self._scrape_page(url)
                
                if content:
                    filename = self._sanitize_filename(url)
                    output_path = self.output_dir / filename
                    output_path.write_text(content, encoding='utf-8')
                    results[url] = True
                    logger.info(f"Successfully saved {filename}")
                else:
                    results[url] = False
            except Exception as e:
                results[url] = False
                logger.error(f"Failed to process {url}: {str(e)}")
            
        self._save_metadata()
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"\nScraping Summary:")
        logger.info(f"Total URLs: {len(results)}")
        logger.info(f"Successfully scraped: {successful}")
        logger.info(f"Failed: {len(results) - successful}")
        
        return results

async def main():
    # Test with just a few URLs first
    test_urls = [
        "https://docs.crawl4ai.com/core/installation/",
        "https://docs.crawl4ai.com/core/quickstart/"
    ]
    
    scraper = PackageDocScraper(
        package_name="crawl4ai",
        docs_url="https://docs.crawl4ai.com",
        version="0.4.3b2"
    )
    
    # Start with a test run
    logger.info("Starting test run with limited URLs")
    results = await scraper.scrape_docs(test_urls)
    
    if any(results.values()):
        logger.info("Test successful, proceeding with full scrape")
        # Full URL list
        all_urls = [
            "https://docs.crawl4ai.com/core/installation/",
            "https://docs.crawl4ai.com/core/quickstart/",
            "https://docs.crawl4ai.com/core/simple-crawling/",
            "https://docs.crawl4ai.com/core/browser-crawler-config/",
            "https://docs.crawl4ai.com/core/markdown-generation/",
            "https://docs.crawl4ai.com/core/fit-markdown/",
            "https://docs.crawl4ai.com/core/page-interaction/",
            "https://docs.crawl4ai.com/core/content-selection/",
            "https://docs.crawl4ai.com/core/local-files/",
            "https://docs.crawl4ai.com/advanced/crawl-dispatcher/",
            "https://docs.crawl4ai.com/advanced/session-management/",
            "https://docs.crawl4ai.com/extraction/no-llm-strategies/",
            "https://docs.crawl4ai.com/extraction/llm-strategies/",
            "https://docs.crawl4ai.com/api/async-webcrawler/",
            "https://docs.crawl4ai.com/api/arun/",
            "https://docs.crawl4ai.com/api/crawl-result/",
            "https://docs.crawl4ai.com/api/strategies/"
        ]
        await scraper.scrape_docs(all_urls)
    else:
        logger.error("Test failed, please check configuration")

if __name__ == "__main__":
    asyncio.run(main())
