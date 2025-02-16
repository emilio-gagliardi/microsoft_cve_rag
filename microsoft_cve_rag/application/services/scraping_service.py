import asyncio
import json
import logging
import os
# import re
from datetime import datetime
# from pickle import False
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse
# from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
)
from pydantic import BaseModel
# import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for web scrapers.

    Configures and initializes the AsyncWebCrawler and core extraction
    strategy. This class centralizes default configurations for crawling
    and provides a unified method to process crawl results. Real-world
    suggestion: Tune the default parameters (viewport size, wait conditions,
    etc.) based on the target website's behavior, and consider applying
    preprocessing to filter out ads or non-relevant sections.
    """

    def __init__(
        self,
        browser_config: Optional[BrowserConfig] = None,
        run_config: Optional[CrawlerRunConfig] = None
    ) -> None:
        """Initializes the base scraper with crawler configuration, an optional
        extraction strategy, and configurable browser and crawler settings.

        Args:
            extraction_strategy (Optional[Any]): An object implementing an 'extract'
                method to convert raw HTML to structured data.
            browser_config (Optional[BrowserConfig]): Configuration for the browser.
            run_config (Optional[CrawlerRunConfig]): Configuration for crawling
                operations.
        """
        logger.info("Initializing BaseScraper.")
        self.crawl_result: Optional[Any] = None
        self.output_dir: Optional[str] = None
        self.urls: List[str] = []
        if browser_config is None:
            browser_config = BrowserConfig(
                browser_type="chromium",      # Recommended: use Chromium for reliable rendering.
                headless=True,               # Enable headless mode for performance.
                viewport_width=1920,         # Standard desktop width.
                viewport_height=1080,        # Standard desktop height.
                verbose=True                 # Detailed logging enabled.
            )
            logger.info("No BrowserConfig provided, using default configuration.")
        self.browser_config = browser_config
        logger.info(f"BrowserConfig: {self.browser_config}")

        if run_config is None:
            run_config = CrawlerRunConfig(
                word_count_threshold=0,
                exclude_external_links=True,
                wait_until="networkidle",
                extraction_strategy=None,
                display_mode="DETAILED"
            )
            logger.info("No CrawlerRunConfig provided, using default configuration.")
        self.run_config = run_config
        logger.info(f"CrawlerRunConfig: {self.run_config}")

        try:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
            logger.info("AsyncWebCrawler initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {str(e)}")
            self.crawler = None

    def get_url(self) -> str:
        """Returns the URL from the crawl result."""
        return getattr(self.crawl_result, "url", "")

    def get_html(self) -> str:
        """Returns the raw HTML from the crawl result."""
        return getattr(self.crawl_result, "html", "")

    def get_success(self) -> bool:
        """Returns whether the crawl was successful."""
        return getattr(self.crawl_result, "success", False)

    def get_cleaned_html(self) -> str:
        """Returns the cleaned HTML (after preprocessing) from the crawl result."""
        return getattr(self.crawl_result, "cleaned_html", "")

    def get_media(self) -> any:
        """Returns any media captured during the crawl (e.g., images, videos)."""
        return getattr(self.crawl_result, "media", None)

    def get_links(self) -> any:
        """Returns any links found during the crawl."""
        return getattr(self.crawl_result, "links", None)

    def get_downloaded_files(self) -> any:
        """Returns any files downloaded during the crawl."""
        return getattr(self.crawl_result, "downloaded_files", None)

    def get_screenshot(self) -> any:
        """Returns a screenshot captured during the crawl, if available."""
        return getattr(self.crawl_result, "screenshot", None)

    def get_pdf(self) -> any:
        """Returns a PDF generated during the crawl, if available."""
        return getattr(self.crawl_result, "pdf", None)

    def get_markdown(self) -> str:
        """Returns the markdown representation of the page."""
        return getattr(self.crawl_result, "markdown", "")

    def get_extracted_content(self) -> Optional[Dict[str, Any]]:
        """Returns structured data extracted from the crawl result.

        Attempts to get the extracted content from the crawl result and parse it
        as JSON if it's a string. The content may be under either 'extracted_data'
        or 'extracted_content'.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON data as a dictionary, or None if
            no data is available or parsing fails.
        """
        content = getattr(self.crawl_result, "extracted_content", None)
        if not content:
            return None

        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted content as JSON: {e}")
                return None

        return content  # If it's already a dictionary, return as is

    def get_metadata(self) -> any:
        """Returns metadata from the crawl result."""
        return getattr(self.crawl_result, "metadata", None)

    def get_error_message(self) -> str:
        """Returns any error message resulting from the crawl."""
        return getattr(self.crawl_result, "error_message", "")

    def get_session_id(self) -> str:
        """Returns the session ID associated with this crawl."""
        return getattr(self.crawl_result, "session_id", "")

    def get_response_headers(self) -> any:
        """Returns the response headers of the crawl."""
        return getattr(self.crawl_result, "response_headers", None)

    def get_status_code(self) -> int:
        """Returns the HTTP status code returned by the crawl."""
        return getattr(self.crawl_result, "status_code", 0)

    def get_browser_config(self) -> BrowserConfig:
        """Returns the BrowserConfig associated with the scraper."""
        return self.browser_config

    def get_run_config(self) -> CrawlerRunConfig:
        """Returns the CrawlerRunConfig associated with the scraper."""
        return self.run_config

    def get_safe_filename(self) -> str:
        """Generate a safe filename from URLs and timestamp.

        Creates a filename-safe string using the most recent URL and current
        timestamp. If no URLs exist, returns a generic timestamp-based name.

        Returns:
            str: A filename-safe string in format:
                'domain_path_YYYYMMDD_HHMMSS' or
                'scrape_YYYYMMDD_HHMMSS' if no URLs.
        """
        if not self.urls:
            return f"scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        url = self.urls[-1]
        parsed = urlparse(url)
        path = parsed.path.rstrip('/').replace('.html', '').replace('.htm', '')

        domain_part = parsed.netloc.replace('.', '_')
        path_part = path.replace('/', '_')
        if len(path_part) > 50:
            path_part = path_part[:50]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{domain_part}{path_part}_{timestamp}"

    def save_crawl_result(
        self,
        output_dir: str,
        html: bool = True,
        markdown: bool = False,
        json_output: bool = False,
        pdf: bool = False,
        thumbnail: bool = False,
        filename_prefix: Optional[str] = None
    ) -> None:
        """
        Saves the crawl result in one or more formats (HTML, Markdown, JSON, PDF, Thumbnail).
        The output_dir folder is created if it doesn't exist.

        Args:
            output_dir (str): Path to the directory where files should be saved.
            html (bool): Save raw HTML to 'page.html' if True. Defaults to True.
            markdown (bool): Save markdown to 'page.md' if True. Defaults to False.
            json_output (bool): Save extracted JSON to 'extracted_data.json' if True. Defaults to False.
            pdf (bool): Save PDF content to 'page.pdf' if True. Defaults to False.
            thumbnail (bool): Save screenshot (PNG) to 'thumbnail.png' if True. Defaults to False.
            filename_prefix: Optional prefix to add to all generated filenames.
                   For example, "kb_article_" would result in
                   "kb_article_domain_path_timestamp.html"
        Notes:
    - Files are saved with format: {prefix}{domain}_{path}_{timestamp}.{ext}
    - Timestamp format: YYYYMMDD_HHMMSS
    - URL components are sanitized for safe filenames
        """
        if not self.crawl_result:
            logger.warning("No crawl_result found, nothing to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        base_filename = self.get_safe_filename()
        if filename_prefix:
            base_filename = f"{filename_prefix}{base_filename}"

        if html:
            html_path = os.path.join(output_dir, f"{base_filename}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.get_cleaned_html())
            logger.info(f"HTML saved to {html_path}")

        if markdown:
            md = self.get_markdown()
            if md:
                md_path = os.path.join(output_dir, f"{base_filename}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
                logger.info(f"Markdown saved to {md_path}")
            else:
                logger.warning("No markdown found in crawl_result.")

        if json_output:
            data = self.get_extracted_content()
            if data:
                json_path = os.path.join(output_dir, f"{base_filename}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"JSON saved to {json_path}")
            else:
                logger.warning("No extracted data found in crawl_result.")

        if pdf:
            pdf_content = self.get_pdf()
            if pdf_content:
                pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(pdf_content)
                logger.info(f"PDF saved to {pdf_path}")
            else:
                logger.warning("No PDF found in crawl_result.")

        if thumbnail:
            screenshot = self.get_screenshot()
            if screenshot:
                thumbnail_path = os.path.join(output_dir, f"{base_filename}.png")
                with open(thumbnail_path, "wb") as f:
                    f.write(screenshot)
                logger.info(f"Thumbnail saved to {thumbnail_path}")
            else:
                logger.warning("No screenshot found in crawl_result.")

        logger.info("save_crawl_result completed.")

    async def close(self) -> None:
        """Closes the crawler session, freeing up resources."""
        logger.info("Closing crawler session.")
        if self.crawler:
            await self.crawler.close()
            logger.info("Crawler session closed.")


class MicrosoftKbScraper(BaseScraper):
    """Scraper for Microsoft Knowledge Base articles.

    Uses crawl4ai to obtain raw HTML, then applies an exclusion-based filtering
    mechanism to remove unwanted sections (e.g., navigation and footer elements).
    Finally, it extracts structured data using one of two extraction strategies:
      - LLMExtractionStrategy: Uses an LLM with a descriptive prompt.
      - JsonCssExtractionStrategy: Uses CSS selectors provided in a JSON schema.
    The caller can choose which extraction method to use.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        extraction_method: str = "llm",
        json_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initializes the KB article scraper with custom Browser and Crawler configs,
        and sets up the extraction strategy according to the specified method using
        a Pydantic-based schema.

        Args:
            output_dir (Optional[str]): Directory to save scraped content.
            extraction_method (str): Extraction method to use. Allowed values are "llm" or "json".
                - "llm" will use LLMExtractionStrategy with a custom instruction prompt.
                - "json" will use JsonCssExtractionStrategy with a defined CSS extraction schema.
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema for the extraction strategy.
               If not provided, a default schema using a Pydantic model for KB articles is used.
        """
        logger.info("Initializing MicrosoftKbScraper.")
        # Create custom BrowserConfig for KB articles.
        kb_browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            verbose=True
        )
        logger.info(f"Custom BrowserConfig for KB: {kb_browser_config}")

        # Factory-style decision for extraction strategy.
        extraction_strategy: Optional[Any] = None
        # Define a default Pydantic model for KB articles if no schema is provided.
        if not json_schema:
            class KBArticle(BaseModel):
                title: str
                url: str
                applies_to: list[str]
                os_builds: str
                page_introduction: str
                highlights: list[str]
                improvements: dict[str, list[str]]
                servicing_stack_update: dict[str, str]
                known_issues_and_workaround: list[dict[str, Any]]
                how_to_get_update: list[dict[str, Any]]
            json_schema = KBArticle.model_json_schema()

        if extraction_method.lower() == "llm":
            logger.info("Using LLM extraction strategy.")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OpenRouter API key not found in environment variables for LLM extraction.")

            # Build the instruction prompt using the JSON schema.
            prompt = (
                "Extract a structured JSON object from the following HTML that represents a Microsoft KB report. "
                "The JSON object must conform to the following schema:\n\n"
                f"{json.dumps(json_schema, indent=4)}\n\n"
                "Ensure that each field is extracted correctly from the HTML. Return only the JSON object."
            )
            extraction_strategy = LLMExtractionStrategy(
                provider="openrouter/google/gemini-2.0-pro-exp-02-05:free",
                api_token=openrouter_api_key,
                schema=json_schema,
                extraction_type="schema",
                instruction=prompt,
                chunk_token_threshold=2000,  # Adjust based on expected HTML size.
                overlap_rate=0.07,
                apply_chunking=True,
                input_format="html",
                verbose=True
            )

        elif extraction_method.lower() == "json":
            logger.info("Using JsonCss extraction strategy.")
            extraction_strategy = JsonCssExtractionStrategy(json_schema)
        else:
            logger.error(f"Unknown extraction_method: {extraction_method}. Defaulting to LLM extraction.")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OpenRouter API key not found in environment variables for LLM extraction.")

            # Build the instruction prompt using the JSON schema.
            prompt = (
                "Extract a structured JSON object from the following HTML that represents a Microsoft KB report. "
                "The JSON object must conform to the following schema:\n\n"
                f"{json.dumps(json_schema, indent=4)}\n\n"
                "Ensure that each field is extracted correctly from the HTML. Return only the JSON object."
            )
            extraction_strategy = LLMExtractionStrategy(
                provider="google/gemini-2.0-flash-exp:free",
                api_token=openrouter_api_key,
                schema=json_schema,
                extraction_type="schema",
                instruction=prompt,
                chunk_token_threshold=2000,
                overlap_rate=0.07,
                apply_chunking=True,
                input_format="html",
                verbose=True
            )
        # Create custom CrawlerRunConfig for KB articles.
        # teachingCalloutHidden.teachingCalloutPopover, popoverMessageWrapper, col-1-5, f-multi-column.f-multi-column-6, c-uhfh-actions, c-uhfh-gcontainer-st
        # "nav", "footer"
        kb_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.DISABLED,
            extraction_strategy=extraction_strategy,
            word_count_threshold=0,
            exclude_external_links=False,
            wait_until="domcontentloaded",
            css_selector=None,
            excluded_tags=["nav", "footer"],
            excluded_selector=".col-1-5, .supLeftNavMobileView, .supLeftNavMobileViewContent.grd, .teachingCalloutHidden.teachingCalloutPopover, .popoverMessageWrapper, .f-multi-column.f-multi-column-6, .c-uhfh-actions, .c-uhfh-gcontainer-st",
            verbose=True,
        )
        logger.info(f"Custom CrawlerRunConfig for KB: {kb_run_config}")
        # Call the BaseScraper constructor with chosen extraction strategy and configurations.
        super().__init__(
            browser_config=kb_browser_config,
            run_config=kb_run_config
        )

        self.output_dir: str = output_dir or os.path.join(
            os.getcwd(),
            "microsoft_cve_rag",
            "application",
            "data",
            "scrapes",
            "kb_articles"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")

    async def scrape_kb_article(self, url: str) -> None:
        """Scrape content from a Microsoft KB article.

        Uses the AsyncWebCrawler to obtain raw HTML, then processes the crawl result
        (including HTML preprocessing and LLM-based extraction) to obtain structured content.

        Args:
            url (str): URL of the KB article to scrape.

        Returns:
            Optional[Dict[str, Any]]: Structured content as a JSON dictionary if successful,
            or None otherwise.
        """
        logger.info(f"Starting crawl for KB article: {url}")
        if not self.crawler:
            logger.error("Crawler not initialized.")

        try:
            async with self.crawler as crawler:
                self.urls.append(url)
                result = await crawler.arun(url=url, config=self.run_config)
                self.crawl_result = result
                logger.info(f"CrawlResult populated with {len(self.get_html())} characters of HTML")
                logger.info("CrawlResult structure:\n%s", json.dumps({
                    'html': bool(self.crawl_result.html),
                    'cleaned_html': bool(self.crawl_result.cleaned_html),
                    'structured_data': bool(self.crawl_result.extracted_content),
                    'screenshot': bool(self.crawl_result.screenshot)
                }, indent=4))
                if not self.get_success():
                    logger.error(f"No valid content retrieved from {url}")
                    if self.get_error_message():
                        logger.error(f"Error message: {self.get_error_message()}")

                logger.info(f"Status code: {self.get_status_code()}")
                logger.info("KB article scraped and processed successfully.")
                # Log content lengths to check for truncation
                raw_html = self.get_html()
                llm_extraction = self.get_extracted_content()
                logger.info(f"Raw HTML length: {len(raw_html)} characters")
                logger.info(f"LLM Extraction length: {len(llm_extraction)}")

                # Log browser and run configurations
                browser_config = self.get_browser_config()
                run_config = self.get_run_config()
                logger.info(f"Browser Config: {browser_config}")
                logger.info(f"Run Config: {run_config}")

        except Exception as e:
            logger.error(f"Error scraping KB article {url}: {str(e)}")


# Example until integration in the main application:
async def main() -> None:
    """Entry-point for testing MicrosoftKbScraper.

    Creates the scraper, scrapes a KB article URL, and prints a detailed report
    of the returned content including both the structured JSON and the pristine markdown.
    """
    kb_url = "https://support.microsoft.com/en-us/topic/january-9-2024-kb5034123-os-builds-22621-3007-and-22631-3007-3f7e169f-56e8-4e6e-b6b8-41f4aa4b9b88"  # Replace with a real URL for testing.

    logging.info("Creating instance of MicrosoftKbScraper using LLM extraction strategy.")
    # You can change extraction_method to "json" if desired.
    kb_scraper = MicrosoftKbScraper(extraction_method="llm")

    logging.info(f"Scraping KB article from URL: {kb_url}")
    await kb_scraper.scrape_kb_article(kb_url)

    if kb_scraper.get_success():
        # Log extraction results
        logging.info("Extraction Results:")

        # Check extracted content
        extracted_content = kb_scraper.get_extracted_content()
        if extracted_content is None:
            logging.error("Extracted content is None!")
            logging.info("Checking raw extraction data...")
            # Try to access any other properties that might contain the extracted data
            logging.info(f"Available CrawlResult attributes: {dir(kb_scraper.crawl_result)}")
        else:
            logging.info(f"Extracted content type: {type(extracted_content)}")
            logging.info(f"Extracted content structure: {json.dumps(extracted_content, indent=2)}")

        # Check markdown content
        markdown_text = kb_scraper.get_markdown()
        if not markdown_text:
            logging.error("Markdown text is empty!")
        else:
            logging.info(f"Markdown length: {len(markdown_text)} characters")
            logging.info("First 500 characters of markdown:")
            logging.info(markdown_text[:500])
            logging.info("Last 500 characters of markdown:")
            logging.info(markdown_text[-500:] if len(markdown_text) > 500 else markdown_text)

        # Show LLM usage statistics
        logging.info("LLM Usage Statistics:")
        kb_scraper.run_config.extraction_strategy.show_usage()

        # Print the final output
        print("\nStructured JSON:")
        print(json.dumps(extracted_content, indent=4) if extracted_content else "No structured data extracted")

        print("\nMarkdown Output:")
        print(markdown_text if markdown_text else "No markdown content extracted")
    else:
        print("Failed to extract content for the KB article.")
        logging.error(f"Crawl failed with status code: {kb_scraper.get_status_code()}")
        logging.error(f"Error message: {kb_scraper.get_error_message()}")

    kb_scraper.save_crawl_result(
        output_dir=kb_scraper.output_dir,
        html=True,
        markdown=True,
        json_output=True,
        pdf=False,
        thumbnail=False,
        filename_prefix="kb_article_"
    )
    logging.info("Closing MicrosoftKbScraper crawler session.")
    await kb_scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
