import asyncio
import json
import logging
import os
import re
import ast
from datetime import datetime
import sys
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse
# Configure Windows event loop for subprocess support
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler, CacheMode, MarkdownGenerationResult, CrawlResult
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
)
from pydantic import BaseModel, Field

# import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

ImprovementsValue = Union[List[str], str]


class KBArticle(BaseModel):
    title: str
    url: str
    applies_to: List[str] = Field(
        default_factory=list,
        description="List of product families the KB article applies to."
    )
    os_builds: str = Field(
        default_factory=str,
        description="OS build(s) the KB article applies to, eg., '17763.5936'"
    )
    page_introduction: str = Field(
        default_factory=str,
        description="Text that appears before the first section header."
    )
    highlights: List[str] = Field(
        default_factory=list,
        description="List of highlights, typically bullet points."
    )
    improvements: Dict[str, ImprovementsValue] = Field(
        default_factory=dict,
        description=(
            "Dictionary of improvements, keyed by subheading or topic. "
            "Each value can be a list of bullet points or a single string of free text."
        )
    )
    servicing_stack_update: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of servicing stack updates, keyed by product family."
    )
    known_issues_and_workaround: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of known issues and workarounds, each stored as a dictionary."
    )
    # Flattened fields from HowToGetUpdate
    how_to_get_update_before_installation: Optional[str] = Field(
        default=None,
        description="Text that appears under 'Before you install this update' (if any)."
    )
    how_to_get_update_prerequisites: Optional[str] = Field(
        default=None,
        description="Text listing any required SSUs, LCUs, or other prerequisites."
    )
    how_to_get_update_install_instructions: Optional[str] = Field(
        default=None,
        description="General instructions for installing the update."
    )
    how_to_get_update_channels: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "A list of dictionaries, each describing a channel from the update table. "
            "Each dictionary MUST contain exactly three keys:\n\n"
            "  1) channel_name: Must be one of [Windows Update, Business, Catalog, Server Update Services, Microsoft Download Center]\n"
            "  2) availability: Typically 'Yes' or 'No', or a short string describing availability\n"
            "  3) next_step: A short explanation or link on how to get the update\n\n"
            "For example:\n\n"
            "  [\n"
            "    {\n"
            "      \"channel_name\": \"Windows Update\",\n"
            "      \"availability\": \"Yes\",\n"
            "      \"next_step\": \"Install automatically via Windows Update\"\n"
            "    },\n"
            "    {\n"
            "      \"channel_name\": \"Catalog/Update Catalog\",\n"
            "      \"availability\": \"Yes\",\n"
            "      \"next_step\": \"Download manually from the Microsoft Update Catalog\"\n"
            "    }\n"
            "  ]\n\n"
            "Do not use any alternative keys (like 'Available' or 'Next Step'); "
            "stick to 'channel_name', 'availability', and 'next_step' exactly."
        )
    )
    how_to_get_update_remove_lcu_instructions: Optional[str] = Field(
        default=None,
        description="Steps for removing the LCU if needed, typically referencing DISM or wusa.exe."
    )
    # Flattened fields from FileInformationBlock and FileInformationTable
    file_information: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description=(
            "List of file information blocks. Each block is a dictionary with 'text' (optional string) "
            "and 'tables' (optional list of dictionaries, each with 'product_family' and 'rows')."
        )
    )


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
        self.crawl_result: Optional[CrawlResult] = None
        self.crawl_results: Optional[List[CrawlResult]] = None
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
                word_count_threshold=5,
                exclude_external_links=True,
                wait_until="networkidle",
                extraction_strategy=None,
                display_mode="DETAILED"
            )
            logger.info("No CrawlerRunConfig provided, using default configuration.")
        self.run_config = run_config
        logger.info(f"CrawlerRunConfig: {self.run_config}")

        try:
            self.crawler = AsyncWebCrawler(
                config=self.browser_config,
                run_config=self.run_config
            )
            logger.info("AsyncWebCrawler initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {str(e)}")
            self.crawler = None

    def get_url(self) -> str:
        """Returns the URL from the crawl result."""
        return getattr(self.crawl_result, "url", "")

    def get_html(self, html_type: str = "cleaned") -> str:
        """Returns the raw or cleaned HTML from the KB article crawl result.

        Args:
            html_type: Type of HTML to return. Must be "cleaned" or "raw".
                Defaults to "cleaned" for processed content.

        Returns:
            str: The requested HTML content or empty string if not available.

        Notes:
            Used by the memory-adaptive dispatcher for bulk KB article processing
            with proper rate limiting between requests.
        """
        if html_type == "cleaned":
            return getattr(self.crawl_result, "cleaned_html", "")
        if html_type == "raw":
            return getattr(self.crawl_result, "html", "")
        logger.warning(f"Invalid HTML type '{html_type}'. Must be 'cleaned' or 'raw'.")
        return ""

    def get_success(self) -> bool:
        """Returns whether the crawl was successful."""
        return getattr(self.crawl_result, "success", False)

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

    def get_markdown(self) -> MarkdownGenerationResult:
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

    def get_safe_filename(self, filename: str) -> str:
        """Generate a safe filename from URLs and timestamp.

        Creates a filename-safe string using the most recent URL and current
        timestamp. If no URLs exist, returns a generic timestamp-based name.

        Returns:
            str: A filename-safe string in format:
                'domain_path_YYYYMMDD_HHMMSS' or
                'scrape_YYYYMMDD_HHMMSS' if no URLs.
        """
        if not filename:
            logger.warning("No filename provided, returning 'no_url_provided'.")
            return "no_url_provided"

        # If it looks like a URL, extract the meaningful portion
        if '://' in filename:
            try:
                # Get everything after the last slash, before any query params
                filename = filename.rstrip('/').split('/')[-1].split('?')[0]
            except Exception:
                pass  # Fall back to normal filename cleaning

        # Replace problematic characters
        filename = filename.lower()
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'[-\s]+', '-', filename).strip('-')

        return filename or "unnamed"

    async def save_crawl_result(
        self,
        crawl_result: Optional[CrawlResult] = None,
        output_dir: Optional[str] = None,
        raw_html: bool = True,
        cleaned_html: bool = True,
        markdown: bool = True,
        extracted_content: bool = True,
        pdf: bool = False,
        screenshot: bool = False,
        filename_prefix: Optional[str] = None
    ) -> None:
        """Save KB article crawl result to disk in multiple formats.

        Uses memory-adaptive dispatcher for bulk processing and implements proper
        rate limiting between requests.

        Args:
            crawl_result: Optional CrawlResult instance to save. If None, uses the
                first result from self.crawl_results or self.crawl_result.
            output_dir: Optional directory to save results in. If not provided,
                uses self.output_dir.
            raw_html: Save original unmodified HTML.
            cleaned_html: Save sanitized HTML with scripts/styles removed.
            markdown: Save markdown version of content.
            extracted_content: Save structured data extracted from page.
            pdf: Save PDF version if available.
            screenshot: Save page screenshot if available.
            filename_prefix: Optional prefix to add to filenames.

        Notes:
            - Files are saved with format: {prefix}{domain}_{path}_{timestamp}.{ext}
            - Timestamp format: YYYYMMDD_HHMMSS
            - URL components are sanitized for safe filenames
            - Uses memory-adaptive dispatcher for bulk operations
            - Implements rate limiting between requests
        """
        result = crawl_result or self.crawl_result
        if not result and self.crawl_results:
            result = self.crawl_results[0]

        if not result:
            logger.warning("No crawl result to save")
            return

        # Get output directory
        output_dir = output_dir or self.output_dir
        if not output_dir:
            logger.error("No output directory specified")
            return

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Generate base filename
        base_filename = self.get_safe_filename(result.url)
        if filename_prefix:
            base_filename = f"{filename_prefix}{base_filename}"

        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{base_filename}_{timestamp}"

        # Save raw HTML if available
        if raw_html and hasattr(result, 'html'):
            await self.async_save_text_content(output_dir, f"{base_filename}_raw.html", result.html)

        # Save cleaned HTML if available
        if cleaned_html and hasattr(result, 'cleaned_html') and result.cleaned_html:
            await self.async_save_text_content(output_dir, f"{base_filename}_cleaned.html", result.cleaned_html)

        # Handle markdown content with version check
        if markdown and hasattr(result, 'markdown'):
            if isinstance(result.markdown, str):
                await self.async_save_text_content(output_dir, f"{base_filename}.md", result.markdown)
            else:
                logger.warning("Unexpected markdown type: %s", type(result.markdown))

            # if hasattr(result, 'fit_markdown'):
            #     await self.async_save_text_content(output_dir, f"{base_filename}_fit_markdown.md", result.fit_markdown)
            # if hasattr(result, 'markdown_with_citations'):
            #     await self.async_save_text_content(output_dir, f"{base_filename}_with_citations_markdown.md", result.markdown_with_citations)
            # if hasattr(result, 'references_markdown'):
            #     await self.async_save_text_content(output_dir, f"{base_filename}_references_markdown.md", result.references_markdown)

        # Save extracted content
        if extracted_content and hasattr(result, 'extracted_content'):
            content = await self.async_parse_json_content(result.extracted_content)
            if content:
                await self.async_save_json_content(output_dir, f"{base_filename}_extracted.json", content)
            else:
                logger.warning("Failed to parse extracted content")
        # Save PDF if available
        if pdf and hasattr(result, 'pdf') and result.pdf:
            await self.async_save_binary_content(output_dir, f"{base_filename}.pdf", result.pdf)

        # Save screenshot if available
        if screenshot and hasattr(result, 'screenshot') and result.screenshot:
            await self.async_save_binary_content(output_dir, f"{base_filename}.png", result.screenshot)

        logger.info(f"Successfully saved KB article content to {output_dir}")

    def _validate_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Validate and sanitize the session ID.

        Args:
            session_id: The session ID to validate

        Returns:
            Optional[str]: The sanitized session ID

        Raises:
            ValueError: If session_id is provided but invalid
        """
        if session_id is None:
            return None

        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")

        # Remove any whitespace and special characters
        sanitized = "".join(
            c for c in session_id.strip() if c.isalnum() or c in "_-"
        )

        if not sanitized:
            raise ValueError(
                "session_id must contain valid characters (alphanumeric,"
                " underscore, or hyphen)"
            )

        return sanitized

    def _save_text_content(
        self, directory: str, filename: str, content: str
    ) -> None:
        """Save text content to file.

        Args:
            directory (str): The directory to save the file in.
            filename (str): The name of the file to save.
            content (str): The text content to save.
        """
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    async def async_save_text_content(self, directory: str, filename: str, content: str) -> None:
        await asyncio.to_thread(self._save_text_content, directory, filename, content)

    async def async_save_binary_content(self, directory: str, filename: str, content: bytes) -> None:
        await asyncio.to_thread(self._save_binary_content, directory, filename, content)

    async def async_save_json_content(self, directory: str, filename: str, content: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._save_json_content, directory, filename, content)

    def _save_binary_content(
        self, directory: str, filename: str, content: bytes
    ) -> None:
        """Save binary content to file.

        Args:
            directory (str): The directory to save the file in.
            filename (str): The name of the file to save.
            content (bytes): The binary content to save.
        """
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            f.write(content)

    def _save_json_content(
        self, directory: str, filename: str, content: Dict[str, Any]
    ) -> None:
        """Save JSON content to file.

        Args:
            directory (str): The directory to save the file in.
            filename (str): The name of the file to save.
            content (Dict[str, Any]): The JSON content to save.
        """
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    async def async_parse_json_content(self, content: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self._parse_json_content, content)

    def _parse_json_content(self, content: Any) -> Dict[str, Any]:
        """Parse JSON content, handling string inputs.

        Args:
            content (Any): The content to parse.

        Returns:
            Dict[str, Any]: The parsed JSON content. If content is a list of dicts,
            returns the last dict as it represents the LLM's final attempt.
        """
        if isinstance(content, str):
            try:
                # First try ast.literal_eval for Python string representations
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    valid_dicts = [item for item in parsed if isinstance(item, dict)]
                    if valid_dicts:
                        if len(valid_dicts) > 1:
                            logging.info(f"Found {len(valid_dicts)} LLM attempts, using final attempt")
                        return valid_dicts[-1]
                    logging.warning("List contained no valid JSON objects")
                return parsed if isinstance(parsed, dict) else {"raw": content}
            except (ValueError, SyntaxError):
                try:
                    # Try JSON parsing as fallback
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        valid_dicts = [item for item in parsed if isinstance(item, dict)]
                        if valid_dicts:
                            if len(valid_dicts) > 1:
                                logging.info(f"Found {len(valid_dicts)} LLM attempts in JSON, using final attempt")
                            return valid_dicts[-1]
                        logging.warning("JSON list contained no valid objects")
                    return parsed if isinstance(parsed, dict) else {"raw": content}
                except json.JSONDecodeError:
                    logging.warning("Content could not be parsed as Python literal or JSON")
                    return {"raw": content}
        return content if isinstance(content, dict) else {"raw": str(content)}

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
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Sec-CH-UA": '"Chromium";v="122", "Not(A:Brand";v="24"',
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": '"Windows"'
            },
            verbose=True
        )
        logger.info(f"Custom BrowserConfig for KB: {kb_browser_config}")

        # Factory-style decision for extraction strategy.
        extraction_strategy: Optional[Any] = None
        # Define a default Pydantic model for KB articles if no schema is provided.
        json_schema = KBArticle.model_json_schema()

        if extraction_method.lower() == "llm":
            logger.info("Using LLM extraction strategy.")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OpenRouter API key not found in environment variables for LLM extraction.")

            # Build the instruction prompt using the JSON schema.
            prompt = (
                "Extract a structured JSON object from the following Markdown that represents a Microsoft KB report. "
                "The JSON object must conform to the following schema:\n\n"
                f"{json.dumps(json_schema, indent=4)}\n\n"
                "Ensure that each field is extracted correctly from the Markdown. Return only the JSON object."
            )
            extraction_strategy = LLMExtractionStrategy(
                provider="openrouter/google/gemini-2.5-pro-exp-03-25:free",
                api_token=openrouter_api_key,
                schema=json_schema,
                extraction_type="schema",
                instruction=prompt,
                chunk_token_threshold=5000,
                overlap_rate=0.02,
                apply_chunking=True,
                input_format="markdown",
                verbose=True
            )

        elif extraction_method.lower() == "css":
            logger.info("Using JsonCss extraction strategy.")
            css_schema = {
                "name": "KBArticleContent",
                "baseSelector": "main#supArticleContent",
                "fields": [
                    {"name": "body_content", "selector": ":scope > *", "type": "html"}
                ]
            }
            extraction_strategy = JsonCssExtractionStrategy(css_schema)
        else:
            logger.error(f"Unknown extraction_method: {extraction_method}. Defaulting to LLM extraction.")
            raise ValueError(f"Unknown extraction_method: {extraction_method}")
        # Create custom CrawlerRunConfig for KB articles.
        # teachingCalloutHidden.teachingCalloutPopover, popoverMessageWrapper, col-1-5, f-multi-column.f-multi-column-6, c-uhfh-actions, c-uhfh-gcontainer-st
        # "nav", "footer"
        kb_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.DISABLED,
            extraction_strategy=extraction_strategy,
            word_count_threshold=4,
            exclude_external_links=False,
            wait_until="networkidle",
            css_selector="main#supArticleContent",
            excluded_tags=["nav", "footer"],
            excluded_selector=(".col-1-5, .supLeftNavMobileView, .supLeftNavMobileViewContent.grd, "
                               ".teachingCalloutHidden.teachingCalloutPopover, .popoverMessageWrapper, "
                               ".f-multi-column.f-multi-column-6, .c-uhfh-actions, .c-uhfh-gcontainer-st, "
                               ".ocArticleFooterElementContainer, .col-1-5.ucsRailContainer, "
                               ".ocArticleFooterShareLinksWrapper"
                               ),
            verbose=True,
        )
        logger.info(f"Custom CrawlerRunConfig for KB: {kb_run_config}")
        # Call the BaseScraper constructor with chosen extraction strategy and configurations.
        super().__init__(
            browser_config=kb_browser_config,
            run_config=kb_run_config
        )

        self.output_dir: str = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "application",
            "data",
            "scrapes",
            "kb_articles"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")

    @staticmethod
    def _is_kb_article_url(url: str) -> bool:
        """Check if the URL is a valid HTML URL.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is a valid HTML URL
        """
        if not isinstance(url, str):
            return False

        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except Exception:
            return False

    def _validate_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Validate and sanitize the session ID.

        Args:
            session_id: The session ID to validate

        Returns:
            Optional[str]: The sanitized session ID

        Raises:
            ValueError: If session_id is provided but invalid
        """
        if session_id is None:
            return None

        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")

        # Remove any whitespace and special characters
        sanitized = "".join(
            c for c in session_id.strip() if c.isalnum() or c in "_-"
        )

        if not sanitized:
            raise ValueError(
                "session_id must contain valid characters (alphanumeric,"
                " underscore, or hyphen)"
            )

        return sanitized

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
            return None

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
                extracted_content = self.get_extracted_content()
                markdown = self.get_markdown()
                logger.info(f"Raw HTML length: {len(raw_html)} characters")
                logger.info(f"LLM Extraction length: {len(extracted_content)}")
                logger.info(f"Markdown length: {len(markdown)} characters")

                # Log browser and run configurations
                browser_config = self.get_browser_config()
                run_config = self.get_run_config()
                logger.info(f"Browser Config: {browser_config}")
                logger.info(f"Run Config: {run_config}")

                return result

        except Exception as e:
            logger.error(f"Error scraping KB article {url}: {str(e)}")
            return None

    async def save_kb_bulk_results(
        self,
        results: List[CrawlResult],
        output_dir: Optional[str] = None
    ) -> Optional[str]:
        """Save bulk crawl results to disk using base class infrastructure.

        Args:
            results: List of crawl results to save
            output_dir: Optional directory to save results

        Returns:
            str: Path to output directory if successful, None otherwise

        Notes:
            - Uses base class save_crawl_result for consistent file operations
            - Adds KB-specific validation using KBArticle Pydantic model
            - Tracks success/failure for monitoring
        """
        if not results:
            logger.warning("No results to save")
            return None

        if output_dir:
            self.output_dir = output_dir

        successful_saves = 0
        failed_saves = 0
        save_errors = []

        for result in results:
            try:
                # Validate extracted content if present
                if hasattr(result, 'extracted_content'):
                    content = self._parse_json_content(result.extracted_content)
                    if isinstance(content, list):
                        valid_items = []
                        for item in content:
                            if isinstance(item, dict) and item.get('url') is not None:
                                valid_items.append(item)
                        result.extracted_content = valid_items if valid_items else None
                    elif isinstance(content, dict):
                        if content.get('url') is None:
                            # Likely blocked or invalid content
                            result.extracted_content = None

                # Use base class save method
                await self.save_crawl_result(
                    crawl_result=result,
                    output_dir=self.output_dir,
                    raw_html=False,
                    cleaned_html=False,
                    filename_prefix="kb_article_"
                )
                successful_saves += 1

            except Exception as e:
                failed_saves += 1
                save_errors.append((getattr(result, "url", "unknown"), str(e)))
                logger.exception(f"Error saving result: {str(e)}")

        # Log summary
        logger.info(f"Successfully saved {successful_saves} results")
        if failed_saves:
            logger.error(f"Failed to save {failed_saves} results")
            for url, error in save_errors:
                logger.error(f"Save failed for {url}: {error}")

        return self.output_dir if successful_saves > 0 else None


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
