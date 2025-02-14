import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for web scrapers.

    Configures and initializes the AsyncWebCrawler.
    """

    def __init__(self):
        """Initializes the base scraper with crawler configuration."""
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            verbose=True
        )
        self.run_config = CrawlerRunConfig(
            word_count_threshold=0,
            exclude_external_links=True,
            wait_until="domcontentloaded",
            css_selector=".ocpArticleMainContent",
            excluded_tags=["nav", "footer"]
        )
        try:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {str(e)}")
            self.crawler = None

    async def close(self):
        """Closes the crawler session."""
        if self.crawler:
            await self.crawler.close()


class MicrosoftKbScraper(BaseScraper):
    """Scraper for Microsoft Knowledge Base articles.

    Inherits from BaseScraper and implements methods specific to scraping KB articles
    using crawl4ai's page object methods.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initializes the KB article scraper.

        Args:
            output_dir (Optional[str]): Directory to save scraped content.
        """
        super().__init__()
        self.output_dir = output_dir or os.path.join(
            os.getcwd(),
            'microsoft_cve_rag',
            'application',
            'data',
            'scrapes',
            'kb_articles'
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_filename(self, url: str) -> str:
        """Create a sanitized filename from a URL.

        Args:
            url (str): URL to create filename from.

        Returns:
            str: Sanitized filename.
        """
        base_name = url.split('/')[-1]
        base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def _extract_text_content(self, element: str) -> str:
        """Extract text content from an HTML element string.

        Args:
            element (str): HTML element string to extract text from.

        Returns:
            str: Extracted text content.
        """
        try:
            # Remove HTML tags and decode entities
            text = re.sub(r'<[^>]+>', '', element)
            text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
            text = text.strip()
            logger.debug(f"Extracted text content: {text[:100]}...")  # Show first 100 chars
            return text
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return ""

    def _extract_section_by_header(self, html: str, header_text: str) -> str:
        """Extract content of a section based on its header text.

        Args:
            html (str): Full HTML content
            header_text (str): Text of the header to find

        Returns:
            str: Content of the section
        """
        try:
            # Find the section with the matching header
            pattern = f"<h[1-6][^>]*>{header_text}</h[1-6]>(.*?)(?=<h[1-6]|$)"
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return ""
        except Exception as e:
            logger.error(f"Error extracting section {header_text}: {str(e)}")
            return ""

    def _extract_list_items(self, html: str, section_header: str = None) -> List[str]:
        """Extract text content from list items in HTML.

        Args:
            html (str): HTML content to extract from
            section_header (str, optional): Header text to limit extraction to a specific section

        Returns:
            List[str]: List of extracted text.
        """
        items = []
        try:
            if section_header:
                html = self._extract_section_by_header(html, section_header)

            # Find all list items
            pattern = r"<li[^>]*>(.*?)</li>"
            matches = re.finditer(pattern, html, re.DOTALL)

            for match in matches:
                text = self._extract_text_content(match.group(1))
                if text.strip():
                    items.append(text.strip())
                    logger.debug(f"Found list item: {text[:100]}...")

            return items
        except Exception as e:
            logger.error(f"Error extracting list items: {str(e)}")
            return items

    def _extract_table_data(self, html: str, table_header: str = None) -> List[Dict[str, str]]:
        """Extract data from a table in HTML.

        Args:
            html (str): HTML content containing the table
            table_header (str, optional): Header text to find the specific table

        Returns:
            List[Dict[str, str]]: List of extracted table data.
        """
        table_data = []
        try:
            if table_header:
                # Find the specific table section
                html = self._extract_section_by_header(html, table_header)

            # Find the table rows
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = re.finditer(row_pattern, html, re.DOTALL)

            header_mapping = None
            for row in rows:
                row_html = row.group(1)

                # Check if this is a header row
                if "<th" in row_html:
                    # Extract header texts to use as keys
                    headers = re.finditer(r"<th[^>]*>(.*?)</th>", row_html, re.DOTALL)
                    header_mapping = [self._extract_text_content(h.group(1)).lower() for h in headers]
                    continue

                # Extract cell data
                cells = re.finditer(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
                cell_data = [self._extract_text_content(c.group(1)) for c in cells]

                if header_mapping and len(cell_data) >= len(header_mapping):
                    row_dict = {header_mapping[i]: cell_data[i] for i in range(len(header_mapping))}
                    if any(row_dict.values()):
                        table_data.append(row_dict)
                        logger.debug(f"Added table row: {str(row_dict)[:200]}...")

            return table_data
        except Exception as e:
            logger.error(f"Error extracting table data: {str(e)}")
            return table_data

    def _extract_content(self, html: str) -> Dict[str, Any]:
        """Extract content from the KB article HTML.

        Args:
            html (str): Raw HTML content from crawl4ai.

        Returns:
            Dict[str, Any]: Extracted content sections.
        """
        content = {
            "applies_to": {
                "label": "Applies To",
                "value": [],
                "default": []
            },
            "os_builds": {
                "label": "Version / OS Builds",
                "value": "N/A",
                "default": "N/A"
            },
            "highlights": {
                "label": "Highlights",
                "value": [],
                "default": []
            },
            "improvements": {
                "label": "Improvements",
                "value": [],
                "default": []
            },
            "servicing_stack": {
                "label": "Servicing Stack",
                "value": "N/A",
                "default": "N/A"
            },
            "known_issues": {
                "label": "Known Issues",
                "value": [],
                "default": []
            },
            "how_to_get_update": {
                "label": "How to Get This Update",
                "pre_install_instructions": "N/A",
                "release_channels": [],
                "default": {
                    "pre_install_instructions": "N/A",
                    "release_channels": []
                }
            }
        }

        try:
            # Extract Applies To
            applies_to_pattern = r"<details><summary>Applies To</summary>(.*?)</details>"
            applies_to_match = re.search(applies_to_pattern, html, re.DOTALL)
            if applies_to_match:
                spans = re.finditer(r"<span[^>]*>(.*?)</span>", applies_to_match.group(1))
                content["applies_to"]["value"] = [
                    self._extract_text_content(span.group(1))
                    for span in spans
                ]

            # Extract OS Builds
            version_pattern = r"<div><p>Version:</p></div>\s*<div>\s*<p>\s*<b>(.*?)</b>\s*</p>\s*</div>"
            version_match = re.search(version_pattern, html, re.DOTALL)
            if version_match:
                content["os_builds"]["value"] = self._extract_text_content(version_match.group(1))

            # Extract Highlights
            highlights_section = self._extract_section_by_header(html, "Highlights")
            if highlights_section:
                content["highlights"]["value"] = self._extract_list_items(highlights_section)

            # Extract Improvements
            improvements_section = self._extract_section_by_header(html, "Improvements")
            if improvements_section:
                # Find each version section
                version_pattern = r"<h3><button>\s*<div>(Windows (?:10|11)[^<]*)</div></button></h3>\s*<div>(.*?)</div>\s*</div>"
                version_sections = re.finditer(version_pattern, improvements_section, re.DOTALL)

                for section in version_sections:
                    version = section.group(1)
                    section_content = section.group(2)

                    # Extract important notes if present
                    important_notes = []
                    important_pattern = r"<p><b>Important:\s*</b>(.*?)</p>"
                    important_matches = re.finditer(important_pattern, section_content, re.DOTALL)
                    for match in important_matches:
                        note = self._extract_text_content(match.group(1))
                        if note:
                            important_notes.append(note)

                    # Extract list items
                    improvements_list = self._extract_list_items(section_content)

                    # Extract additional paragraphs (excluding Important notes)
                    paragraphs = []
                    p_pattern = r"<p>(?!<b>Important:)(.*?)</p>"
                    p_matches = re.finditer(p_pattern, section_content, re.DOTALL)
                    for match in p_matches:
                        p_text = self._extract_text_content(match.group(1))
                        if p_text:
                            paragraphs.append(p_text)

                    # Combine all content in a structured way
                    details = {
                        "important_notes": important_notes,
                        "improvements": improvements_list,
                        "additional_info": paragraphs
                    }

                    content["improvements"]["value"].append({
                        "version": version,
                        "details": details
                    })

            # Extract Servicing Stack
            servicing_pattern = r"<h3>Windows 11 servicing stack update[^<]*</h3>\s*<p>(.*?)</p>"
            servicing_match = re.search(servicing_pattern, html, re.DOTALL)
            if servicing_match:
                content["servicing_stack"]["value"] = self._extract_text_content(servicing_match.group(1))

            # Extract Known Issues
            issues_section = self._extract_section_by_header(html, "Known issues in this update")
            if issues_section:
                content["known_issues"]["value"] = self._extract_table_data(issues_section)

            # Extract How to Get Update
            update_section = self._extract_section_by_header(html, "How to get this update")
            if update_section:
                # Extract pre-install instructions
                pre_install = re.search(r"<b>Before installing this update</b></p>\s*<p>(.*?)</p>", update_section, re.DOTALL)
                if pre_install:
                    content["how_to_get_update"]["pre_install_instructions"] = self._extract_text_content(pre_install.group(1))

                # Extract release channels
                content["how_to_get_update"]["release_channels"] = self._extract_table_data(update_section)

            return content

        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return content

    async def scrape_kb_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a Microsoft KB article.

        Args:
            url (str): URL of the KB article to scrape.

        Returns:
            Optional[Dict[str, Any]]: Extracted content if successful, None otherwise.
        """
        if not self.crawler:
            logger.error("Crawler not initialized")
            return None

        try:
            async with self.crawler as crawler:
                logger.info(f"Starting crawl of {url}")
                result = await crawler.arun(
                    url=url,
                    config=self.run_config
                )
                logger.info(f"result.cleaned_html: {result.cleaned_html}")
                logger.info(f"Crawl result: {result}")
                if not result or not result.cleaned_html:
                    logger.error(f"Failed to get content from {url}")
                    return None

                # Save raw HTML (optional)
                filename = self._sanitize_filename(url)
                output_path = os.path.join(self.output_dir, f"{filename}_raw.html")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.cleaned_html)

                # Extract content using crawl4ai page methods (no BeautifulSoup)
                content = self._extract_content(result.cleaned_html)

                return content

        except Exception as e:
            logger.error(f"Error scraping KB article {url}: {str(e)}")
            return None


# Example until integration in the main application:
async def main():
    try:
        logger.info("Starting KB article scraper")
        scraper = MicrosoftKbScraper()
        test_url = "https://support.microsoft.com/en-us/topic/january-9-2024-kb5034123-os-builds-22621-3007-and-22631-3007-3f7e169f-56e8-4e6e-b6b8-41f4aa4b9b88"

        logger.info(f"Scraping article: {test_url}")
        content = await scraper.scrape_kb_article(test_url)

        if content:
            logger.info("Successfully scraped article")
            print(json.dumps(content, indent=4))
        else:
            logger.error("Failed to scrape article")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
