import asyncio
import json
import traceback
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import random

# Add the project root to the Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from microsoft_cve_rag.application.services.scraping_service import MicrosoftKbScraper
except Exception as e:
    logger.error(f"Error importing MicrosoftKbScraper: {e}")
    logger.error(traceback.format_exc())
    # Try direct import as fallback
    try:
        from application.services.scraping_service import MicrosoftKbScraper
        logger.info("Successfully imported MicrosoftKbScraper using direct import path")
    except Exception as e2:
        logger.error(f"Error with direct import: {e2}")
        logger.error(traceback.format_exc())
        raise


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame with KB article data."""
    sample_data = [
        {
            "article_url": "https://support.microsoft.com/en-us/topic/kb5034123-os-builds-22621-3007-and-22631-3007-3f7e169f-56e8-4e6e-b6b8-41f4aa4b9b88",
            "kb_id": "5034123"
        },
        {
            "article_url": "https://support.microsoft.com/en-us/topic/kb5034441-windows-11-update-history-b7f648b7-31a3-4800-9913-7f7284a89b97",
            "kb_id": "5034441"
        },
        {
            "article_url": "https://support.microsoft.com/en-us/topic/kb5034765-windows-10-update-history-9c85d8a3-aa65-4f44-a6c1-dc0893dca17f",
            "kb_id": "5034765"
        }
    ]
    return pd.DataFrame(sample_data)


async def test_single_kb_article():
    """Test the KB scraper with a single article."""
    url = "https://support.microsoft.com/en-us/topic/kb5034123-os-builds-22621-3007-and-22631-3007-3f7e169f-56e8-4e6e-b6b8-41f4aa4b9b88"
    logger.info(f"Testing single KB article: {url}")

    # Create a single scraper instance with the optimized settings
    scraper = MicrosoftKbScraper()

    try:
        # Add a small random delay before making the request
        delay = random.uniform(2, 5)
        logger.info(f"Adding random delay of {delay:.2f} seconds before request")
        await asyncio.sleep(delay)

        # Use the scrape_kb_article method for a single URL
        await scraper.scrape_kb_article(url)
        
        # Check if we got valid HTML content
        html_content = scraper.get_html()
        if html_content and len(html_content) > 0:
            logger.info("Successfully scraped article")
            
            # Extract the markdown content
            markdown_content = scraper.get_markdown()
            extracted_content = scraper.get_extracted_content()
            
            # Save the HTML result to a JSON file for inspection
            output_path = Path(__file__).parent / "single_kb_article.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "url": url,
                    "html_length": len(html_content),
                    "status_code": scraper.get_status_code(),
                    "extracted_content": extracted_content
                }, f, indent=2)
            logger.info(f"Saved JSON result to {output_path}")

            # Save markdown content if available
            if markdown_content:
                markdown_path = Path(__file__).parent / "single_kb_article.md"
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                logger.info(f"Saved markdown to {markdown_path}")

            return True
        else:
            logger.error("Failed to scrape article - no HTML content retrieved")
            return False
    except Exception as e:
        logger.exception(f"Error in test_single_kb_article: {str(e)}")
        return False


async def test_bulk_extract_markdown():
    """Test the bulk_extract_markdown method with multiple KB articles."""
    logger.info("Testing bulk_extract_markdown with multiple KB articles")

    # Create a DataFrame with sample KB article data
    df = create_sample_dataframe()
    logger.info(f"Created sample DataFrame with {len(df)} KB articles")

    # Create a single scraper instance for all URLs
    scraper = MicrosoftKbScraper()

    try:
        # Extract all URLs from the DataFrame
        urls = df['article_url'].tolist()

        # Add a small random delay before making the request
        delay = random.uniform(2, 5)
        logger.info(f"Adding random delay of {delay:.2f} seconds before bulk request")
        await asyncio.sleep(delay)

        # Process all URLs in a batch with bulk_extract_markdown
        logger.info(f"Processing {len(urls)} URLs with bulk_extract_markdown")
        results = await scraper.bulk_extract_markdown(urls)

        # Check the results
        success_count = sum(1 for content in results.values() if content is not None)
        logger.info(f"Successfully scraped {success_count} out of {len(urls)} KB articles")

        # Map results back to the DataFrame
        df['markdown'] = df['article_url'].map(results)

        # Save the results to a CSV file for inspection
        output_path = Path(__file__).parent / "kb_articles_markdown.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Also save individual markdown files for successful scrapes
        for url, markdown in results.items():
            if markdown:
                kb_id = url.split('/')[-1].split('-')[0]
                markdown_path = Path(__file__).parent / f"kb_{kb_id}.md"
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                logger.info(f"Saved markdown for {kb_id} to {markdown_path}")

        return success_count > 0
    except Exception as e:
        logger.exception(f"Error in test_bulk_extract_markdown: {str(e)}")
        return False


async def main():
    """Run the KB scraper tests."""
    logger.info("Starting KB scraper tests")

    # First test with a single KB article
    single_success = await test_single_kb_article()
    logger.info(f"Single KB article test {'succeeded' if single_success else 'failed'}")

    # If single article test succeeds, try bulk extraction
    if single_success:
        logger.info("Single article test succeeded, proceeding with bulk extraction test")
        bulk_success = await test_bulk_extract_markdown()
        logger.info(f"Bulk extraction test {'succeeded' if bulk_success else 'failed'}")
    else:
        logger.warning("Single article test failed, skipping bulk extraction test")


if __name__ == "__main__":
    try:
        # Use Windows-compatible event loop policy
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # Create a new event loop and run the main function
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error running main: {e}")
        logger.error(traceback.format_exc())
