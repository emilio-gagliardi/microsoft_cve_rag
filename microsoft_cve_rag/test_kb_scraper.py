import asyncio
import json
import traceback
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from microsoft_cve_rag.application.services.scraping_service import MicrosoftKbScraper
except Exception as e:
    logger.error("Error importing MicrosoftKbScraper:")
    logger.error(traceback.format_exc())
    raise

async def test_kb_scraper():
    """Test the KB scraper with a sample article."""
    # Use a real, existing KB article
    test_url = "https://support.microsoft.com/en-us/topic/january-9-2024-kb5034123-os-builds-22621-3007-and-22631-3007-3f7e169f-56e8-4e6e-b6b8-41f4aa4b9b88"
    logger.info(f"Scraping KB article: {test_url}")

    scraper = MicrosoftKbScraper()
    try:
        result = await scraper.scrape_kb_article(test_url)
    except Exception as e:
        logger.error("Error scraping KB article:")
        logger.error(traceback.format_exc())
        return

    if result:
        logger.info("\nSuccessfully extracted structured content:")
        logger.info("=" * 50)
        logger.info(json.dumps(result, indent=2))

        # Print output directory info
        output_dir = scraper.output_dir
        logger.info(f"\nOutput files saved in: {output_dir}")

        # List generated files
        logger.info("\nGenerated files:")
        for file in Path(output_dir).glob("*"):
            logger.info(f"- {file.name}")
    else:
        logger.error("Failed to scrape the article")

if __name__ == "__main__":
    try:
        asyncio.run(test_kb_scraper())
    except Exception as e:
        logger.error("Error running test_kb_scraper:")
        logger.error(traceback.format_exc())
