<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" width="120"/>

# I want to use the crawl4ai python package to scrape some complex web pages. I thought I could use it directly but gpt says I need to use beautiful soup to streamline the process. what is the latest syntax to implement crawl4ai in both sync and async contexts?

---
It appears there may be some confusion regarding the use of Crawl4AI and BeautifulSoup. Crawl4AI is a powerful web crawling and scraping package that doesn't require BeautifulSoup for its core functionality. Let's clarify the latest syntax for implementing Crawl4AI in both synchronous and asynchronous contexts.

## Synchronous Usage

For synchronous usage, you can use the `WebCrawler` class:

```python
from crawl4ai import WebCrawler

# Create an instance of WebCrawler
crawler = WebCrawler()

# Warm up the crawler (load necessary models)
crawler.warmup()

# Run the crawler on a URL
result = crawler.run(url="https://example.com")

# Print the extracted content in Markdown format
print(result.markdown)
```


## Asynchronous Usage

For asynchronous usage, which is recommended for better performance, especially when crawling multiple pages, you can use the `AsyncWebCrawler` class:

```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def main():
    # Configure the browser
    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True
    )

    # Configure the crawler run
    run_cfg = CrawlerRunConfig(
        css_selector="main.article",
        word_count_threshold=10,
        screenshot=True
    )

    # Use the crawler in an async context manager
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun("https://example.com", config=run_cfg)
        
        print("Crawled HTML length:", len(result.cleaned_html))
        if result.screenshot:
            print("Screenshot base64 length:", len(result.screenshot))

# Run the async function
asyncio.run(main())
```


## Advanced Usage

For more complex scraping tasks, Crawl4AI offers advanced features:

1. **Content Filtering**: You can use strategies like `PruningContentFilter` or `BM25ContentFilter` to refine the extracted content[^4].
2. **Extraction Strategies**: For structured data extraction, you can use `JsonCssExtractionStrategy` or `LLMExtractionStrategy`[^1][^2].
3. **Batch Processing**: Use `arun_many()` for processing multiple URLs with intelligent rate limiting and resource monitoring[^1].

Here's an example of a more advanced setup:

```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json

async def main():
    browser_cfg = BrowserConfig(
        browser_type="firefox",
        headless=False,
        verbose=True
    )

    schema = {
        "name": "Articles",
        "baseSelector": "article.post",
        "fields": [
            {
                "name": "title",
                "selector": "h2",
                "type": "text"
            },
            {
                "name": "url",
                "selector": "a",
                "type": "attribute",
                "attribute": "href"
            }
        ]
    }

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(schema),
        word_count_threshold=15,
        remove_overlay_elements=True,
        wait_for="css:.post"
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://example.com/blog",
            config=run_cfg
        )

        if result.success:
            print("Cleaned HTML length:", len(result.cleaned_html))
            if result.extracted_content:
                articles = json.loads(result.extracted_content)
                print("Extracted articles:", articles[:2])
        else:
            print("Error:", result.error_message)

asyncio.run(main())
```

This setup allows for complex scraping tasks without the need for BeautifulSoup. Crawl4AI provides built-in capabilities for handling JavaScript-rendered content, content filtering, and structured data extraction[^1][^2][^4].

Remember to always respect websites' terms of service and robots.txt files when scraping. Also, consider implementing proper rate limiting and error handling in your crawling scripts to be a good web citizen.

<div style="text-align: center">⁂</div>

[^1]: https://docs.crawl4ai.com/api/async-webcrawler/

[^2]: https://www.revanthquicklearn.com/post/exploring-crawl4ai-enhancing-ai-agents-with-advanced-web-crawling-and-scraping-capabilities

[^3]: https://crawlee.dev/python/docs/examples/beautifulsoup-crawler

[^4]: https://docs.crawl4ai.com/core/markdown-generation/

[^5]: https://github.com/duponfiona8/Scraping-Data/blob/main/blog/Crawl4AI: An Asynchronous Web Scraping Tool Optimized for LLMs.md

[^6]: https://docs.crawl4ai.com/core/quickstart/

[^7]: https://realpython.com/beautiful-soup-web-scraper-python/

[^8]: https://docs.crawl4ai.com/core/content-selection/

[^9]: https://www.pondhouse-data.com/blog/webcrawling-with-crawl4ai

[^10]: https://github.com/unclecode/crawl4ai

[^11]: https://news.ycombinator.com/item?id=39442273

[^12]: https://github.com/unclecode/crawl4ai/blob/main/docs/examples/hello_world.py

[^13]: https://docs.crawl4ai.com

[^14]: https://docs.phidata.com/tools/crawl4ai

[^15]: https://www.zenrows.com/blog/asynchronous-web-scraping-python

[^16]: https://www.reddit.com/r/DataHoarder/comments/1iknxwj/crawl4ai_code_example/

[^17]: https://gitee.com/myhfw003/crawl4ai

[^18]: https://www.linkedin.com/posts/bishnudevkhutia_machinelearning-ai-fastapi-activity-7213454198422614017-UD8x

[^19]: https://www.youtube.com/watch?v=DcI_AZqfZVc

[^20]: https://python.langchain.com/v0.1/docs/integrations/document_transformers/beautiful_soup/

[^21]: https://stackoverflow.com/questions/72795498/crawling-news-articles-with-beautifulsoup

[^22]: https://www.linkedin.com/posts/mohamed-ashour-0727_build-your-own-webscrapper-using-crawl4ai-activity-7254498327038226436-Y_2Z

[^23]: https://stackoverflow.com/questions/21570780/using-python-and-beautifulsoup-saved-webpage-source-codes-into-a-local-file

[^24]: https://www.linkedin.com/posts/usmancout_crawl4ai-beautifulsoup-webscraping-activity-7282366564731940865-VGSY

[^25]: https://jkropko.github.io/surfing-the-data-pipeline/ch5.html

[^26]: https://crawlbase.com/blog/how-to-use-beautifulsoup-in-python/

[^27]: https://crawlbase.com/blog/scrape-dynamic-content-with-selenium-and-beautifulsoup/

[^28]: https://www.youtube.com/watch?v=GjKQ6V_ViQE

