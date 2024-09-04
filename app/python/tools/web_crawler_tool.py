from crawlee import PuppeteerCrawler

def crawl_website(url: str) -> str:
    async def crawler_handler(page, request):
        title = await page.title()
        return f"Crawled {request.url} with title: {title}"

    crawler = PuppeteerCrawler()
    crawler.run([url], crawler_handler)
    return f"Crawled {url}"

class WebCrawlerTool:
    def execute(self, url: str) -> str:
        return crawl_website(url)
