import requests
from bs4 import BeautifulSoup
from app.python.helpers.tool import Tool, Response


def crawl_website(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        return f"Crawled {url} with title: {title}"
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"


class WebCrawlerTool(Tool):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "web_crawler_tool"

    def execute(self, url: str) -> str:
        return crawl_website(url)
