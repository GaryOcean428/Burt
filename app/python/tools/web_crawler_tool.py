import requests
from bs4 import BeautifulSoup


def crawl_website(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        return f"Crawled {url} with title: {title}"
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"


class WebCrawlerTool:
    def __init__(self, agent):
        self.agent = agent
        self.name = "web_crawler_tool"

    def execute(self, url: str) -> str:
        return crawl_website(url)
