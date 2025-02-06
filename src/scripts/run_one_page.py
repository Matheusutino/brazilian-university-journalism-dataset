import argparse
from src.crawlers.UFRJ_crawler import UFRJCrawler
from src.crawlers.UNESP_crawler import UNESPCrawler
from src.crawlers.UNICAMP_crawler import UNICAMPCrawler
from src.crawlers.USP_crawler import USPCrawler
from src.crawlers.UFRB_crawler import UFRBCrawler

def main(crawler_name, url):
    if crawler_name == 'UFRJ':
        scraper = UFRJCrawler(delay=1)
    elif crawler_name == 'UNESP':
        scraper = UNESPCrawler(delay=1)
    elif crawler_name == 'UNICAMP':
        scraper = UNICAMPCrawler(delay=1)
    elif crawler_name == 'USP':
        scraper = USPCrawler(delay=1)
    elif crawler_name == 'UFRB':
        scraper = UFRBCrawler(delay=1)
    else:
        print(f"Unknown crawler name: {crawler_name}")
        return

    scraper.extract_news_details(url)
    print(f"News details extracted successfully for {crawler_name} from {url}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the crawler by name and provide a URL.")
    parser.add_argument('crawler_name', type=str, choices=['UFRJ', 'UNESP', 'UNICAMP', 'USP', 'UFRB'],
                        help="The name of the crawler to run.")
    parser.add_argument('url', type=str, help="The URL to extract news details from.")
    
    args = parser.parse_args()
    
    main(args.crawler_name, args.url)
