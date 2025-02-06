import argparse
from src.crawlers.UFRJ_crawler import UFRJCrawler
from src.crawlers.UNESP_crawler import UNESPCrawler
from src.crawlers.UNICAMP_crawler import UNICAMPCrawler
from src.crawlers.USP_crawler import USPCrawler
from src.crawlers.UFRB_crawler import UFRBCrawler

def run_all_pages(crawler_name):
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

    scraper.scrape_all_pages()
    print(f"Data scraped and saved successfully for {crawler_name}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the crawler by name.")
    parser.add_argument('crawler_name', type=str, choices=['UFRJ', 'UNESP', 'UNICAMP', 'USP', 'UFRB'],
                        help="The name of the crawler to run.")
    args = parser.parse_args()
    
    run_all_pages(args.crawler_name)
