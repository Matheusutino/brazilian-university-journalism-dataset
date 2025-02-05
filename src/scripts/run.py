from src.crawlers.UFRJ_crawler import UFRJCrawler
from src.crawlers.UNESP_crawler import UNESPCrawler
from src.crawlers.UNICAMP_crawler import UNICAMPCrawler
from src.crawlers.USP_crawler import USPCrawler
from src.crawlers.UFRB_crawler import UFRBCrawler

if __name__ == "__main__":
    #scraper = UFRJCrawler(delay=1)
    #scraper = UNESPCrawler(delay=1)
    #scraper = UNICAMPCrawler(delay=1)
    scraper = USPCrawler(delay=1)
    #scraper = UFRBCrawler(delay=1)
    
    # try:
        # Comece a scraping a partir do progresso salvo
    #print(scraper.extract_news_details("https://jornal.usp.br/articulistas/luiz-roberto-serrano/como-a-globo-exorcizou-a-ameaca-de-se-tornar-um-dying-business-e-mantem-a-lideranca-de-audiencia-na-tv-aberta/")['Authors'])
    #print("\n\n")
    #print(scraper.extract_news_details("https://ufrb.edu.br/portal/noticias/539-professores-da-ufrb-participaram-de-oficina-tematica-do-sisal-na-secti")['Text'])
    scraper.scrape_all_pages()
    #scraper.scrape_page(10)

        
    print("All gazeta data scraped and saved successfully!")

    # except Exception as e:
    #     # Se ocorrer um erro, mostre uma mensagem
    #     print(f"An error occurred: {e}")
    #     print("Progress saved. You can restart the scraper to resume.")

    
