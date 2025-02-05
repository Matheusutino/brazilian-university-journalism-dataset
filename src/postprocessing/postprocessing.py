import re
import langid

class PostProcessing:
    """Classe para realizar pós-processamento nos dados."""
    
    def __init__(self, data):
        """
        Inicializa o pós-processamento com os dados fornecidos.
        :param data: Lista de dicionários, onde cada dicionário contém 'title' e 'text'.
        """
        self.data = data

        self.months = {
            'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04', 'maio': '05', 'junho': '06',
            'julho': '07', 'agosto': '08', 'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12',
            'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04', 'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
            'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
        }
    
    def add_sequential_id(self):
        """Adiciona um ID sequencial para cada entrada no dado."""
        for idx, entry in enumerate(self.data, start=1):
            entry['ID'] = idx - 1  # Atribui um ID sequencial começando de 1

    def remove_empty_title_text_authors(self):
        self.data = [
            entry for entry in self.data
            if entry.get('Title') and entry.get('Text') and entry['Title'].strip() and entry['Text'].strip() and entry['Authors'] != []
        ]

    def standardize_date(self, date_str):
        """Padroniza a data para o formato 'yyyy-mm-dd HH:MM'."""
        # Tenta identificar e padronizar a data
        date_str = date_str.lower().strip()

        # Formato 1: "13 de dezembro de 2024"
        match1 = re.match(r"(\d{1,2}) de (\w+) de (\d{4})", date_str)
        if match1:
            day, month_name, year = match1.groups()
            month = self.months.get(month_name)
            if month:
                return f"{year}-{month}-{day.zfill(2)} 00:00"

        # Formato 2: "04/10/21 08:00"
        match2 = re.match(r"(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2})", date_str)
        if match2:
            day, month, year, hour, minute = match2.groups()
            return f"20{year}-{month}-{day} {hour}:{minute}"

        # Formato 3: "05/04/2021, 18h55"
        match3 = re.match(r"(\d{2})/(\d{2})/(\d{4}), (\d{2})h(\d{2})", date_str)
        if match3:
            day, month, year, hour, minute = match3.groups()
            return f"{year}-{month}-{day} {hour}:{minute}"

        # Formato 4: "25 nov 2024"
        match4 = re.match(r"(\d{2}) (\w+) (\d{4})", date_str)
        if match4:
            day, month_name, year = match4.groups()
            month = self.months.get(month_name)
            if month:
                return f"{year}-{month}-{day.zfill(2)} 00:00"

        # Formato 5: "11/12/2024 às 13:00"
        match5 = re.match(r"(\d{2})/(\d{2})/(\d{4}) às (\d{2}):(\d{2})", date_str)
        if match5:
            day, month, year, hour, minute = match5.groups()
            return f"{year}-{month}-{day} {hour}:{minute}"

        # Formato 6: "11/12/2024 às 8:54" (trata o caso com o caractere \u00e0s)
        match6 = re.match(r"(\d{2})/(\d{2})/(\d{4}) \u00e0s (\d{1,2}):(\d{2})", date_str)
        if match6:
            day, month, year, hour, minute = match6.groups()
            return f"{year}-{month}-{day} {hour.zfill(2)}:{minute}"

        # Caso não reconheça o formato, retorna None
        return None
    
    def standardize_dates(self):
            """Padroniza as datas de 'Publication Date' e 'Updated Date'."""
            for entry in self.data:
                for date_field in ['Publication Date', 'Updated Date']:
                    if date_field in entry and entry[date_field]:
                        standardized_date = self.standardize_date(entry[date_field])
                        entry[date_field] = standardized_date if standardized_date else None

    def standardize_news_category(self):
        """Converte todas as categorias de notícias para minúsculas."""
        for entry in self.data:
            if 'News Category' in entry and entry['News Category']:
                # Verifica se 'News Category' é uma lista
                if isinstance(entry['News Category'], list):
                    # Converte cada item da lista para minúsculas
                    entry['News Category'] = [category.lower() for category in entry['News Category']]
                else:
                    # Caso contrário, converte o único item para minúsculas
                    entry['News Category'] = entry['News Category'].lower()


    def remove_duplicate_news_urls(self):
        """Remove instâncias com 'News URL' duplicadas."""
        seen_urls = set()
        unique_data = []
        for entry in self.data:
            url = entry.get('News URL')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_data.append(entry)
        self.data = unique_data

    def filter_non_portuguese_news(self):
        """Remove notícias que não estão em português."""
        self.data = [entry for entry in self.data if langid.classify(entry.get('Text', ''))[0] == 'pt']

    def clean_data(self):
        """Remove instâncias onde 'title' ou 'text' são nulos ou contêm apenas espaços."""
        # Filtra os dados para manter apenas os registros válidos
        self.remove_empty_title_text_authors()
        self.remove_duplicate_news_urls()
        self.filter_non_portuguese_news()
        self.standardize_dates()
        self.standardize_news_category()
        self.add_sequential_id()

        return self.data