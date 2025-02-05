import pandas as pd
import nltk
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from src.utils import save_json, create_folder_if_not_exists

# Função para carregar dados
def load_data(file_path, text_column="Text"):
    data = pd.read_json(file_path)
    return data[text_column].tolist()

# Função de pré-processamento de textos
def preprocess_texts(texts):
    texts_split = []
    stop_words = nltk.corpus.stopwords.words('portuguese')
    for text in texts:
        # Remover pontuação e números, e transformar para minúsculas
        cleaned_text = re.sub(r'\W+', ' ', text.lower())
        # Tokenizar e remover stopwords
        tokens = [word for word in cleaned_text.split() if word not in stop_words]
        texts_split.append(tokens)
    return texts_split

# Função para calcular coerência para K-Means
def calculate_coherence_kmeans(top_words, dictionary, corpus, texts):
    coherence_scores = {}
    metrics = ["c_v", "c_npmi", "u_mass", "c_uci"]
    for metric in metrics:
        coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, corpus=corpus, coherence=metric)
        coherence_scores[metric] = coherence_model.get_coherence()
    return coherence_scores

# Função para calcular coerência para LDA
def calculate_coherence_lda(lda_model, corpus, dictionary, texts):
    coherence_scores = {}
    metrics = ["c_v", "c_npmi", "u_mass", "c_uci"]
    for metric in metrics:
        coherence_model = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=dictionary, coherence=metric)
        coherence_scores[metric] = coherence_model.get_coherence()
    return coherence_scores

# Função para obter palavras principais do K-Means
def get_top_words_per_cluster(kmeans, vectorizer, num_top_words=20):
    terms = vectorizer.get_feature_names_out()
    top_words = []
    for i in range(kmeans.n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_terms_idx = cluster_center.argsort()[-num_top_words:][::-1]
        top_terms_idx = cluster_center.argsort()[::-1]
        top_terms = [terms[j] for j in top_terms_idx]
        top_words.append(top_terms)
    return top_words

# Função para treinar LDA
def train_lda_model(texts, num_topics=5):
    processed_texts = preprocess_texts(texts)
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary, processed_texts

# Função para avaliar K-Means e LDA
def evaluate_methods(X, texts, num_clusters_list):
    results = []
    processed_texts = preprocess_texts(texts)
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    for num_clusters in tqdm(num_clusters_list):
        # K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        top_words_kmeans = get_top_words_per_cluster(kmeans, vectorizer)
        coherence_kmeans = calculate_coherence_kmeans(top_words_kmeans, dictionary, corpus, processed_texts)
        silhouette_avg = silhouette_score(X, kmeans.labels_)

        # LDA
        lda_model, lda_corpus, lda_dictionary, lda_texts = train_lda_model(texts, num_topics=num_clusters)
        coherence_lda = calculate_coherence_lda(lda_model, lda_corpus, lda_dictionary, lda_texts)

        results.append({
            'num_topics/clusters': num_clusters,
            'silhouette_score_kmeans': silhouette_avg,
            'coherence_kmeans': coherence_kmeans,
            'coherence_lda': coherence_lda,
            'top_words_kmeans': top_words_kmeans,
            'lda_topics': lda_model.print_topics(num_words=10)
        })
    return results
# Carregar dados
university = "UNESP"
file_path = f"data/postprocessing/{university}/data.json"
texts = load_data(file_path)

# Vetorização com TF-IDF
stopwords = nltk.corpus.stopwords.words('portuguese')
vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
X = vectorizer.fit_transform(texts)

# Avaliar para diferentes números de clusters/tópicos
# num_clusters_list = [3, 5, 10, 20, 30, 50]
num_clusters_list = [3]
results = evaluate_methods(X, texts, num_clusters_list)

create_folder_if_not_exists(f"results/topics/{university}")
save_json(results, f"results/topics/{university}/lda_kmeans_comparison_results.json")
