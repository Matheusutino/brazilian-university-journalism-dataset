import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from src.utils import save_json, load_json, find_npy_files, create_folder_if_not_exists

# Define parameters
n_iter = 20
n_splits = 5
SEED = 42

embeddings_folder = "data/embeddings"
dataset_selected = "UFRJ"

embeddings_paths = find_npy_files(embeddings_folder)

for embeddings_path in embeddings_paths:
    path_parts = embeddings_path.split('/')

    dataset = path_parts[2]
    embedding_model = path_parts[3]
    file_path = path_parts[4]

    # Ignorar as embeddigns de título
    if file_path != "Title_embeddings.npy" or dataset != dataset_selected:
        continue

    path_data = f"data/postprocessing/{dataset}/data.json"
    path_save = f"results/category_classification/{dataset}/{embedding_model}"

    data = load_json(path_data)

    X = np.load(embeddings_path)
    y = []

    valid_indices = []  # Lista para armazenar os índices válidos

    # Verifica se "News Category" não é uma lista e filtra
    for i, instance in enumerate(data):
        category = instance.get("News Category")
        if category and not isinstance(category, list):
            y.append(category)
            valid_indices.append(i)  # Armazena o índice da instância válida

    # Filtrar X e y de acordo com os índices válidos
    X = X[valid_indices]

    # Filtrar as classes com menos de 10 instâncias
    class_counts = Counter(y)
    valid_classes = [cls for cls, count in class_counts.items() if count >= 15]

    # Filtrar os índices para manter apenas as classes válidas
    final_valid_indices = [i for i, label in enumerate(y) if label in valid_classes and label is not None]

    # Atualizar X e y com os índices finais válidos
    X = X[final_valid_indices]
    y = [y[i].lower() for i in final_valid_indices]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Definir a validação cruzada estratificada
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Definir o modelo KNN
    knn = KNeighborsClassifier()

    # Definir os parâmetros para a busca bayesiana
    param_grid = {
        'n_neighbors': list(range(2, 100)),  # Variação de K
        'metric': ['euclidean', 'manhattan', 'cosine'],  # Variação de metric
        'weights': ['uniform', 'distance']  # Variação de pesos
    }

    # Define the score metrics
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro")
    }

    # Realizar a busca em grade com validação cruzada estratificada
    bayes_search = BayesSearchCV(estimator=knn, 
                                search_spaces=param_grid, 
                                n_iter = n_iter,
                                cv=skf, 
                                scoring=scoring, 
                                return_train_score=True, 
                                refit="f1_score", 
                                random_state=SEED, 
                                n_jobs=-1)

    # Ajustar o modelo aos dados
    bayes_search.fit(X, y)

    # Converter os resultados do BayesSearchCV em um DataFrame
    results_df = pd.DataFrame(bayes_search.cv_results_)

    create_folder_if_not_exists(path_save)
    
    # Salvar todos os resultados
    results_df.to_csv(f'{path_save}/bayes_search_results.csv', index=False)

    # Encontrar a linha com o maior f1_score médio no teste
    best_f1_row = results_df.loc[results_df['mean_test_f1_score'].idxmax()]

    # Salvar essa linha específica em um arquivo CSV
    best_f1_row.to_frame().T.to_csv(f'{path_save}/best_f1_score_model.csv', index=False)

    infos = {
        "Number of instances": len(y),
        "Number of categories": len(set(y))
    }

    save_json(infos, f'{path_save}/infos.json')