import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import save_json, load_json, create_folder_if_not_exists, check_directory_exists

datasets = ["data/postprocessing/UFRB/data.json", 
            "data/postprocessing/UFRJ/data.json", 
            "data/postprocessing/UNESP/data.json", 
            "data/postprocessing/UNICAMP/data.json",
            "data/postprocessing/USP/data.json"]

model_names = ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
               "sentence-transformers/clip-ViT-B-32-multilingual-v1", 
               "alfaneo/bertimbau-base-portuguese-sts",
               "ibm-granite/granite-embedding-107m-multilingual",
               "intfloat/multilingual-e5-base"]

fields = ["Title", "Text"]

for dataset in datasets:
    for model_name in model_names:
        for field in fields:
            dataset_name = dataset.split("/")[2]
            model_name_last = model_name.split('/', 1)[1]
            path_save = f"data/embeddings/{dataset_name}/{model_name_last}"
            embedding_path = f"{path_save}/{field}_embeddings.npy"
            time_path = f"{path_save}/{field}_embeddings_time.json"

            if check_directory_exists(embedding_path):
                continue
            
            create_folder_if_not_exists(path_save)
            
            data = load_json(dataset)
            model = SentenceTransformer(model_name, trust_remote_code=True)
        
            #Preparar os dados
            texts = [instance.get(field) for instance in data]
            start_time = time.time()
            embeddings = model.encode(texts, show_progress_bar = True)
            end_time = time.time()

            save_json({"embeddings_generation_time": end_time - start_time}, time_path)
            np.save(embedding_path, embeddings)

