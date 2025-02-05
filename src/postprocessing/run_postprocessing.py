from src.utils import save_json, load_json, create_folder_if_not_exists
from src.postprocessing.postprocessing import PostProcessing

# datasets = ["data/original/UFRB/data.json", 
#             "data/original/UFRJ/data.json", 
#             "data/original/UNESP/data.json", 
#             "data/original/UNICAMP/data.json",
#             "data/original/USP/data.json"]

datasets = ["data/original/UFRB/data.json", 
            "data/original/UFRJ/data.json", 
            "data/original/UNESP/data.json", 
            "data/original/UNICAMP/data.json"]

for dataset in datasets:
    dataset_name = dataset.split("/")[2]
    path_save = f"data/postprocessing/{dataset_name}"

    create_folder_if_not_exists(path_save)
            
    data = load_json(dataset)

    post_processing = PostProcessing(data)
    data_clean = post_processing.clean_data()

    save_json(data_clean, f"{path_save}/data.json")

