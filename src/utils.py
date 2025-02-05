import os
import json

def save_json(data, file_path):
    """
    Save data to a JSON file.

    Args:
        data (any): The data to save, typically a dictionary or a list.
        file_path (str): The path to the JSON file where the data will be saved.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file to read from.

    Returns:
        any: The data read from the JSON file. If the file does not exist, returns None.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_progress(save_file_progress: str, page_number: int, last_saved_index: int) -> None:
    """
    Save the current progress to a JSON file.

    This function saves the current progress, including the page number and 
    the last saved index, to a specified JSON file. It helps track the 
    scraping progress between executions.

    Args:
        save_file_progress (str): The path to the file where progress should be saved.
        page_number (int): The current page number being scraped.
        last_saved_index (int): The index of the last successfully scraped item.

    Returns:
        None
    """
    progress = {"start_page": page_number, "last_saved_index": last_saved_index}
    save_json(progress, save_file_progress)


def save_data(save_file_data: str, all_data: list) -> None:
    """
    Save scraped data to a JSON file.

    This function saves all scraped data, such as news articles or other 
    information, to a specified JSON file.

    Args:
        save_file_data (str): The path to the file where the scraped data should be saved.
        all_data (list): The data to save, typically a list of dictionaries.

    Returns:
        None
    """
    save_json(all_data, save_file_data)


def save_error_log(save_file_error_log: str, error_log: list) -> None:
    """
    Save the error log to a JSON file.

    This function saves any errors encountered during the scraping process 
    to a specified JSON file for debugging and analysis.

    Args:
        save_file_error_log (str): The path to the file where the error log should be saved.
        error_log (list): The list of error messages or records encountered.

    Returns:
        None
    """
    save_json(error_log, save_file_error_log)

def check_directory_exists(directory_path: str) -> None:
    """
    Checks if a directory exists and raises an error if it does.

    Args:
        directory_path (str): The path to the directory to check.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        return True

    return False

def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it does not already exist.

    This function checks if a folder exists at the specified path. If the folder 
    does not exist, it creates it.

    Args:
        folder_path (str): The path of the folder to be checked and created if necessary.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def find_npy_files(directory):
    """
    Recursively finds all .npy files in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for .npy files.

    Returns:
        list: A list of paths to all .npy files found within the directory and its subdirectories.

    Example:
        >>> find_npy_files('/path/to/directory')
        ['/path/to/directory/file1.npy', '/path/to/directory/subdir/file2.npy']
    """
    npy_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files