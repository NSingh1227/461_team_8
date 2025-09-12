import os
import requests
from .metric_calculator import ModelContext


class StorageManager:
    """
    Handles local storage of models and datasets.
    """

    def __init__(self, base_path: str = "./storage"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def download_file(self, url: str, subfolder: str) -> str:
        folder = os.path.join(self.base_path, subfolder)
        os.makedirs(folder, exist_ok=True)
        local_filename = os.path.join(folder, url.split("/")[-1])
        if os.path.exists(local_filename):
            return local_filename
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(response.content)
        return local_filename

    def store_model(self, context: ModelContext) -> None:
        if context.model_url:
            context.local_model_path = self.download_file(context.model_url, "models")
        if context.dataset_url:
            context.local_dataset_path = self.download_file(context.dataset_url, "datasets")
