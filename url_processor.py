import re
import json
from enum import Enum
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class URLType(Enum):
    HUGGINGFACE_MODEL = 'model'
    HUGGINGFACE_DATASET = 'dataset'
    GITHUB_REPO = 'code'
    UNKNOWN = 'unknown'

def is_valid_url(url_string):
    parsed_url = urlparse(url_string)
    if parsed_url.scheme in ("https", "http") and parsed_url.netloc:
        return True
    else:
        return False
    
def categorize_url(url_string):
    parsed_url = urlparse(url_string)
    
    if parsed_url.netloc == "huggingface.co":
        if parsed_url.path.startswith("/datasets"):
            return URLType.HUGGINGFACE_DATASET
        else:
            return URLType.HUGGINGFACE_MODEL
    elif parsed_url.netloc == "github.com":
        return URLType.GITHUB_REPO
    else:
        return URLType.UNKNOWN

def process_url(url_string):
    if is_valid_url(url_string):
        return categorize_url(url_string)
    else:
        return URLType.UNKNOWN

def get_handler(url_type: URLType):
    if url_type == URLType.HUGGINGFACE_MODEL:
        return ModelHandler()
    elif url_type == URLType.HUGGINGFACE_DATASET:
        return DatasetHandler()
    elif url_type == URLType.GITHUB_REPO:
        return CodeHandler()
    return None

class URLProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_urls(self):
        try:
            with open(self.file_path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: The file '{self.file_path}' was not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    def process_urls(self):
        urls = self.read_urls()
        results = []
        for url in urls:
            url_type = process_url(url)
            handler = get_handler(url_type)
                    
            if handler:
                results.append(handler.process_url(url))
            else:
                results.append({
                    "url": url,
                    "type": URLType.UNKNOWN.value,
                    "info": {"status": "no handler available"}
                })
        return results

class URLHandler(ABC):
    @abstractmethod
    def process_url(self, url):
        """
        Process a URL and return structured metadata about it.
        For now, return a dictionary (later this will return a ModelContext).
        """
        pass
    
class DatasetHandler(URLHandler):
    def process_url(self, url):
        return {
            "url": url,
            "type": URLType.HUGGINGFACE_DATASET.value,
            "info": {"status": "stubbed dataset handler"}
        }


class ModelHandler(URLHandler):
    def process_url(self, url):
        return {
            "url": url,
            "type": URLType.HUGGINGFACE_MODEL.value,
            "info": {"status": "stubbed model handler"}
        }


class CodeHandler(URLHandler):
    def process_url(self, url):
        return {
            "url": url,
            "type": URLType.GITHUB_REPO.value,
            "info": {"status": "stubbed code handler"}
        }