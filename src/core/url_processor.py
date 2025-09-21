import re
import json
import requests
import time
import datetime
from enum import Enum
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from ..metrics.base import ModelContext
from ..storage.results_storage import ResultsStorage, MetricResult, ModelResult
from .exceptions import *
from ..metrics.license_calculator import LicenseCalculator
from ..metrics.dataset_code_calculator import DatasetCodeCalculator
from ..metrics.dataset_quality_calculator import DatasetQualityCalculator
from ..metrics.busfactor_calculator import BusFactorCalculator

class URLType(Enum):
    HUGGINGFACE_MODEL = 'model'
    HUGGINGFACE_DATASET = 'dataset'
    GITHUB_REPO = 'code'
    UNKNOWN = 'unknown'

# Fetch metadata from HuggingFace API for models or datasets
def fetch_huggingface_metadata(url: str, api_type: str = "models") -> Optional[Dict[str, Any]]:
    """Retrieve model or dataset metadata from HuggingFace API."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if api_type == "datasets":
            if len(path_parts) >= 2 and path_parts[0] == "datasets":
                if len(path_parts) >= 3:
                    repo_id = "/".join(path_parts[1:3])
                else:
                    repo_id = path_parts[1]
            else:
                return None
        else:
            if len(path_parts) >= 2:
                repo_id = "/".join(path_parts[0:2])
            else:
                return None
        
        api_url = f"https://huggingface.co/api/{api_type}/{repo_id}"
        
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            time.sleep(1)
            return None
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to fetch HuggingFace metadata for {url}: {e}")
        return None

# Fetch metadata from GitHub API for repositories
def fetch_github_metadata(url: str) -> Optional[Dict[str, Any]]:
    """Retrieve repository metadata from GitHub API."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(1)
                return None
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to fetch GitHub metadata for {url}: {e}")
        return None

# Validate URL format
def is_valid_url(url_string):
    """Check if URL has valid format and scheme."""
    if ' ' in url_string:
        return False
    
    parsed_url = urlparse(url_string)
    if parsed_url.scheme in ("https", "http") and parsed_url.netloc:
        return True
    else:
        return False
    
# Categorize URL by domain and path
def categorize_url(url_string):
    """Determine URL type based on domain and path structure."""
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

# Process URL and return its type
def process_url(url_string):
    """Validate and categorize URL."""
    if is_valid_url(url_string):
        return categorize_url(url_string)
    else:
        return URLType.UNKNOWN

# Get appropriate handler for URL type
def get_handler(url_type: URLType):
    """Return appropriate handler instance for URL type."""
    if url_type == URLType.HUGGINGFACE_MODEL:
        return ModelHandler()
    elif url_type == URLType.HUGGINGFACE_DATASET:
        return DatasetHandler()
    elif url_type == URLType.GITHUB_REPO:
        return CodeHandler()
    return None

class URLProcessor:
    """URL file processor for trustworthy model metrics pipeline."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results_storage = ResultsStorage()
    
    def read_urls(self) -> List[str]:
        """Read URLs from input file."""
        try:
            with open(self.file_path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: The file '{self.file_path}' was not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    def process_urls(self) -> List[Dict[str, Any]]:
        """Process URLs and return basic metadata."""
        urls = self.read_urls()
        results = []
        
        for url in urls:
            try:
                url_type = process_url(url)
                handler = get_handler(url_type)
                        
                if handler:
                    model_context = handler.process_url(url)
                    
                    result = {
                        "url": model_context.model_url,
                        "type": url_type.value,
                        "model_info": model_context.model_info,
                        "dataset_url": model_context.dataset_url,
                        "code_url": model_context.code_url,
                        "has_metadata": model_context.huggingface_metadata is not None
                    }
                    results.append(result)
                    
                else:
                    results.append({
                        "url": url,
                        "type": URLType.UNKNOWN.value,
                        "info": {"status": "no handler available"}
                    })
                    
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                results.append({
                    "url": url,
                    "type": URLType.UNKNOWN.value,
                    "info": {"status": f"error: {str(e)}"}
                })
        
        return results
    
    def process_urls_with_metrics(self) -> List[ModelResult]:
        """Process URLs with full metric calculation."""
        urls = self.read_urls()
        model_results = []
        
        for url in urls:
            try:
                url_type = process_url(url)
                handler = get_handler(url_type)
                
                if handler:
                    model_context = handler.process_url(url)
                    
                    dummy_metrics = self._create_dummy_metrics(model_context)
                    
                    net_score = sum(metric.score for metric in dummy_metrics.values()) / len(dummy_metrics)
                    net_score_latency = sum(metric.calculation_time_ms for metric in dummy_metrics.values())
                    
                    for metric in dummy_metrics.values():
                        self.results_storage.store_metric_result(url, metric)
                    
                    model_result = self.results_storage.finalize_model_result(url, net_score, net_score_latency)
                    model_results.append(model_result)
                    
                else:
                    print(f"Warning: No handler available for URL: {url}")
                    
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                
        return model_results
    
    def _create_dummy_metrics(self, model_context: ModelContext) -> Dict[str, MetricResult]:
        """Create metrics with real calculators where implemented, dummy scores for others."""
        import datetime
        
        timestamp = datetime.datetime.now().isoformat()
        
        license_calc = LicenseCalculator()
        license_score = license_calc.calculate_score(model_context)
        license_latency = license_calc.get_calculation_time()
        
        dac_calc = DatasetCodeCalculator()
        dac_score = dac_calc.calculate_score(model_context)
        dac_latency = dac_calc.get_calculation_time()
        
        dq_calc = DatasetQualityCalculator()
        dq_score = dq_calc.calculate_score(model_context)
        dq_latency = dq_calc.get_calculation_time()
        
        busfactor_calc = BusFactorCalculator()
        busfactor_score = busfactor_calc.calculate_score(model_context)
        busfactor_latency = busfactor_calc.get_calculation_time()
        
        return {
            "Size": MetricResult("Size", 0.8, 100, timestamp),
            "License": MetricResult("License", license_score, license_latency, timestamp),
            "RampUp": MetricResult("RampUp", 0.7, 200, timestamp),
            "BusFactor": MetricResult("BusFactor", busfactor_score, busfactor_latency, timestamp),
            "DatasetCode": MetricResult("DatasetCode", dac_score, dac_latency, timestamp),
            "DatasetQuality": MetricResult("DatasetQuality", dq_score, dq_latency, timestamp),
            "CodeQuality": MetricResult("CodeQuality", 0.9, 180, timestamp),
            "PerformanceClaims": MetricResult("PerformanceClaims", 0.8, 220, timestamp)
        }

class URLHandler(ABC):
    """Abstract interface for URL processing."""
    @abstractmethod
    def process_url(self, url: str) -> ModelContext:
        pass
    
class DatasetHandler(URLHandler):
    """Handle HuggingFace dataset URLs."""
    def process_url(self, url: str) -> ModelContext:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        dataset_info = {
            "source": "huggingface",
            "type": "dataset",
            "url": url,
            "path": parsed_url.path
        }
        
        if len(path_parts) >= 2 and path_parts[0] == "datasets":
            if len(path_parts) >= 3:
                dataset_info["owner"] = path_parts[1]
                dataset_info["name"] = path_parts[2]
            else:
                dataset_info["owner"] = None
                dataset_info["name"] = path_parts[1]
        
        huggingface_metadata = fetch_huggingface_metadata(url, "datasets")
        
        return ModelContext(
            model_url=url,
            model_info=dataset_info,
            dataset_url=url,
            code_url=None,
            local_repo_path=None,
            huggingface_metadata=huggingface_metadata
        )


class ModelHandler(URLHandler):
    """Handle HuggingFace model URLs."""
    def process_url(self, url: str) -> ModelContext:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        model_info = {
            "source": "huggingface",
            "type": "model",
            "url": url,
            "path": parsed_url.path
        }
        
        if len(path_parts) >= 2:
            model_info["owner"] = path_parts[0] if path_parts[0] else None
            model_info["name"] = path_parts[1] if len(path_parts) > 1 else None
        
        huggingface_metadata = fetch_huggingface_metadata(url, "models")
        
        return ModelContext(
            model_url=url,
            model_info=model_info,
            dataset_url=None,
            code_url=None,
            local_repo_path=None,
            huggingface_metadata=huggingface_metadata
        )


class CodeHandler(URLHandler):
    """Handle GitHub repository URLs."""
    def process_url(self, url: str) -> ModelContext:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        model_info = {
            "source": "github",
            "type": "repository",
            "url": url,
            "path": parsed_url.path
        }
        
        if len(path_parts) >= 2:
            model_info["owner"] = path_parts[0]
            model_info["repo"] = path_parts[1]
        
        github_metadata = fetch_github_metadata(url)
        if github_metadata:
            model_info.update({
                "github_metadata": github_metadata,
                "stars": github_metadata.get("stargazers_count", 0),
                "forks": github_metadata.get("forks_count", 0),
                "language": github_metadata.get("language"),
                "description": github_metadata.get("description"),
                "created_at": github_metadata.get("created_at"),
                "updated_at": github_metadata.get("updated_at")
            })
            
        return ModelContext(
            model_url=url,
            model_info=model_info,
            dataset_url=None,
            code_url=url,
            local_repo_path=None,
            huggingface_metadata=None
        )