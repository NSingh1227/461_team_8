import re
import json
import requests
import time
import datetime
from enum import Enum
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from ..metrics.base import ModelContext
from ..storage.results_storage import ResultsStorage, MetricResult, ModelResult
from .exceptions import *
from ..metrics.license_calculator import LicenseCalculator
from ..metrics.dataset_code_calculator import DatasetCodeCalculator
from ..metrics.dataset_quality_calculator import DatasetQualityCalculator
from ..metrics.busfactor_calculator import BusFactorCalculator
from .http_client import get_with_rate_limit
from .rate_limiter import APIService

class URLType(Enum):
    HUGGINGFACE_MODEL = 'model'
    HUGGINGFACE_DATASET = 'dataset'
    GITHUB_REPO = 'code'
    UNKNOWN = 'unknown'

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
        
        response = get_with_rate_limit(api_url, APIService.HUGGINGFACE, timeout=10)
        if response and response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to fetch HuggingFace metadata for {url}: {e}")
        return None

def fetch_github_metadata(url: str) -> Optional[Dict[str, Any]]:
    """Retrieve repository metadata from GitHub API."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            
            response = get_with_rate_limit(api_url, APIService.GITHUB, timeout=10)
            if response and response.status_code == 200:
                return response.json()
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to fetch GitHub metadata for {url}: {e}")
        return None

def is_valid_url(url_string):
    """Check if URL has valid format and scheme."""
    if ' ' in url_string:
        return False
    
    parsed_url = urlparse(url_string)
    if parsed_url.scheme in ("https", "http") and parsed_url.netloc:
        return True
    else:
        return False
    
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

def process_url(url_string):
    """Validate and categorize URL."""
    if is_valid_url(url_string):
        return categorize_url(url_string)
    else:
        return URLType.UNKNOWN

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
        self.processed_datasets = set()  # Track datasets we've seen
    
    def parse_input_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a line with comma-separated URLs: code_url, dataset_url, model_url"""
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        # Split by comma and clean up
        parts = [part.strip() if part.strip() else None for part in line.split(',')]
        
        # Ensure we have exactly 3 parts
        while len(parts) < 3:
            parts.append(None)
        
        code_url = parts[0] if parts[0] else None
        dataset_url = parts[1] if parts[1] else None
        model_url = parts[2] if parts[2] else None
        
        return code_url, dataset_url, model_url
    
    def read_url_lines(self) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Read and parse URL lines from input file."""
        try:
            with open(self.file_path, 'r') as file:
                lines = []
                for line_num, line in enumerate(file, 1):
                    try:
                        code_url, dataset_url, model_url = self.parse_input_line(line)
                        if model_url:  # Only process lines with model URLs
                            lines.append((code_url, dataset_url, model_url))
                    except Exception as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
                return lines
        except FileNotFoundError:
            print(f"ERROR: The file '{self.file_path}' was not found.")
            return []
        except Exception as e:
            print(f"An error occurred reading file: {e}")
            return []
    
    def process_urls_with_metrics(self) -> List[ModelResult]:
        """Process URLs with full metric calculation."""
        url_lines = self.read_url_lines()
        model_results = []
        
        for code_url, dataset_url, model_url in url_lines:
            try:
                # Create model context
                model_context = self._create_model_context(model_url, code_url, dataset_url)
                
                if not model_context:
                    print(f"Warning: Could not create context for model: {model_url}")
                    continue
                
                # Calculate all metrics
                metrics = self._calculate_all_metrics(model_context)
                
                # Calculate net score using the specified weights
                net_score = self._calculate_net_score(metrics)
                net_score_latency = sum(metric.calculation_time_ms for metric in metrics.values())
                
                # Store metrics
                for metric in metrics.values():
                    self.results_storage.store_metric_result(model_url, metric)
                
                # Create and store final result
                model_result = self.results_storage.finalize_model_result(model_url, net_score, net_score_latency)
                model_results.append(model_result)
                
            except Exception as e:
                print(f"Error processing model {model_url}: {e}")
                continue
                
        return model_results
    
    def _create_model_context(self, model_url: str, code_url: Optional[str] = None, dataset_url: Optional[str] = None) -> Optional[ModelContext]:
        """Create ModelContext from URLs."""
        try:
            url_type = process_url(model_url)
            handler = get_handler(url_type)
            
            if not handler:
                return None
                
            # Get base context from handler
            context = handler.process_url(model_url)
            
            # Override with provided URLs
            if code_url:
                context.code_url = code_url
            if dataset_url:
                context.dataset_url = dataset_url
                self.processed_datasets.add(dataset_url)
            
            return context
            
        except Exception as e:
            print(f"Error creating context for {model_url}: {e}")
            return None
    
    def _calculate_all_metrics(self, model_context: ModelContext) -> Dict[str, MetricResult]:
        """Calculate all required metrics."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Import size calculator
        from ..metrics.size_calculator import SizeCalculator
        
        # Real metric calculators
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
        
        size_calc = SizeCalculator()
        size_scores = size_calc.calculate_score(model_context)  # Returns dict
        size_latency = size_calc.get_calculation_time()
        
        # Create MetricResult with size scores as a dict
        size_metric = MetricResult("Size", size_scores, size_latency, timestamp)
        
        # Dummy metrics for ones not yet implemented
        return {
            "Size": size_metric,
            "License": MetricResult("License", license_score, license_latency, timestamp),
            "RampUp": MetricResult("RampUp", 0.7, 200, timestamp),
            "BusFactor": MetricResult("BusFactor", busfactor_score, busfactor_latency, timestamp),
            "DatasetCode": MetricResult("DatasetCode", dac_score, dac_latency, timestamp),
            "DatasetQuality": MetricResult("DatasetQuality", dq_score, dq_latency, timestamp),
            "CodeQuality": MetricResult("CodeQuality", 0.9, 180, timestamp),
            "PerformanceClaims": MetricResult("PerformanceClaims", 0.8, 220, timestamp)
        }
    
    def _calculate_net_score(self, metrics: Dict[str, MetricResult]) -> float:
        """Calculate weighted net score using the formula from milestone document."""
        weights = {
            "License": 0.20,
            "RampUp": 0.20, 
            "BusFactor": 0.15,
            "DatasetCode": 0.15,
            "DatasetQuality": 0.10,
            "CodeQuality": 0.10,
            "PerformanceClaims": 0.05,
            "Size": 0.05
        }
        
        net_score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                metric_score = metrics[metric_name].score
                
                # Handle size score specially (it's a dict of platform scores)
                if metric_name == "Size" and isinstance(metric_score, dict):
                    # Use average of all platform scores as the single size score
                    size_score = sum(metric_score.values()) / len(metric_score)
                    net_score += weight * size_score
                else:
                    net_score += weight * metric_score
        
        return round(net_score, 3)


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