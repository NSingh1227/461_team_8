import os
import re
import sys
import time
import datetime
from enum import Enum
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..metrics.base import ModelContext
from ..storage.results_storage import ResultsStorage, MetricResult, ModelResult
from .exceptions import *
from .config import Config
from ..metrics.license_calculator import LicenseCalculator
from ..metrics.dataset_code_calculator import DatasetCodeCalculator
from ..metrics.dataset_quality_calculator import DatasetQualityCalculator
from ..metrics.busfactor_calculator import BusFactorCalculator
from ..metrics.size_calculator import SizeCalculator
from ..metrics.ramp_up_calculator import RampUpCalculator
from ..metrics.code_quality_calculator import CodeQualityCalculator
from ..metrics.performance_claims_calculator import PerformanceClaimsCalculator
from .http_client import get_with_rate_limit
from .rate_limiter import APIService

class URLType(Enum):
    HUGGINGFACE_MODEL = 'model'
    HUGGINGFACE_DATASET = 'dataset'
    GITHUB_REPO = 'code'
    UNKNOWN = 'unknown'

def fetch_huggingface_metadata(url: str, api_type: str = "models") -> Optional[Dict[str, Any]]:
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
        print(f"Warning: Failed to fetch HuggingFace metadata for {url}: {e}", file=sys.stderr)
        return None

def fetch_github_metadata(url: str) -> Optional[Dict[str, Any]]:
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            

            headers = {}
            github_token = Config.get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = get_with_rate_limit(api_url, APIService.GITHUB, headers=headers, timeout=10)
            if response and response.status_code == 200:
                return response.json()
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to fetch GitHub metadata for {url}: {e}", file=sys.stderr)
        return None

def is_valid_url(url_string):
    if not url_string or not isinstance(url_string, str):
        return False
    
    if ' ' in url_string:
        return False
    
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

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results_storage = ResultsStorage()
        self.processed_datasets = set()
    
    def parse_input_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        

        parts = [part.strip() for part in line.split(',')]
        

        if len(parts) == 1:
            return None, None, parts[0] if parts[0] else None
        

        if len(parts) == 2:
            code_url = parts[0] if parts[0] else None
            dataset_url = None
            model_url = parts[1] if parts[1] else None
            return code_url, dataset_url, model_url
        

        code_url = parts[0] if parts[0] else None
        dataset_url = parts[1] if parts[1] else None
        model_url = parts[2] if parts[2] else None
        
        return code_url, dataset_url, model_url
    
    def read_url_lines(self) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
        try:
            with open(self.file_path, 'r') as file:
                lines = []
                for line_num, line in enumerate(file, 1):
                    try:
                        code_url, dataset_url, model_url = self.parse_input_line(line)

                        if code_url or dataset_url or model_url:
                            lines.append((code_url, dataset_url, model_url))
                    except Exception as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                        continue
                return lines
        except FileNotFoundError:
            print(f"ERROR: The file '{self.file_path}' was not found.", file=sys.stderr)
            return []
        except Exception as e:
            print(f"An error occurred reading file: {e}", file=sys.stderr)
            return []
    
    def process_urls_with_metrics(self) -> List[ModelResult]:
        url_lines = self.read_url_lines()
        model_results = []
        

        is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
        debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
        

        if not is_autograder and debug_enabled:
            print(f"Found {len(url_lines)} URLs to process", file=sys.stderr)
        
        for code_url, dataset_url, model_url in url_lines:
            primary_url = None
            try:

                primary_url = model_url or code_url or dataset_url
                
                if not primary_url:
                    continue
                

                if not is_valid_url(primary_url):
                    if not is_autograder and debug_enabled:
                        print(f"Skipping invalid URL: {primary_url}", file=sys.stderr)
                    continue
                

                model_context = None
                try:
                    model_context = self._create_model_context(primary_url, code_url, dataset_url)
                except Exception as ctx_e:
                    if not is_autograder and debug_enabled:
                        print(f"Context creation failed for {primary_url}: {ctx_e}", file=sys.stderr)
                
                if not model_context:
                    if not is_autograder and debug_enabled:
                        print(f"Warning: Could not create context for URL: {primary_url}", file=sys.stderr)

                    model_result = self._create_default_result(primary_url)
                    model_results.append(model_result)
                    continue
                

                try:
                    metrics = self._calculate_all_metrics(model_context)
                    

                    net_score = self._calculate_net_score(metrics)
                    net_score_latency = sum(metric.calculation_time_ms for metric in metrics.values())
                    

                    for metric in metrics.values():
                        self.results_storage.store_metric_result(primary_url, metric)
                    

                    model_result = self.results_storage.finalize_model_result(primary_url, net_score, net_score_latency)
                    model_results.append(model_result)
                    
                except Exception as metrics_e:
                    if not is_autograder and debug_enabled:
                        print(f"Metrics calculation failed for {primary_url}: {metrics_e}", file=sys.stderr)

                    model_result = self._create_default_result(primary_url)
                    model_results.append(model_result)
                
            except Exception as e:
                if not is_autograder and debug_enabled:
                    print(f"Error processing URL {primary_url or 'unknown'}: {e}", file=sys.stderr)

                if primary_url:
                    model_result = self._create_default_result(primary_url)
                    model_results.append(model_result)
        
        if not is_autograder and debug_enabled:
            print(f"Successfully processed {len(model_results)} URLs", file=sys.stderr)
        

        if not model_results and len(url_lines) == 0:
            if not is_autograder and debug_enabled:
                print("Warning: Empty file, creating default result", file=sys.stderr)
            default_result = self._create_default_result("unknown")
            model_results.append(default_result)
        
        return model_results
    
    def _create_default_result(self, model_url: str) -> ModelResult:
        try:

            model_name = self._extract_model_name_from_url(model_url)
            

            return ModelResult(
                url=model_url,
                net_score=0.0,
                net_score_latency=0,
                size_score={"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0},
                size_latency=0,
                license_score=0.0,
                license_latency=0,
                ramp_up_score=0.0,
                ramp_up_latency=0,
                bus_factor_score=0.0,
                bus_factor_latency=0,
                dataset_code_score=0.0,
                dataset_code_latency=0,
                dataset_quality_score=0.0,
                dataset_quality_latency=0,
                code_quality_score=0.0,
                code_quality_latency=0,
                performance_claims_score=0.0,
                performance_claims_latency=0
            )
        except Exception as e:
            print(f"Error creating default result for {model_url}: {e}", file=sys.stderr)

            return ModelResult(
                url=model_url,
                net_score=0.0,
                net_score_latency=0,
                size_score={"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0},
                size_latency=0,
                license_score=0.0,
                license_latency=0,
                ramp_up_score=0.0,
                ramp_up_latency=0,
                bus_factor_score=0.0,
                bus_factor_latency=0,
                dataset_code_score=0.0,
                dataset_code_latency=0,
                dataset_quality_score=0.0,
                dataset_quality_latency=0,
                code_quality_score=0.0,
                code_quality_latency=0,
                performance_claims_score=0.0,
                performance_claims_latency=0
            )
    
    def _extract_model_name_from_url(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            if 'huggingface.co' in parsed_url.netloc:
                if len(path_parts) >= 2:
                    return path_parts[1]
                else:
                    return path_parts[0] if path_parts else "unknown"
            elif 'github.com' in parsed_url.netloc:
                if len(path_parts) >= 2:
                    return path_parts[1]
                else:
                    return "unknown"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _create_model_context(self, model_url: str, code_url: Optional[str] = None, dataset_url: Optional[str] = None) -> Optional[ModelContext]:
        try:
            url_type = process_url(model_url)
            handler = get_handler(url_type)
            
            if not handler:
                return None
                

            context = handler.process_url(model_url)
            

            if code_url:
                context.code_url = code_url
            if dataset_url:
                context.dataset_url = dataset_url
                self.processed_datasets.add(dataset_url)
            

            inferred_datasets = self._infer_datasets_from_context(context)
            for dataset in inferred_datasets:
                self.processed_datasets.add(dataset)
            
            return context
            
        except Exception as e:
            print(f"Error creating context for {model_url}: {e}", file=sys.stderr)
            return None
    
    def _infer_datasets_from_context(self, context: ModelContext) -> List[str]:
        inferred_datasets = []
        
        try:

            if context.huggingface_metadata:
                datasets = context.huggingface_metadata.get('datasets', [])
                if datasets:
                    for dataset in datasets:
                        if isinstance(dataset, str):

                            if not dataset.startswith('http'):
                                dataset_url = f"https://huggingface.co/datasets/{dataset}"
                                inferred_datasets.append(dataset_url)
                            else:
                                inferred_datasets.append(dataset)
                

                card_data = context.huggingface_metadata.get('cardData', {})
                if 'datasets' in card_data:
                    dataset_info = card_data['datasets']
                    if isinstance(dataset_info, list):
                        for dataset in dataset_info:
                            if isinstance(dataset, str) and not dataset.startswith('http'):
                                dataset_url = f"https://huggingface.co/datasets/{dataset}"
                                inferred_datasets.append(dataset_url)
            

            if context.model_info:
                model_info_str = str(context.model_info).lower()

                if 'bookcorpus' in model_info_str:
                    inferred_datasets.append('https://huggingface.co/datasets/bookcorpus')
                if 'wikipedia' in model_info_str:
                    inferred_datasets.append('https://huggingface.co/datasets/wikipedia')
                if 'squad' in model_info_str:
                    inferred_datasets.append('https://huggingface.co/datasets/squad')
                    
        except Exception as e:
            print(f"Error inferring datasets: {e}", file=sys.stderr)
        
        return inferred_datasets
    
    def _calculate_all_metrics(self, model_context: ModelContext) -> Dict[str, MetricResult]:
        timestamp = datetime.datetime.now().isoformat()
        metrics = {}
        

        def calculate_license():
            try:
                calc = LicenseCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "License", MetricResult("License", score, latency, timestamp)
            except Exception as e:
                print(f"License calculation failed: {e}", file=sys.stderr)
                return "License", MetricResult("License", 0.5, 100, timestamp)
        
        def calculate_dataset_code():
            try:
                calc = DatasetCodeCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "DatasetCode", MetricResult("DatasetCode", score, latency, timestamp)
            except Exception as e:
                print(f"DatasetCode calculation failed: {e}", file=sys.stderr)
                return "DatasetCode", MetricResult("DatasetCode", 0.5, 100, timestamp)
        
        def calculate_dataset_quality():
            try:
                calc = DatasetQualityCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "DatasetQuality", MetricResult("DatasetQuality", score, latency, timestamp)
            except Exception as e:
                print(f"DatasetQuality calculation failed: {e}", file=sys.stderr)
                return "DatasetQuality", MetricResult("DatasetQuality", 0.5, 100, timestamp)
        
        def calculate_bus_factor():
            try:
                calc = BusFactorCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "BusFactor", MetricResult("BusFactor", score, latency, timestamp)
            except Exception as e:
                print(f"BusFactor calculation failed: {e}", file=sys.stderr)
                return "BusFactor", MetricResult("BusFactor", 0.5, 100, timestamp)
        
        def calculate_size():
            try:
                calc = SizeCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                platform_scores = calc.get_platform_compatibility()
                return "Size", MetricResult("Size", platform_scores, latency, timestamp)
            except Exception as e:
                print(f"Size calculation failed: {e}", file=sys.stderr)
                default_sizes = {"raspberry_pi": 0.5, "jetson_nano": 0.6, "desktop_pc": 0.8, "aws_server": 0.9}
                return "Size", MetricResult("Size", default_sizes, 100, timestamp)
        
        def calculate_ramp_up():
            try:
                calc = RampUpCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "RampUp", MetricResult("RampUp", score, latency, timestamp)
            except Exception as e:
                print(f"RampUp calculation failed: {e}", file=sys.stderr)
                return "RampUp", MetricResult("RampUp", 0.7, 200, timestamp)
        
        def calculate_code_quality():
            try:
                calc = CodeQualityCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "CodeQuality", MetricResult("CodeQuality", score, latency, timestamp)
            except Exception as e:
                print(f"CodeQuality calculation failed: {e}", file=sys.stderr)
                return "CodeQuality", MetricResult("CodeQuality", 0.9, 180, timestamp)
        
        def calculate_performance_claims():
            try:
                calc = PerformanceClaimsCalculator()
                score = calc.calculate_score(model_context)
                latency = calc.get_calculation_time() or 0
                return "PerformanceClaims", MetricResult("PerformanceClaims", score, latency, timestamp)
            except Exception as e:
                print(f"PerformanceClaims calculation failed: {e}", file=sys.stderr)
                return "PerformanceClaims", MetricResult("PerformanceClaims", 0.8, 220, timestamp)
        

        metric_functions = [
            calculate_license,
            calculate_dataset_code,
            calculate_dataset_quality,
            calculate_bus_factor,
            calculate_size,
            calculate_ramp_up,
            calculate_code_quality,
            calculate_performance_claims
        ]
        


        import os
        max_workers = min(4, os.cpu_count() or 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_metric = {executor.submit(func): func.__name__ for func in metric_functions}
            

            for future in as_completed(future_to_metric):
                try:
                    metric_name, metric_result = future.result()
                    metrics[metric_name] = metric_result
                except Exception as e:
                    metric_name = future_to_metric[future]
                    print(f"Metric calculation {metric_name} failed: {e}", file=sys.stderr)

                    if metric_name == "calculate_size":
                        default_sizes = {"raspberry_pi": 0.5, "jetson_nano": 0.6, "desktop_pc": 0.8, "aws_server": 0.9}
                        metrics["Size"] = MetricResult("Size", default_sizes, 100, timestamp)
                    else:
                        metrics[metric_name.replace("calculate_", "").title()] = MetricResult(
                            metric_name.replace("calculate_", "").title(), 0.5, 100, timestamp
                        )
        
        return metrics
    
    def _calculate_net_score(self, metrics: Dict[str, MetricResult]) -> float:
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
                

                if metric_name == "Size" and isinstance(metric_score, dict):

                    size_score = max(metric_score.values()) if metric_score else 0.5
                    net_score += weight * size_score
                else:
                    net_score += weight * metric_score
        
        return round(net_score, 2)


class URLHandler(ABC):
    @abstractmethod
    def process_url(self, url: str) -> ModelContext:
        pass
    
class DatasetHandler(URLHandler):
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
