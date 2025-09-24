from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from urllib.parse import urlparse


@dataclass
class MetricResult:
    """Individual metric calculation result."""
    metric_name: str
    score: float
    calculation_time_ms: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult:
    """Complete result set for a single model."""
    url: str
    net_score: float
    net_score_latency: int
    size_score: Dict[str, float]  # Changed to dict for hardware platforms
    size_latency: int
    license_score: float
    license_latency: int
    ramp_up_score: float
    ramp_up_latency: int
    bus_factor_score: float
    bus_factor_latency: int
    dataset_code_score: float
    dataset_code_latency: int
    dataset_quality_score: float
    dataset_quality_latency: int
    code_quality_score: float
    code_quality_latency: int
    performance_claims_score: float
    performance_claims_latency: int
    
    def _extract_model_name(self) -> str:
        """Extract model name from URL."""
        try:
            parsed_url = urlparse(self.url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            if 'huggingface.co' in parsed_url.netloc:
                if len(path_parts) >= 2:
                    return path_parts[1]  # Second part is the model name (owner/model)
                else:
                    return path_parts[0] if path_parts else "unknown"
            elif 'github.com' in parsed_url.netloc:
                if len(path_parts) >= 2:
                    return path_parts[1]  # Repo name
                else:
                    return "unknown"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def to_ndjson_line(self) -> str:
        """Convert to NDJSON format matching expected output."""
        model_name = self._extract_model_name()
        
        result_dict = {
            "name": model_name,
            "category": "MODEL",
            "netscore": round(self.net_score, 2),
            "netscore_latency": self.net_score_latency,
            "ramp_up_time": round(self.ramp_up_score, 2),
            "ramp_up_time_latency": self.ramp_up_latency,
            "bus_factor": round(self.bus_factor_score, 2),
            "bus_factor_latency": self.bus_factor_latency,
            "performance_claims": round(self.performance_claims_score, 2),
            "performance_claims_latency": self.performance_claims_latency,
            "license": round(self.license_score, 2),
            "license_latency": self.license_latency,
            "size_score": {
                "raspberry_pi": round(self.size_score.get("raspberry_pi", 0.0), 2),
                "jetson_nano": round(self.size_score.get("jetson_nano", 0.0), 2),
                "desktop_pc": round(self.size_score.get("desktop_pc", 0.0), 2),
                "aws_server": round(self.size_score.get("aws_server", 0.0), 2)
            },
            "size_score_latency": self.size_latency,
            "dataset_and_code_score": round(self.dataset_code_score, 2),
            "dataset_and_code_score_latency": self.dataset_code_latency,
            "dataset_quality": round(self.dataset_quality_score, 2),
            "dataset_quality_latency": self.dataset_quality_latency,
            "code_quality": round(self.code_quality_score, 2),
            "code_quality_latency": self.code_quality_latency
        }
        return json.dumps(result_dict)


class ResultsStorage:
    """Central storage for all metric calculation results."""
    def __init__(self):
        self._model_results: Dict[str, Dict[str, MetricResult]] = {}
        self._completed_models: List[ModelResult] = []
    
    def store_metric_result(self, model_url: str, metric_result: MetricResult) -> None:
        if model_url not in self._model_results:
            self._model_results[model_url] = {}
        
        self._model_results[model_url][metric_result.metric_name] = metric_result
    
    def get_metric_result(self, model_url: str, metric_name: str) -> Optional[MetricResult]:
        return self._model_results.get(model_url, {}).get(metric_name)
    
    def get_all_metrics_for_model(self, model_url: str) -> Dict[str, MetricResult]:
        return self._model_results.get(model_url, {})
    
    def is_model_complete(self, model_url: str) -> bool:
        required_metrics = {
            "Size", "License", "RampUp", "BusFactor", 
            "DatasetCode", "DatasetQuality", "CodeQuality", "PerformanceClaims"
        }
        
        model_metrics = set(self._model_results.get(model_url, {}).keys())
        return required_metrics.issubset(model_metrics)
    
    def finalize_model_result(self, model_url: str, net_score: float, net_score_latency: int) -> ModelResult:
        if not self.is_model_complete(model_url):
            raise ValueError(f"Model {model_url} does not have all required metrics calculated")
        
        metrics = self._model_results[model_url]
        
        # Handle size score - convert from single value to hardware platform dict
        size_metric = metrics["Size"]
        if isinstance(size_metric.score, dict):
            size_score = size_metric.score
        else:
            # Create default hardware compatibility scores
            size_score = {
                "raspberry_pi": size_metric.score * 0.2,
                "jetson_nano": size_metric.score * 0.4,
                "desktop_pc": size_metric.score * 0.8,
                "aws_server": size_metric.score
            }
        
        model_result = ModelResult(
            url=model_url,
            net_score=net_score,
            net_score_latency=net_score_latency,
            size_score=size_score,
            size_latency=metrics["Size"].calculation_time_ms,
            license_score=metrics["License"].score,
            license_latency=metrics["License"].calculation_time_ms,
            ramp_up_score=metrics["RampUp"].score,
            ramp_up_latency=metrics["RampUp"].calculation_time_ms,
            bus_factor_score=metrics["BusFactor"].score,
            bus_factor_latency=metrics["BusFactor"].calculation_time_ms,
            dataset_code_score=metrics["DatasetCode"].score,
            dataset_code_latency=metrics["DatasetCode"].calculation_time_ms,
            dataset_quality_score=metrics["DatasetQuality"].score,
            dataset_quality_latency=metrics["DatasetQuality"].calculation_time_ms,
            code_quality_score=metrics["CodeQuality"].score,
            code_quality_latency=metrics["CodeQuality"].calculation_time_ms,
            performance_claims_score=metrics["PerformanceClaims"].score,
            performance_claims_latency=metrics["PerformanceClaims"].calculation_time_ms
        )
        
        self._completed_models.append(model_result)
        return model_result
    
    def get_completed_models(self) -> List[ModelResult]:
        return self._completed_models.copy()
    
    def clear(self) -> None:
        self._model_results.clear()
        self._completed_models.clear()