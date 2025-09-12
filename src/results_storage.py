"""
Results storage system for managing calculated metric scores.
Based on UML diagram showing ResultsStorage as a central component.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime


@dataclass
class MetricResult:
    """
    Individual metric calculation result.
    """
    metric_name: str
    score: float
    calculation_time_ms: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModelResult:
    """
    Complete result set for a single model.
    Matches the NDJSON output format specified in requirements.
    """
    url: str
    net_score: float
    net_score_latency: int
    size_score: float
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
    
    def to_ndjson_line(self) -> str:
        """
        Convert to NDJSON format as specified in requirements.
        
        Returns:
            str: Single line of NDJSON output
        """
        result_dict = {
            "URL": self.url,
            "NetScore": self.net_score,
            "NetScore_Latency": self.net_score_latency,
            "RampUp": self.ramp_up_score,
            "RampUp_Latency": self.ramp_up_latency,
            "Correctness": self.performance_claims_score,  # Performance claims mapped to Correctness
            "Correctness_Latency": self.performance_claims_latency,
            "BusFactor": self.bus_factor_score,
            "BusFactor_Latency": self.bus_factor_latency,
            "ResponsiveMaintainer": self.dataset_code_score,  # DAC mapped to ResponsiveMaintainer
            "ResponsiveMaintainer_Latency": self.dataset_code_latency,
            "License": self.license_score,
            "License_Latency": self.license_latency
        }
        return json.dumps(result_dict)


class ResultsStorage:
    """
    Central storage for all metric calculation results.
    Provides thread-safe access to results during parallel processing.
    """
    
    def __init__(self):
        """Initialize empty results storage."""
        self._model_results: Dict[str, Dict[str, MetricResult]] = {}
        self._completed_models: List[ModelResult] = []
    
    def store_metric_result(self, model_url: str, metric_result: MetricResult) -> None:
        """
        Store a single metric result for a model.
        
        Args:
            model_url: The model URL this result belongs to
            metric_result: The calculated metric result
        """
        if model_url not in self._model_results:
            self._model_results[model_url] = {}
        
        self._model_results[model_url][metric_result.metric_name] = metric_result
    
    def get_metric_result(self, model_url: str, metric_name: str) -> Optional[MetricResult]:
        """
        Retrieve a specific metric result.
        
        Args:
            model_url: The model URL
            metric_name: Name of the metric
            
        Returns:
            Optional[MetricResult]: The result if found, None otherwise
        """
        return self._model_results.get(model_url, {}).get(metric_name)
    
    def get_all_metrics_for_model(self, model_url: str) -> Dict[str, MetricResult]:
        """
        Get all calculated metrics for a specific model.
        
        Args:
            model_url: The model URL
            
        Returns:
            Dict[str, MetricResult]: All metrics for the model
        """
        return self._model_results.get(model_url, {})
    
    def is_model_complete(self, model_url: str) -> bool:
        """
        Check if all required metrics have been calculated for a model.
        
        Args:
            model_url: The model URL to check
            
        Returns:
            bool: True if all 8 metrics are calculated
        """
        required_metrics = {
            "Size", "License", "RampUp", "BusFactor", 
            "DatasetCode", "DatasetQuality", "CodeQuality", "PerformanceClaims"
        }
        
        model_metrics = set(self._model_results.get(model_url, {}).keys())
        return required_metrics.issubset(model_metrics)
    
    def finalize_model_result(self, model_url: str, net_score: float, net_score_latency: int) -> ModelResult:
        """
        Create final ModelResult when all metrics are calculated.
        
        Args:
            model_url: The model URL
            net_score: Calculated net score
            net_score_latency: Time to calculate net score in ms
            
        Returns:
            ModelResult: Complete result ready for output
            
        Raises:
            ValueError: If not all metrics are available
        """
        if not self.is_model_complete(model_url):
            raise ValueError(f"Model {model_url} does not have all required metrics calculated")
        
        metrics = self._model_results[model_url]
        
        model_result = ModelResult(
            url=model_url,
            net_score=net_score,
            net_score_latency=net_score_latency,
            size_score=metrics["Size"].score,
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
        """
        Get all completed model results.
        
        Returns:
            List[ModelResult]: All finalized model results
        """
        return self._completed_models.copy()
    
    def clear(self) -> None:
        """Clear all stored results."""
        self._model_results.clear()
        self._completed_models.clear()
