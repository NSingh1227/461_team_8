"""
Unit tests for ResultsStorage system.
Part of Milestone 1 validation requirements.
"""

import pytest
import json
from datetime import datetime
from src.results_storage import ResultsStorage, MetricResult, ModelResult


class TestMetricResult:
    """Test suite for MetricResult data class."""
    
    def test_metric_result_creation(self):
        """Test MetricResult creation and serialization."""
        result = MetricResult(
            metric_name="License",
            score=1.0,
            calculation_time_ms=150,
            timestamp="2024-09-12T10:30:00Z"
        )
        
        assert result.metric_name == "License"
        assert result.score == 1.0
        assert result.calculation_time_ms == 150
        assert result.timestamp == "2024-09-12T10:30:00Z"
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        expected = {
            "metric_name": "License",
            "score": 1.0,
            "calculation_time_ms": 150,
            "timestamp": "2024-09-12T10:30:00Z"
        }
        assert result_dict == expected


class TestModelResult:
    """Test suite for ModelResult data class."""
    
    def test_model_result_ndjson_output(self):
        """Test ModelResult NDJSON output format matches specification."""
        model_result = ModelResult(
            url="https://huggingface.co/test/model",
            net_score=0.75,
            net_score_latency=500,
            size_score=0.8,
            size_latency=100,
            license_score=1.0,
            license_latency=150,
            ramp_up_score=0.6,
            ramp_up_latency=200,
            bus_factor_score=0.7,
            bus_factor_latency=300,
            dataset_code_score=0.9,
            dataset_code_latency=250,
            dataset_quality_score=0.5,
            dataset_quality_latency=400,
            code_quality_score=0.8,
            code_quality_latency=350,
            performance_claims_score=0.6,
            performance_claims_latency=180
        )
        
        ndjson_line = model_result.to_ndjson_line()
        parsed = json.loads(ndjson_line)
        
        # Verify required NDJSON fields match specification
        expected_fields = {
            "URL": "https://huggingface.co/test/model",
            "NetScore": 0.75,
            "NetScore_Latency": 500,
            "RampUp": 0.6,
            "RampUp_Latency": 200,
            "Correctness": 0.6,  # Performance claims mapped to Correctness
            "Correctness_Latency": 180,
            "BusFactor": 0.7,
            "BusFactor_Latency": 300,
            "ResponsiveMaintainer": 0.9,  # DAC mapped to ResponsiveMaintainer
            "ResponsiveMaintainer_Latency": 250,
            "License": 1.0,
            "License_Latency": 150
        }
        
        for field, expected_value in expected_fields.items():
            assert field in parsed
            assert parsed[field] == expected_value


class TestResultsStorage:
    """Test suite for ResultsStorage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.storage = ResultsStorage()
        self.model_url = "https://huggingface.co/test/model"
        
        # Create sample metric results
        self.license_result = MetricResult(
            metric_name="License",
            score=1.0,
            calculation_time_ms=150,
            timestamp="2024-09-12T10:30:00Z"
        )
        
        self.size_result = MetricResult(
            metric_name="Size",
            score=0.8,
            calculation_time_ms=100,
            timestamp="2024-09-12T10:30:01Z"
        )
    
    def test_store_and_retrieve_metric_result(self):
        """Test storing and retrieving individual metric results."""
        # Store result
        self.storage.store_metric_result(self.model_url, self.license_result)
        
        # Retrieve result
        retrieved = self.storage.get_metric_result(self.model_url, "License")
        
        assert retrieved is not None
        assert retrieved.metric_name == "License"
        assert retrieved.score == 1.0
        assert retrieved.calculation_time_ms == 150
    
    def test_get_nonexistent_metric(self):
        """Test retrieving non-existent metric returns None."""
        result = self.storage.get_metric_result("nonexistent", "License")
        assert result is None
        
        result = self.storage.get_metric_result(self.model_url, "NonexistentMetric")
        assert result is None
    
    def test_get_all_metrics_for_model(self):
        """Test retrieving all metrics for a model."""
        # Store multiple results
        self.storage.store_metric_result(self.model_url, self.license_result)
        self.storage.store_metric_result(self.model_url, self.size_result)
        
        # Get all metrics
        all_metrics = self.storage.get_all_metrics_for_model(self.model_url)
        
        assert len(all_metrics) == 2
        assert "License" in all_metrics
        assert "Size" in all_metrics
        assert all_metrics["License"].score == 1.0
        assert all_metrics["Size"].score == 0.8
    
    def test_is_model_complete(self):
        """Test checking if all required metrics are calculated."""
        # Initially incomplete
        assert not self.storage.is_model_complete(self.model_url)
        
        # Add some metrics (still incomplete)
        self.storage.store_metric_result(self.model_url, self.license_result)
        self.storage.store_metric_result(self.model_url, self.size_result)
        assert not self.storage.is_model_complete(self.model_url)
        
        # Add all required metrics
        required_metrics = [
            "Size", "License", "RampUp", "BusFactor",
            "DatasetCode", "DatasetQuality", "CodeQuality", "PerformanceClaims"
        ]
        
        for metric_name in required_metrics:
            if metric_name not in ["License", "Size"]:  # Already added
                result = MetricResult(
                    metric_name=metric_name,
                    score=0.5,
                    calculation_time_ms=100,
                    timestamp="2024-09-12T10:30:00Z"
                )
                self.storage.store_metric_result(self.model_url, result)
        
        # Now should be complete
        assert self.storage.is_model_complete(self.model_url)
    
    def test_finalize_model_result(self):
        """Test finalizing a complete model result."""
        # Add all required metrics
        required_metrics = [
            "Size", "License", "RampUp", "BusFactor",
            "DatasetCode", "DatasetQuality", "CodeQuality", "PerformanceClaims"
        ]
        
        for i, metric_name in enumerate(required_metrics):
            result = MetricResult(
                metric_name=metric_name,
                score=0.1 * (i + 1),  # Different scores for each metric
                calculation_time_ms=100 + i * 10,
                timestamp="2024-09-12T10:30:00Z"
            )
            self.storage.store_metric_result(self.model_url, result)
        
        # Finalize result
        model_result = self.storage.finalize_model_result(
            self.model_url, 
            net_score=0.75, 
            net_score_latency=500
        )
        
        assert model_result.url == self.model_url
        assert model_result.net_score == 0.75
        assert model_result.net_score_latency == 500
        assert model_result.license_score == 0.2  # Second metric added
        assert model_result.size_score == 0.1     # First metric added
        
        # Check it's in completed models
        completed = self.storage.get_completed_models()
        assert len(completed) == 1
        assert completed[0].url == self.model_url
    
    def test_finalize_incomplete_model_raises_error(self):
        """Test that finalizing incomplete model raises ValueError."""
        # Only add one metric
        self.storage.store_metric_result(self.model_url, self.license_result)
        
        with pytest.raises(ValueError, match="does not have all required metrics"):
            self.storage.finalize_model_result(self.model_url, 0.5, 100)
    
    def test_clear_storage(self):
        """Test clearing all stored results."""
        # Add some data
        self.storage.store_metric_result(self.model_url, self.license_result)
        
        # Verify data exists
        assert len(self.storage.get_all_metrics_for_model(self.model_url)) == 1
        
        # Clear and verify empty
        self.storage.clear()
        assert len(self.storage.get_all_metrics_for_model(self.model_url)) == 0
        assert len(self.storage.get_completed_models()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
