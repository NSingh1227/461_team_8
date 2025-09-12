"""
Unit tests for the abstract MetricCalculator base class.
Part of Milestone 1 validation requirements.
"""

import pytest
from unittest.mock import Mock
from src.metric_calculator import MetricCalculator, ModelContext
from src.exceptions import MetricCalculationException


class ConcreteMetricCalculator(MetricCalculator):
    """Concrete implementation for testing the abstract base class."""
    
    def __init__(self, name: str, return_score: float = 0.5):
        super().__init__(name)
        self.return_score = return_score
    
    def calculate_score(self, context: ModelContext) -> float:
        """Test implementation that returns a fixed score."""
        import time
        start_time = time.time()
        
        # Simulate some calculation time
        time.sleep(0.001)
        
        calculation_time_ms = int((time.time() - start_time) * 1000)
        self._set_score(self.return_score, calculation_time_ms)
        
        return self.return_score


class TestMetricCalculator:
    """Test suite for MetricCalculator abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that MetricCalculator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricCalculator("test")
    
    def test_concrete_implementation_initialization(self):
        """Test that concrete implementations can be created properly."""
        calculator = ConcreteMetricCalculator("TestMetric")
        
        assert calculator.name == "TestMetric"
        assert calculator.get_score() is None
        assert calculator.get_calculation_time() is None
    
    def test_calculate_score_interface(self):
        """Test the calculate_score interface works correctly."""
        calculator = ConcreteMetricCalculator("TestMetric", 0.8)
        context = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={"name": "test-model"}
        )
        
        score = calculator.calculate_score(context)
        
        assert score == 0.8
        assert calculator.get_score() == 0.8
        assert calculator.get_calculation_time() is not None
        assert calculator.get_calculation_time() > 0
    
    def test_score_validation(self):
        """Test that scores must be in valid range [0, 1]."""
        calculator = ConcreteMetricCalculator("TestMetric")
        
        # Test invalid scores
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            calculator._set_score(-0.1, 100)
        
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            calculator._set_score(1.1, 100)
        
        # Test valid scores
        calculator._set_score(0.0, 100)
        assert calculator.get_score() == 0.0
        
        calculator._set_score(1.0, 100)
        assert calculator.get_score() == 1.0
        
        calculator._set_score(0.5, 100)
        assert calculator.get_score() == 0.5
    
    def test_reset_functionality(self):
        """Test that reset clears calculator state."""
        calculator = ConcreteMetricCalculator("TestMetric", 0.7)
        context = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={"name": "test-model"}
        )
        
        # Calculate score
        calculator.calculate_score(context)
        assert calculator.get_score() == 0.7
        assert calculator.get_calculation_time() is not None
        
        # Reset and verify state is cleared
        calculator.reset()
        assert calculator.get_score() is None
        assert calculator.get_calculation_time() is None
    
    def test_string_representations(self):
        """Test string representation methods."""
        calculator = ConcreteMetricCalculator("TestMetric")
        
        assert str(calculator) == "ConcreteMetricCalculator(name='TestMetric')"
        assert repr(calculator) == "ConcreteMetricCalculator(name='TestMetric', score=None)"
        
        # After calculation
        context = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={"name": "test-model"}
        )
        calculator.calculate_score(context)
        
        assert repr(calculator) == "ConcreteMetricCalculator(name='TestMetric', score=0.5)"


class TestModelContext:
    """Test suite for ModelContext data class."""
    
    def test_model_context_creation(self):
        """Test ModelContext can be created with required fields."""
        context = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={"name": "test-model", "downloads": 1000}
        )
        
        assert context.model_url == "https://huggingface.co/test/model"
        assert context.model_info == {"name": "test-model", "downloads": 1000}
        assert context.dataset_url is None
        assert context.code_url is None
        assert context.local_repo_path is None
        assert context.huggingface_metadata is None
    
    def test_model_context_with_optional_fields(self):
        """Test ModelContext with all optional fields populated."""
        context = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={"name": "test-model"},
            dataset_url="https://huggingface.co/datasets/test-dataset",
            code_url="https://github.com/test/repo",
            local_repo_path="/tmp/test-repo",
            huggingface_metadata={"license": "mit", "tags": ["nlp"]}
        )
        
        assert context.dataset_url == "https://huggingface.co/datasets/test-dataset"
        assert context.code_url == "https://github.com/test/repo"
        assert context.local_repo_path == "/tmp/test-repo"
        assert context.huggingface_metadata == {"license": "mit", "tags": ["nlp"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
