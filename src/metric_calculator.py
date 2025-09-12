"""
Abstract base class for all metric calculators in the trustworthy model reuse system.
This follows the UML class diagram specification for MetricCalculator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelContext:
    """
    Context object containing model information passed to metric calculators.
    Based on UML diagram showing context flow between components.
    """
    model_url: str
    model_info: Dict[str, Any]
    dataset_url: Optional[str] = None
    code_url: Optional[str] = None
    local_repo_path: Optional[str] = None
    huggingface_metadata: Optional[Dict[str, Any]] = None


class MetricCalculator(ABC):
    """
    Abstract base class for all metric calculators.
    
    This class defines the interface that all concrete metric calculators must implement.
    Each calculator computes a specific metric (Size, License, RampUp, etc.) and returns
    a score between 0 and 1.
    
    Following the UML class diagram, this serves as the base for:
    - SizeCalculator
    - LicenseCalculator  
    - RampUpTimeCalculator
    - BusFactorCalculator
    - DatasetCodeScoreCalculator
    - DatasetQualityCalculator
    - CodeQualityCalculator
    - PerformanceClaimsCalculator
    """
    
    def __init__(self, name: str):
        """
        Initialize the metric calculator.
        
        Args:
            name: Human-readable name of the metric (e.g., "License", "Size")
        """
        self.name = name
        self._score: Optional[float] = None
        self._calculation_time_ms: Optional[int] = None
    
    @abstractmethod
    def calculate_score(self, context: ModelContext) -> float:
        """
        Calculate the metric score for the given model context.
        
        This is the core method that each concrete calculator must implement.
        The score must be in the range [0, 1] as specified in the requirements.
        
        Args:
            context: ModelContext containing all relevant model information
            
        Returns:
            float: Score between 0 and 1, where 1 is the best possible score
            
        Raises:
            MetricCalculationException: If calculation fails
        """
        pass
    
    def get_score(self) -> Optional[float]:
        """
        Get the last calculated score.
        
        Returns:
            Optional[float]: The score if calculated, None otherwise
        """
        return self._score
    
    def get_calculation_time(self) -> Optional[int]:
        """
        Get the time taken for the last calculation in milliseconds.
        
        Returns:
            Optional[int]: Calculation time in ms if available, None otherwise
        """
        return self._calculation_time_ms
    
    def _set_score(self, score: float, calculation_time_ms: int) -> None:
        """
        Internal method to set the calculated score and timing.
        
        Args:
            score: The calculated score (must be between 0 and 1)
            calculation_time_ms: Time taken for calculation in milliseconds
            
        Raises:
            ValueError: If score is not in valid range [0, 1]
        """
        if not (0 <= score <= 1):
            raise ValueError(f"Score must be between 0 and 1, got {score}")
        
        self._score = score
        self._calculation_time_ms = calculation_time_ms
    
    def reset(self) -> None:
        """
        Reset the calculator state for a new calculation.
        """
        self._score = None
        self._calculation_time_ms = None
    
    def __str__(self) -> str:
        """String representation of the calculator."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', score={self._score})"