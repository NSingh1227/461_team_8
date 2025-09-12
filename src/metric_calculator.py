from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelContext:
    """
    Context object containing model information passed to metric calculators.
    """
    model_url: str
    model_info: Dict[str, Any]
    dataset_url: Optional[str] = None
    code_url: Optional[str] = None
    local_repo_path: Optional[str] = None
    huggingface_metadata: Optional[Dict[str, Any]] = None

    # Phase 2 storage paths
    local_dataset_path: Optional[str] = None
    local_model_path: Optional[str] = None


class MetricCalculator(ABC):
    """
    Abstract base class for all metric calculators.
    """

    def __init__(self, name: str):
        self.name = name
        self._score: Optional[float] = None
        self._calculation_time_ms: Optional[int] = None

    @abstractmethod
    def calculate_score(self, context: ModelContext) -> float:
        pass

    def get_score(self) -> Optional[float]:
        return self._score

    def get_calculation_time(self) -> Optional[int]:
        return self._calculation_time_ms

    def _set_score(self, score: float, calculation_time_ms: int) -> None:
        if not (0 <= score <= 1):
            raise ValueError(f"Score must be between 0 and 1, got {score}")
        self._score = score
        self._calculation_time_ms = calculation_time_ms

    def reset(self) -> None:
        self._score = None
        self._calculation_time_ms = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', score={self._score})"
