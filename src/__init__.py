"""
Trustworthy Model Reuse CLI Package
ACME Corporation - Phase 1 Implementation

This package implements a CLI tool for evaluating the trustworthiness of machine learning models
from Hugging Face and GitHub repositories based on multiple metrics.
"""

__version__ = "1.0.0"
__author__ = "Team 8 - Purdue ECE 46100"

# Core components following UML class diagram
from .metric_calculator import MetricCalculator, ModelContext
from .results_storage import ResultsStorage, MetricResult, ModelResult
from .exceptions import (
    TrustworthyModelException,
    MetricCalculationException, 
    APIRateLimitException,
    InvalidURLException,
    ConfigurationException
)

__all__ = [
    "MetricCalculator",
    "ModelContext", 
    "ResultsStorage",
    "MetricResult",
    "ModelResult",
    "TrustworthyModelException",
    "MetricCalculationException",
    "APIRateLimitException", 
    "InvalidURLException",
    "ConfigurationException"
]