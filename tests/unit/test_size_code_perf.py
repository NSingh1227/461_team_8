#!/usr/bin/env python3
"""
Unit tests for Size, CodeQuality, and PerformanceClaims calculators.
"""

import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.metrics.base import ModelContext
from src.metrics.size_calculator import SizeCalculator
from src.metrics.code_quality_calculator import CodeQualityCalculator
from src.metrics.performance_claims_calculator import PerformanceClaimsCalculator


class TestSizeCalculator(unittest.TestCase):
    def test_size_from_hf_siblings(self):
        context = ModelContext(
            model_url="https://huggingface.co/org/model",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata={
                "siblings": [
                    {"rfilename": "pytorch_model.bin", "size": 50 * 1024 * 1024},
                    {"rfilename": "config.json", "size": 1024},
                    {"rfilename": "tokenizer.json", "size": 2048},
                ]
            },
        )
        calc = SizeCalculator()
        score = calc.calculate_score(context)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(calc.get_calculation_time())

    def test_size_unknown_metadata(self):
        context = ModelContext(
            model_url="https://huggingface.co/org/model",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata=None,
        )
        calc = SizeCalculator()
        score = calc.calculate_score(context)
        self.assertEqual(score, 0.5)


class TestCodeQualityCalculator(unittest.TestCase):
    def test_code_quality_github_metadata(self):
        context = ModelContext(
            model_url="https://github.com/user/repo",
            model_info={
                "source": "github",
                "type": "repository",
                "stars": 1500,
                "forks": 600,
                "language": "Python",
                "description": "A test repo",
                "updated_at": "2025-01-15T00:00:00Z",
            },
        )
        calc = CodeQualityCalculator()
        score = calc.calculate_score(context)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(calc.get_calculation_time())

    def test_code_quality_fallback_llm(self):
        context = ModelContext(
            model_url="https://huggingface.co/org/model",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata={"cardData": {"some": "field"}},
        )
        calc = CodeQualityCalculator()
        score = calc.calculate_score(context)
        self.assertAlmostEqual(score, 0.6, places=3)


class TestPerformanceClaimsCalculator(unittest.TestCase):
    def test_performance_from_metadata(self):
        context = ModelContext(
            model_url="https://huggingface.co/org/model",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata={
                "cardData": {"metrics": [{"name": "accuracy", "value": 0.9}]},
                "tags": ["benchmark:glue"],
            },
        )
        calc = PerformanceClaimsCalculator()
        score = calc.calculate_score(context)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_performance_fallback_llm(self):
        context = ModelContext(
            model_url="https://huggingface.co/org/model",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata={"cardData": {"no": "signals"}},
        )
        calc = PerformanceClaimsCalculator()
        score = calc.calculate_score(context)
        self.assertAlmostEqual(score, 0.5, places=3)


if __name__ == '__main__':
    unittest.main()


