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
from src.core.url_processor import fetch_github_metadata


class TestSizeCalculator(unittest.TestCase):
    def test_size_real_hf_model(self):
        # Use a tiny model to avoid large downloads
        context = ModelContext(
            model_url="https://huggingface.co/microsoft/DialoGPT-medium",
            model_info={"source": "huggingface", "type": "model"},
        )
        
        calc = SizeCalculator()
        score = calc.calculate_score(context)
        print(score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(calc.get_calculation_time())


class TestCodeQualityCalculator(unittest.TestCase):
    def test_code_quality_real_github_repo(self):
        # Use a popular public repo; metadata fetched externally for model_info
        code_url = "https://github.com/microsoft/DialoGPT"
        github_meta = fetch_github_metadata(code_url)
        model_info = {"source": "github", "type": "repository"}
        if isinstance(github_meta, dict):
            model_info.update({
                "stars": github_meta.get("stargazers_count"),
                "forks": github_meta.get("forks_count"),
                "description": github_meta.get("description"),
                "updated_at": github_meta.get("updated_at"),
            })
        context = ModelContext(
            model_url=code_url,
            code_url=code_url,
            model_info=model_info,
        )
        calc = CodeQualityCalculator()
        score = calc.calculate_score(context)
        print(score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(calc.get_calculation_time())


class TestPerformanceClaimsCalculator(unittest.TestCase):
    def test_performance_real_hf_readme(self):
        # Use tiny model with README
        context = ModelContext(
            model_url="https://huggingface.co/sshleifer/tiny-gpt2",
            model_info={"source": "huggingface", "type": "model"},
        )
        calc = PerformanceClaimsCalculator()
        score = calc.calculate_score(context)
        print("PerformanceClaimsCalculator score: ", score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()


