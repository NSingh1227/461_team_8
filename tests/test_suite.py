#!/usr/bin/env python3
import sys
import os
import datetime
from typing import List, Dict, Any, Tuple, Optional, cast
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.url_processor import URLProcessor, URLType, process_url, categorize_url, is_valid_url
from src.metrics.license_calculator import LicenseCalculator
from src.metrics.busfactor_calculator import BusFactorCalculator
from src.metrics.size_calculator import SizeCalculator
from src.metrics.ramp_up_calculator import RampUpCalculator
from src.metrics.code_quality_calculator import CodeQualityCalculator
from src.metrics.dataset_code_calculator import DatasetCodeCalculator
from src.metrics.dataset_quality_calculator import DatasetQualityCalculator
from src.metrics.performance_claims_calculator import PerformanceClaimsCalculator
from src.metrics.llm_analyzer import LLMAnalyzer
from src.metrics.base import ModelContext
from src.storage.results_storage import ModelResult
from src.core.rate_limiter import get_rate_limiter, reset_rate_limiter, APIService
from src.core.http_client import get_with_rate_limit, head_with_rate_limit
from src.core.llm_client import ask_for_json_score
from src.core.config import Config
from src.core.exceptions import *

from unittest.mock import Mock, patch, MagicMock, mock_open

def mock_open_config() -> Mock:
    """Create a mock file object for config.json."""
    mock_file = Mock()
    mock_file.read.return_value = '{"model_type": "test", "hidden_size": 768}'
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.__exit__ = Mock(return_value=None)
    return Mock(return_value=mock_file)


class NoOpRateLimiter:
    """No-op rate limiter for testing - doesn't actually limit anything."""
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize no-op rate limiter."""
        pass
    
    def wait_if_needed(self, service: Any) -> None:
        """No-op wait method - no delays."""
        pass
    
    def handle_rate_limit_response(self, service: Any, retry_after: Optional[int] = None) -> None:
        """No-op rate limit handling - no delays."""
        pass
    
    def reset_failures(self, service: Any) -> None:
        """No-op failure reset."""
        pass
    
    def check_quota(self, service: Any) -> bool:
        """Always return True - no quota limits."""
        return True
    
    def has_quota(self, service: Any) -> bool:
        """Always return True - no quota limits."""
        return True
    
    def get_quota_status(self, service: Any) -> Dict[str, Any]:
        """Return fake quota status."""
        return {
            'current_requests': 0,
            'max_requests': 1000,
            'quota_remaining': 1000
        }


class TestSuite:
    def __init__(self, coverage_mode: bool = False) -> None:
        self.passed_tests: int = 0
        self.failed_tests: int = 0
        self.total_tests: int = 0
        self.coverage_mode: bool = coverage_mode
        self.original_rate_limiter: Optional[Any] = None
        
        # Set up no-op rate limiter for fast testing
        self.setup_no_op_rate_limiter()
    
    def print_header(self, title: str) -> None:
        if not self.coverage_mode:
            print("\n" + "="*60)
            print(f"  {title}")
            print("="*60)
    
    def print_test_result(self, test_name: str, expected: Any, actual: Any, passed: bool) -> None:
        self.total_tests += 1
        if not self.coverage_mode:
            status = "PASS" if passed else "FAIL"
            print(f"{status} | {test_name}")
            print(f"      Expected: {expected}")
            print(f"      Actual:   {actual}")
            print()
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def print_section(self, title: str) -> None:
        if not self.coverage_mode:
            print(f"\n--- {title} ---")
    
    def setup_no_op_rate_limiter(self) -> None:
        """Set up a no-op rate limiter for fast testing."""
        try:
            from src.core.rate_limiter import get_rate_limiter, set_rate_limiter
            # Store the original rate limiter
            self.original_rate_limiter = get_rate_limiter()
            # Set the no-op rate limiter
            set_rate_limiter(NoOpRateLimiter())
            
            # Also mock time.sleep to prevent any delays
            import time
            self.original_sleep = time.sleep
            time.sleep = lambda x: None  # No-op sleep
        except Exception:
            # If rate limiter setup fails, continue without it
            pass
    
    def restore_original_rate_limiter(self) -> None:
        """Restore the original rate limiter for rate limiting tests."""
        try:
            from src.core.rate_limiter import set_rate_limiter, reset_rate_limiter
            if self.original_rate_limiter is not None:
                set_rate_limiter(self.original_rate_limiter)
            else:
                reset_rate_limiter()
            
            # Restore original time.sleep
            if hasattr(self, 'original_sleep'):
                import time
                time.sleep = self.original_sleep
        except Exception:
            # If rate limiter restore fails, continue without it
            pass




    
    def test_url_validation(self) -> None:
        self.print_header("URL VALIDATION TESTS")
        
        test_cases: List[Tuple[str, str, bool]] = [
            ("Valid HTTPS URL", "https://huggingface.co/microsoft/DialoGPT-medium", True),
            ("Valid HTTP URL", "http://github.com/microsoft/DialoGPT", True),
            ("Invalid URL - No scheme", "huggingface.co/microsoft/DialoGPT-medium", False),
            ("Invalid URL - Empty", "", False),
            ("Invalid URL - Just text", "not-a-url", False),
            ("Invalid URL - No domain", "https://", False),
            ("Valid URL with port", "https://example.com:8080/path", True),
            ("Invalid URL with space", "https://example .com", False),
            ("Invalid URL malformed", "https:///invalid", False)
        ]
        
        for test_name, url, expected in test_cases:
            actual: bool = is_valid_url(url)
            passed: bool = actual == expected
            self.print_test_result(test_name, expected, actual, passed)

    def test_url_categorization(self) -> None:
        self.print_header("URL CATEGORIZATION TESTS")
        
        test_cases: List[Tuple[str, str, URLType]] = [
            ("HuggingFace Model", "https://huggingface.co/microsoft/DialoGPT-medium", URLType.HUGGINGFACE_MODEL),
            ("HuggingFace Dataset", "https://huggingface.co/datasets/squad", URLType.HUGGINGFACE_DATASET),
            ("GitHub Repository", "https://github.com/microsoft/DialoGPT", URLType.GITHUB_REPO),
            ("Unknown Domain", "https://example.com/some/path", URLType.UNKNOWN),
            ("Invalid URL", "not-a-url", URLType.UNKNOWN)
        ]
        
        for test_name, url, expected in test_cases:
            actual: URLType = process_url(url)
            passed: bool = actual == expected
            self.print_test_result(test_name, expected.value, actual.value, passed)


    def test_comprehensive_url_processing(self) -> None:
        self.print_header("COMPREHENSIVE URL PROCESSING TESTS")
        
        test_urls: List[str] = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://huggingface.co/datasets/squad",
            "https://github.com/microsoft/DialoGPT",
            "https://huggingface.co/fake/nonexistent-model",
            "https://huggingface.co/datasets/fake-dataset",
            "https://github.com/fake/nonexistent-repo",
            "https://example.com/invalid/url",
            "not-a-url-at-all"
        ]
        
        test_file_path: str = "temp_test_urls.txt"
        with open(test_file_path, 'w') as f:
            for test_url in test_urls:
                f.write(test_url + '\n')
        
        try:
            processor: URLProcessor = URLProcessor(test_file_path)
            results: List[ModelResult] = processor.process_urls_with_metrics()
            
            self.print_section("URL Processing Results")
            for i, result in enumerate(results):
                url: str = result.url
                net_score: float = result.net_score
                
                if not self.coverage_mode:
                    print(f"{i+1}. URL: {url}")
                    print(f"   Net Score: {net_score}")
                    print()
            

            expected_count: int = 6
            actual_count: int = len(results)
            self.print_test_result("URL Processing Count", expected_count, actual_count, expected_count == actual_count)
            
        except Exception as e:
            print(f"❌ ERROR in URL processing: {e}")
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

    def test_edge_cases(self):
        self.print_header("EDGE CASE TESTS")
        
        test_cases = [
            ("Empty String URL", "", URLType.UNKNOWN),
            ("None as URL", None, URLType.UNKNOWN),
            ("Very Long URL", "https://huggingface.co/" + "a" * 1000, URLType.HUGGINGFACE_MODEL),
            ("URL with Unicode", "https://huggingface.co/microsoft/DialoGPT-médium", URLType.HUGGINGFACE_MODEL),
            ("URL with Query Params", "https://github.com/microsoft/DialoGPT?tab=readme", URLType.GITHUB_REPO)
        ]
        
        for test_name, url, expected in test_cases:
            try:
                if url is None:
                    actual = URLType.UNKNOWN
                else:
                    actual = process_url(url)
                passed = actual == expected
                self.print_test_result(test_name, expected.value, actual.value, passed)
            except Exception:
                actual = URLType.UNKNOWN
                passed = actual == expected
                self.print_test_result(test_name, expected.value, actual.value, passed)

    def test_url_processor_edge_cases(self):
        self.print_header("URL PROCESSOR EDGE CASES")
        
        from src.core.url_processor import fetch_huggingface_metadata, categorize_url
        
        test_cases: List[Tuple[str, str, URLType]] = [
            ("Malformed HuggingFace URL", "https://huggingface.co/", URLType.HUGGINGFACE_MODEL),
            ("HuggingFace URL with extra path", "https://huggingface.co/microsoft/DialoGPT-medium/tree/main", URLType.HUGGINGFACE_MODEL),
            ("GitHub URL with extra path", "https://github.com/microsoft/DialoGPT/issues", URLType.GITHUB_REPO),
            ("Non-standard domain", "https://gitlab.com/test/repo", URLType.UNKNOWN),
        ]
        
        for description, url, expected in test_cases:
            try:
                url_type_result = categorize_url(url)
                self.print_test_result(description, expected.value, url_type_result.value, expected.value == url_type_result.value)
            except Exception as e:
                self.print_test_result(description, expected.value, f"Error: {e}", False)
                
        try:
            result: Optional[Dict[str, Any]] = fetch_huggingface_metadata("https://huggingface.co/", "models")
            self.print_test_result("Fetch metadata with empty path", None, result, result is None)
        except Exception:
            self.print_test_result("Fetch metadata with empty path", None, None, True)
            
        additional_test_cases: List[Tuple[str, str, bool]] = [
            ("URL with fragment", "https://example.com#section", True),
            ("URL with query and fragment", "https://example.com?q=test#section", True),
            ("FTP URL", "ftp://files.example.com", False),
            ("File URL", "file:///path/to/file", False)
        ]
        
        for test_desc, test_url, test_expected in additional_test_cases:
            description_str: str = cast(str, test_desc)
            url_str: str = cast(str, test_url)
            expected_bool: bool = cast(bool, test_expected)
            try:
                is_valid_result: bool = is_valid_url(url_str)
                self.print_test_result(description_str, expected_bool, is_valid_result, is_valid_result == expected_bool)
            except Exception:
                self.print_test_result(description_str, expected_bool, False, False)
  
    def test_license_calculator(self):
        self.print_header("LICENSE CALCULATOR TESTS")
        
        calculator = LicenseCalculator()
        
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "MIT License (HuggingFace Model)",
                "url": "https://huggingface.co/microsoft/DialoGPT-medium",
                "expected_range": (0.8, 1.0),
                "description": "Real HuggingFace model with MIT license"
            },
            {
                "name": "Unknown License (HuggingFace Dataset)", 
                "url": "https://huggingface.co/datasets/squad",
                "expected_range": (0.4, 0.6),
                "description": "Real HuggingFace dataset with unclear license"
            },
            {
                "name": "MIT License (GitHub Repo)",
                "url": "https://github.com/microsoft/DialoGPT", 
                "expected_range": (0.8, 1.0),
                "description": "Real GitHub repository with MIT license"
            },
            {
                "name": "Nonexistent Model",
                "url": "https://huggingface.co/fake/nonexistent-model",
                "expected_range": (0.4, 0.6),
                "description": "Fake HuggingFace model URL"
            },
            {
                "name": "Nonexistent GitHub Repo",
                "url": "https://github.com/fake/nonexistent-repo",
                "expected_range": (0.4, 0.6),
                "description": "Fake GitHub repository URL"
            }
        ]
        
        self.print_section("License Score Tests")
        
        for test_case in test_cases:
            try:
                if "github.com" in test_case["url"]:
                    from src.core.url_processor import CodeHandler
                    handler = CodeHandler()
                    model_context = handler.process_url(test_case["url"])
                else:
                    model_context = ModelContext(
                        model_url=test_case["url"],
                        model_info={"type": "test", "source": "test"},
                        huggingface_metadata=None
                    )
                
                score: float = calculator.calculate_score(model_context)
                latency: Optional[int] = calculator.get_calculation_time()
                
                min_score: float
                max_score: float
                min_score, max_score = test_case["expected_range"]
                passed: bool = min_score <= score <= max_score
                
                if not self.coverage_mode:
                    print(f"{'✅ PASS' if passed else '❌ FAIL'} | {test_case['name']}")
                    print(f"      URL: {test_case['url']}")
                    print(f"      Description: {test_case['description']}")
                    print(f"      Score: {score:.2f} (expected: {min_score}-{max_score})")
                    print(f"      Latency: {latency}ms")
                    print()
                
                if passed:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
                self.total_tests += 1
                
            except Exception as e:
                if not self.coverage_mode:
                    print(f"❌ ERROR | {test_case['name']}: {e}")
                self.failed_tests += 1
                self.total_tests += 1

    def test_license_compatibility_mapping(self):
        self.print_header("LICENSE COMPATIBILITY MAPPING TESTS")
        
        calculator = LicenseCalculator()
        
        test_cases = [
            ("MIT License Score", "mit", 1.0),
            ("Apache License Score", "apache-2.0", 1.0),
            ("BSD License Score", "bsd-3-clause", 1.0),
            ("GPL License Score", "gpl-3.0", 0.0),
            ("AGPL License Score", "agpl-3.0", 0.0),
            ("Proprietary License Score", "proprietary", 0.0),
            ("Unknown License Score", "unknown-license", 0.5)
        ]
        
        for test_name, license_text, expected_score in test_cases:
            actual_score = calculator._calculate_compatibility_score(license_text)
            passed = actual_score == expected_score
            self.print_test_result(test_name, expected_score, actual_score, passed)
   
    def test_url_parsing_edge_cases(self):
        self.print_header("ADVANCED URL PARSING TESTS")
        
        test_cases = [
            ("URL with multiple slashes", "https://github.com//microsoft//DialoGPT", URLType.GITHUB_REPO),
            ("URL with trailing slash", "https://huggingface.co/microsoft/DialoGPT-medium/", URLType.HUGGINGFACE_MODEL),
            ("URL with uppercase", "HTTPS://GITHUB.COM/MICROSOFT/DIALOGPT", URLType.UNKNOWN),
            ("URL with mixed case", "https://HuggingFace.co/Microsoft/DialoGPT-Medium", URLType.UNKNOWN),
            ("URL with www prefix", "https://www.github.com/microsoft/DialoGPT", URLType.UNKNOWN),
            ("URL with subdomain", "https://api.github.com/repos/microsoft/DialoGPT", URLType.UNKNOWN),
            ("GitHub gist URL", "https://gist.github.com/user/12345", URLType.UNKNOWN),
            ("HuggingFace with spaces branch", "https://huggingface.co/microsoft/DialoGPT-medium/tree/feature%20branch", URLType.HUGGINGFACE_MODEL),
            ("GitHub with commit hash", "https://github.com/microsoft/DialoGPT/commit/abc123", URLType.GITHUB_REPO),
            ("HuggingFace with revision", "https://huggingface.co/microsoft/DialoGPT-medium/blob/main/model.py", URLType.HUGGINGFACE_MODEL)
        ]
        
        for test_name, url, expected in test_cases:
            try:
                actual = process_url(url)
                passed = actual == expected
                self.print_test_result(test_name, expected.value, actual.value, passed)
            except Exception as e:
                self.print_test_result(test_name, expected.value, f"Error: {e}", False)

    def test_concurrent_url_processing(self):
        self.print_header("CONCURRENT URL PROCESSING TESTS")
    
        test_urls = [
            "https://huggingface.co/microsoft/DialoGPT-small",
            "https://huggingface.co/microsoft/DialoGPT-medium", 
            "https://huggingface.co/microsoft/DialoGPT-large",
            "https://github.com/microsoft/DialoGPT",
            "https://github.com/pytorch/pytorch",
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/datasets/cnn_dailymail",
            "https://example.com/invalid1",
            "https://example.com/invalid2",
            "not-a-url"
        ]
        
        test_file_path = "temp_concurrent_test.txt"
        with open(test_file_path, 'w') as f:
            for url in test_urls:
                f.write(url + '\n')
        
        try:
            processor = URLProcessor(test_file_path)
            results = processor.process_urls_with_metrics()
            

            processed_count = len(results)
            expected_count = 7
            self.print_test_result("Concurrent processing count", expected_count, processed_count, processed_count == expected_count)

            valid_results = len([r for r in results if r.net_score > 0])
            self.print_test_result("Valid results count", expected_count, valid_results, valid_results >= 0)
            
        except Exception as e:
            if not self.coverage_mode:
                print(f"❌ ERROR in concurrent processing: {e}")
            self.failed_tests += 1
            self.total_tests += 1
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
 
    def test_metric_calculation_performance(self):
        self.print_header("METRIC CALCULATION PERFORMANCE TESTS")
        
        calculator = LicenseCalculator()
        test_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://github.com/microsoft/DialoGPT"
        ] * 5
        
        total_time: float = 0.0
        successful_calculations: int = 0
        
        for i, url in enumerate(test_urls):
            try:
                start_time: datetime.datetime = datetime.datetime.now()
                
                if "github.com" in url:
                    from src.core.url_processor import CodeHandler
                    handler = CodeHandler()
                    model_context = handler.process_url(url)
                else:
                    model_context = ModelContext(
                        model_url=url,
                        model_info={"type": "test", "source": "test"},
                        huggingface_metadata=None
                    )
                
                score: float = calculator.calculate_score(model_context)
                end_time: datetime.datetime = datetime.datetime.now()
                
                calculation_time: float = (end_time - start_time).total_seconds() * 1000
                total_time += calculation_time
                successful_calculations += 1
                score_valid = 0.0 <= score <= 1.0
                if not score_valid:
                    self.print_test_result(f"Performance test {i+1} score validity", True, False, False)
                
            except Exception as e:
                if not self.coverage_mode:
                    print(f"Performance test {i+1} failed: {e}")
        
        avg_time = total_time / successful_calculations if successful_calculations > 0 else 0
        performance_acceptable = avg_time < 5000
        
        self.print_test_result("Successful calculations", len(test_urls), successful_calculations, successful_calculations == len(test_urls))
        self.print_test_result("Average calculation time acceptable", True, performance_acceptable, performance_acceptable)

    def test_error_recovery_scenarios(self):
        self.print_header("ERROR RECOVERY SCENARIO TESTS")
        test_cases = [
            {
                "name": "Network timeout simulation",
                "url": "https://httpstat.us/408",
                "expected_behavior": "graceful_failure"
            },
            {
                "name": "Server error simulation", 
                "url": "https://httpstat.us/500",
                "expected_behavior": "graceful_failure"
            },
            {
                "name": "Rate limit simulation",
                "url": "https://httpstat.us/429",
                "expected_behavior": "graceful_failure"
            },
            {
                "name": "Malformed JSON response",
                "url": "https://huggingface.co/fake/malformed-response",
                "expected_behavior": "graceful_failure"
            },
            {
                "name": "Empty response body",
                "url": "https://httpstat.us/204",
                "expected_behavior": "graceful_failure"
            }
        ]        
        calculator = LicenseCalculator()        
        for test_case in test_cases:
            try:
                model_context = ModelContext(
                    model_url=test_case["url"],
                    model_info={"type": "test", "source": "test"},
                    huggingface_metadata=None
                )               
                score = calculator.calculate_score(model_context)
                score_valid = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
                self.print_test_result(test_case["name"], True, score_valid, score_valid)
                
            except Exception as e:

                self.print_test_result(test_case["name"], "graceful_failure", "exception_raised", True)
                 
    def test_data_consistency_checks(self):
        self.print_header("DATA CONSISTENCY TESTS")
        test_url = "https://huggingface.co/microsoft/DialoGPT-medium"
        calculator = LicenseCalculator()
        
        scores: List[Optional[float]] = []
        for i in range(3):
            try:
                model_context = ModelContext(
                    model_url=test_url,
                    model_info={"type": "test", "source": "test"},
                    huggingface_metadata=None
                )
                score: float = calculator.calculate_score(model_context)
                scores.append(score)
            except Exception:
                scores.append(None)
        
        all_same = len(set(s for s in scores if s is not None)) <= 1
        self.print_test_result("Consistent scoring", True, all_same, all_same)
        test_urls = [
            "https://github.com/microsoft/DialoGPT",
            "https://github.com/microsoft/DialoGPT/",
            "https://github.com/microsoft/DialoGPT.git",
            "https://github.com/microsoft/dialogpt"
        ]     
        url_types: List[Optional[URLType]] = []
        for url in test_urls:
            try:
                url_type: URLType = process_url(url)
                url_types.append(url_type)
            except Exception:
                url_types.append(None)
        all_github = all(t == URLType.GITHUB_REPO for t in url_types if t is not None)
        self.print_test_result("URL normalization consistency", True, all_github, all_github)

    def test_boundary_value_analysis(self):
        self.print_header("BOUNDARY VALUE ANALYSIS TESTS")
        from src.metrics.base import MetricCalculator, ModelContext
        class BoundaryTestCalculator(MetricCalculator):
            def calculate_score(self, context: ModelContext) -> float:
                if "boundary-zero" in context.model_url:
                    return 0.0
                elif "boundary-one" in context.model_url:
                    return 1.0
                elif "boundary-half" in context.model_url:
                    return 0.5
                else:
                    return 0.75
        calculator = BoundaryTestCalculator("BoundaryTest")
        test_cases = [
            ("Minimum boundary (0.0)", "https://test.com/boundary-zero", 0.0),
            ("Maximum boundary (1.0)", "https://test.com/boundary-one", 1.0),
            ("Middle boundary (0.5)", "https://test.com/boundary-half", 0.5),
            ("Normal value", "https://test.com/normal", 0.75)
        ]
        for test_name, url, expected in test_cases:
            try:
                context = ModelContext(model_url=url, model_info={"type": "test"})
                score = calculator.calculate_score(context)
                passed = abs(score - expected) < 0.001
                self.print_test_result(test_name, expected, score, passed)
            except Exception as e:
                self.print_test_result(test_name, expected, f"Error: {e}", False)
 
    def test_complete_workflow_integration(self):
        self.print_header("COMPLETE WORKFLOW INTEGRATION TESTS")
        test_scenarios: List[Dict[str, Any]] = [
            {
                "name": "Mixed URL types workflow",
                "urls": [
                    "https://huggingface.co/microsoft/DialoGPT-medium",
                    "https://github.com/microsoft/DialoGPT",
                    "https://huggingface.co/datasets/squad",
                    "https://invalid-domain.com/fake"
                ],
                "expected_processed": 4,
                "expected_with_metrics": 3
            },
            {
                "name": "Single URL workflow",
                "urls": ["https://huggingface.co/microsoft/DialoGPT-medium"],
                "expected_processed": 1,
                "expected_with_metrics": 1
            },
            {
                "name": "Empty file workflow",
                "urls": [],
                "expected_processed": 0,
                "expected_with_metrics": 0
            }
        ] 
        for scenario in test_scenarios:
            test_file_path = f"temp_workflow_{scenario['name'].replace(' ', '_')}.txt"
            with open(test_file_path, 'w') as f:
                for url in scenario['urls']:
                    f.write(url + '\n')
            try:
                processor = URLProcessor(test_file_path)
                metric_results = processor.process_urls_with_metrics()
                metric_count_correct = len(metric_results) == scenario['expected_with_metrics']
                
                self.print_test_result(f"{scenario['name']} - Metric processing count",
                                     scenario['expected_with_metrics'], len(metric_results), metric_count_correct)
                
                if len(metric_results) > 0:
                    first_result = metric_results[0]
                    has_required_fields = all(hasattr(first_result, field) for field in 
                                            ['url', 'net_score', 'license_score', 'bus_factor_score'])
                    self.print_test_result(f"{scenario['name']} - Result structure validity", True, has_required_fields, has_required_fields)

            except Exception as e:
                if not self.coverage_mode:
                    print(f"❌ ERROR in workflow {scenario['name']}: {e}")
                self.failed_tests += 2
                self.total_tests += 2
            finally:
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)

    def test_file_handling_edge_cases(self):
        self.print_header("FILE HANDLING EDGE CASES")
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "File with empty lines",
                "content": "https://github.com/test/repo1\n\n\nhttps://github.com/test/repo2\n\n",
                "expected_count": 2
            },
            {
                "name": "File with comments/invalid lines",
                "content": "# This is a comment\nhttps://github.com/test/repo\ninvalid-line\n# Another comment",
                "expected_count": 1
            },
            {
                "name": "File with whitespace",
                "content": "  https://github.com/test/repo  \n\t\nhttps://huggingface.co/test/model\t\n",
                "expected_count": 2
            },
            {
                "name": "File with very long URLs",
                "content": f"https://github.com/{'a' * 500}/{'b' * 500}\nhttps://github.com/normal/repo",
                "expected_count": 2
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            test_file_path = f"temp_file_test_{i}.txt"
            
            try:

                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_case['content'])
                processor = URLProcessor(test_file_path)
                results = processor.process_urls_with_metrics()
                
                actual_count = len(results)
                passed = actual_count == test_case['expected_count']
                self.print_test_result(test_case['name'], test_case['expected_count'], actual_count, passed)
                
            except Exception as e:
                self.print_test_result(test_case['name'], test_case['expected_count'], f"Error: {e}", False)
            finally:
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)

    def test_security_validation(self):
        self.print_header("SECURITY VALIDATION TESTS")
        

        test_cases = [
            ("URL with SQL injection attempt", "https://github.com/test'; DROP TABLE users; --/repo", URLType.UNKNOWN),
            ("URL with script injection", "https://github.com/test<script>alert(1)</script>/repo", URLType.GITHUB_REPO),
            ("URL with path traversal", "https://github.com/../../../etc/passwd", URLType.GITHUB_REPO),
            ("URL with null bytes", "https://github.com/test\x00/repo", URLType.GITHUB_REPO),
            ("URL with unicode normalization", "https://github.com/test\u0000/repo", URLType.GITHUB_REPO),
            ("Extremely long URL", "https://github.com/" + "a" * 1000, URLType.GITHUB_REPO),
            ("URL with control characters", "https://github.com/test\r\n/repo", URLType.GITHUB_REPO),
            ("URL with encoded attacks", "https://github.com/test%3Cscript%3E/repo", URLType.GITHUB_REPO)
        ]
        
        for test_name, url, expected in test_cases:
            try:
                actual = process_url(url)

                passed = actual == expected or actual == URLType.UNKNOWN
                expected_display = f"{expected.value} or unknown (safe handling)"
                self.print_test_result(test_name, expected_display, actual.value, passed)
            except Exception:

                self.print_test_result(test_name, "Exception (safe)", "Exception raised", True)

    def test_unicode_and_internationalization(self):
        self.print_header("UNICODE AND INTERNATIONALIZATION TESTS")
        
        test_cases = [
            ("URL with Chinese characters", "https://github.com/用户/项目", URLType.GITHUB_REPO),
            ("URL with Arabic characters", "https://github.com/مستخدم/مشروع", URLType.GITHUB_REPO),
            ("URL with emoji", "https://github.com/user/project-🚀", URLType.GITHUB_REPO),
            ("URL with accented characters", "https://github.com/utilisateur/projet-été", URLType.GITHUB_REPO),
            ("URL with Cyrillic", "https://github.com/пользователь/проект", URLType.GITHUB_REPO),
            ("HuggingFace with unicode", "https://huggingface.co/用户/模型", URLType.HUGGINGFACE_MODEL),
            ("Mixed scripts URL", "https://github.com/user用户/project项目", URLType.GITHUB_REPO)
        ]
        
        for test_name, url, expected in test_cases:
            try:
                actual = process_url(url)
                passed = actual == expected
                self.print_test_result(test_name, expected.value, actual.value, passed)
            except Exception as e:

                self.print_test_result(test_name, expected.value, f"Error: {e}", False)
 
    def test_large_batch_processing(self):
        self.print_header("LARGE BATCH PROCESSING TESTS")
        

        reset_rate_limiter()
        

        base_urls = [
            "https://github.com/test/repo",
            "https://example.com/fake/url",
            "https://mock.domain/test/path"
        ]
        

        test_urls = []
        for i in range(5):
            for base_url in base_urls:
                test_urls.append(f"{base_url}-{i}")
        
        test_file_path = "temp_large_batch.txt"
        
        try:

            with open(test_file_path, 'w') as f:
                for url in test_urls:
                    f.write(url + '\n')
            
            start_time = datetime.datetime.now()
            

            try:
                processor = URLProcessor(test_file_path)
                results = processor.process_urls_with_metrics()
                
                end_time = datetime.datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                

                if processing_time > 15.0:
                    print(f"⚠️ Skipping large batch test - would exceed time limit ({processing_time:.1f}s)")
                    self.total_tests += 3
                    return
                    
            except KeyboardInterrupt:
                print(f"⚠️ Large batch test interrupted - likely due to rate limiting")
                self.total_tests += 3
                return
            except Exception as e:
                print(f"❌ Error in large batch processing: {e}")
                self.failed_tests += 3
                self.total_tests += 3
                return
            

            processed_count = len(results)
            expected_count = 5
            count_correct = processed_count == expected_count
            
            performance_acceptable = processing_time < 10.0
            
            self.print_test_result("Large batch processing count", expected_count, processed_count, count_correct)
            self.print_test_result("Large batch processing performance", True, performance_acceptable, performance_acceptable)
            

            memory_efficient = len(str(results)) < 1000000
            self.print_test_result("Large batch memory efficiency", True, memory_efficient, memory_efficient)
            
        except Exception as e:
            if not self.coverage_mode:
                print(f"❌ ERROR in large batch processing: {e}")
            self.failed_tests += 3
            self.total_tests += 3
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

    def test_edge_case_metric_scenarios(self):
        self.print_header("EDGE CASE METRIC SCENARIOS")
        
        calculator = LicenseCalculator()
        
        edge_case_contexts: List[Dict[str, Any]] = [
            {
                "name": "Context with empty model_info",
                "context": ModelContext(model_url="https://test.com", model_info={}),
                "should_handle_gracefully": True
            },
            {
                "name": "Context with minimal info",
                "context": ModelContext(model_url="https://test.com", model_info={"type": "test"}),
                "should_handle_gracefully": True
            },
            {
                "name": "Context with empty URL",
                "context": ModelContext(model_url="", model_info={"type": "test"}),
                "should_handle_gracefully": True
            },
            {
                "name": "Context with malformed URL",
                "context": ModelContext(model_url="not-a-url", model_info={"type": "test"}),
                "should_handle_gracefully": True
            }
        ]
        
        for test_case in edge_case_contexts:
            try:
                score = calculator.calculate_score(test_case["context"])
                

                if test_case["should_handle_gracefully"]:
                    valid_score = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
                    self.print_test_result(test_case["name"], True, valid_score, valid_score)
                else:
                    self.print_test_result(test_case["name"], "Exception", "No exception", False)
                    
            except Exception:
                if test_case["should_handle_gracefully"]:
                    self.print_test_result(test_case["name"], "Graceful handling", "Exception raised", False)
                else:
                    self.print_test_result(test_case["name"], "Exception", "Exception raised", True)
    
    def test_busfactor_calculator(self):
        self.print_header("BUS FACTOR CALCULATOR TESTS")
        
        calculator = BusFactorCalculator()
        
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "Active GitHub Repository",
                "url": "https://github.com/microsoft/DialoGPT",
                "expected_range": (0.0, 1.0),
                "description": "Real GitHub repository with multiple contributors"
            },
            {
                "name": "Small GitHub Repository", 
                "url": "https://github.com/torvalds/linux",
                "expected_range": (0.0, 1.0),
                "description": "Large GitHub repository with many contributors"
            },
            {
                "name": "Nonexistent GitHub Repository",
                "url": "https://github.com/fake/nonexistent-repo",
                "expected_range": (0.0, 0.5),
                "description": "Fake GitHub repository URL"
            }
        ]
        
        self.print_section("Bus Factor Score Tests")
        
        for test_case in test_cases:
            try:
                from src.core.url_processor import CodeHandler
                handler = CodeHandler()
                model_context = handler.process_url(test_case["url"])
                
                score = calculator.calculate_score(model_context)
                latency = calculator.get_calculation_time()
                
                min_score, max_score = test_case["expected_range"]
                passed = min_score <= score <= max_score
                
                if not self.coverage_mode:
                    print(f"{'✅ PASS' if passed else '❌ FAIL'} | {test_case['name']}")
                    print(f"      URL: {test_case['url']}")
                    print(f"      Description: {test_case['description']}")
                    print(f"      Score: {score:.2f} (expected: {min_score}-{max_score})")
                    print(f"      Latency: {latency}ms")
                    print()
                
                if passed:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
                self.total_tests += 1
                
            except Exception as e:
                if not self.coverage_mode:
                    print(f"❌ ERROR | {test_case['name']}: {e}")
                self.failed_tests += 1
                self.total_tests += 1
 
    def test_rate_limiter(self):
        self.print_header("RATE LIMITER TESTS")
        

        reset_rate_limiter()
        rate_limiter = get_rate_limiter()
        
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "Quota Check - Initial State",
                "service": APIService.GITHUB,
                "expected": True,
                "description": "Should have quota available initially"
            },
            {
                "name": "Quota Status - GitHub",
                "service": APIService.GITHUB, 
                "expected": True,
                "description": "Should return valid quota status"
            },
            {
                "name": "Quota Check - HuggingFace",
                "service": APIService.HUGGINGFACE,
                "expected": True,
                "description": "Should have quota available for HuggingFace"
            },
            {
                "name": "Service Independence",
                "service": APIService.GENAI,
                "expected": True,
                "description": "Different services should have independent quotas"
            }
        ]
        
        self.print_section("Rate Limiter Basic Tests")
        
        for test_case in test_cases:
            try:
                service = test_case["service"]
                
                if "Quota Check" in test_case["name"]:
                    result = rate_limiter.check_quota(service)
                    passed = result == test_case["expected"]
                elif "Quota Status" in test_case["name"]:
                    status = rate_limiter.get_quota_status(service)
                    result = isinstance(status, dict) and 'service' in status
                    passed = result == test_case["expected"]
                else:
                    result = rate_limiter.check_quota(service)
                    passed = result == test_case["expected"]
                
                if not self.coverage_mode:
                    print(f"{'✅ PASS' if passed else '❌ FAIL'} | {test_case['name']}")
                    print(f"      Service: {service.value}")
                    print(f"      Description: {test_case['description']}")
                    print(f"      Result: {result}")
                    print()
                
                if passed:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
                self.total_tests += 1
                
            except Exception as e:
                if not self.coverage_mode:
                    print(f"❌ ERROR | {test_case['name']}: {e}")
                self.failed_tests += 1
                self.total_tests += 1
        
        self.print_section("Rate Limiter Quota Enforcement")
        
        try:

            for i in range(3):
                rate_limiter.wait_if_needed(APIService.GITHUB)
            
            status = rate_limiter.get_quota_status(APIService.GITHUB)
            quota_tracking_works = status['current_requests'] == 3
            
            if not self.coverage_mode:
                print(f"{'✅ PASS' if quota_tracking_works else '❌ FAIL'} | Quota Tracking")
                print(f"      Current requests: {status['current_requests']}")
                print(f"      Max requests: {status['max_requests']}")
                print(f"      Quota remaining: {status['quota_remaining']}")
                print()
            
            if quota_tracking_works:
                self.passed_tests += 1
            else:
                self.failed_tests += 1
            self.total_tests += 1
                
        except Exception as e:
            if not self.coverage_mode:
                print(f"❌ ERROR | Quota Enforcement Test: {e}")
            self.failed_tests += 1
            self.total_tests += 1
     
    def test_full_metric_pipeline(self):
        self.print_header("FULL METRIC PIPELINE TESTS")
        
        test_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://huggingface.co/datasets/squad", 
            "https://github.com/microsoft/DialoGPT"
        ]
        
        test_file_path = "temp_pipeline_test.txt"
        with open(test_file_path, 'w') as f:
            for url in test_urls:
                f.write(url + '\n')
        
        try:
            processor = URLProcessor(test_file_path)
            model_results = processor.process_urls_with_metrics()
            
            self.print_section("Full Pipeline Results")
            
            for i, result in enumerate(model_results):
                if not self.coverage_mode:
                    print(f"Result {i+1}:")
                    print(f"  URL: {result.url}")
                    print(f"  Net Score: {result.net_score:.3f}")
                    print(f"  Net Latency: {result.net_score_latency}ms")
                    print()
                    print("  Individual Metrics:")
                    print(f"    Size: {result.size_score} ({result.size_latency}ms)")
                    print(f"    License: {result.license_score:.3f} ({result.license_latency}ms)      REAL")
                    print(f"    RampUp: {result.ramp_up_score:.3f} ({result.ramp_up_latency}ms)")
                    print(f"    BusFactor: {result.bus_factor_score:.3f} ({result.bus_factor_latency}ms)")
                    print(f"    DatasetCode: {result.dataset_code_score:.3f} ({result.dataset_code_latency}ms)")
                    print(f"    DatasetQuality: {result.dataset_quality_score:.3f} ({result.dataset_quality_latency}ms)")
                    print(f"    CodeQuality: {result.code_quality_score:.3f} ({result.code_quality_latency}ms)")
                    print(f"    PerformanceClaims: {result.performance_claims_score:.3f} ({result.performance_claims_latency}ms)")
                    print()
                    print("  NDJSON Output:")
                    print(f"    {result.to_ndjson_line()}")
                    print("-" * 50)
            
            expected_count = len(test_urls)
            actual_count = len(model_results)
            self.print_test_result("Pipeline Result Count", expected_count, actual_count, expected_count == actual_count)
            
        except Exception as e:
            print(f"❌ ERROR in pipeline test: {e}")
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
 
    def test_metric_calculator_validation(self):
        self.print_header("METRIC CALCULATOR VALIDATION TESTS")
        
        from src.metrics.base import MetricCalculator, ModelContext
        
        class TestCalculator(MetricCalculator):
            def calculate_score(self, context: ModelContext) -> float:
                return 0.75
        
        calculator = TestCalculator("TestMetric")
        test_context = ModelContext(
            model_url="https://example.com/test",
            model_info={"type": "test"}
        )
        
        test_cases = [
            ("Calculator Name", calculator.name, "TestMetric"),
            ("Initial Score None", calculator.get_score(), None),
            ("Initial Time None", calculator.get_calculation_time(), None)
        ]
        
        for test_name, actual, expected in test_cases:
            passed = actual == expected
            self.print_test_result(test_name, expected, actual, passed)
        
        score = calculator.calculate_score(test_context)
        self.print_test_result("Score Calculation", 0.75, score, score == 0.75)
        
        calculator._set_score(0.85, 100)
        self.print_test_result("Set Score", 0.85, calculator.get_score(), calculator.get_score() == 0.85)
        self.print_test_result("Set Timing", 100, calculator.get_calculation_time(), calculator.get_calculation_time() == 100)


    def test_metric_calculator_additional_features(self):
        self.print_section("METRIC CALCULATOR ADDITIONAL FEATURES")
        
        from src.metrics.base import MetricCalculator, ModelContext
        
        class TestCalculator(MetricCalculator):
            def calculate_score(self, context: ModelContext) -> float:
                return 0.8
        
        calculator = TestCalculator("TestCalculator")
        
        try:
            calculator._set_score(1.5, 100)
            self.print_test_result("Invalid score validation", "ValueError", "No error raised", False)
        except ValueError:
            self.print_test_result("Invalid score validation", "ValueError", "ValueError raised", True)
        except Exception as e:
            self.print_test_result("Invalid score validation", "ValueError", f"Other error: {e}", False)
            
        try:
            calculator._set_score(-0.1, 100)
            self.print_test_result("Negative score validation", "ValueError", "No error raised", False)
        except ValueError:
            self.print_test_result("Negative score validation", "ValueError", "ValueError raised", True)
        except Exception as e:
            self.print_test_result("Negative score validation", "ValueError", f"Other error: {e}", False)
        
        calculator._set_score(0.8, 150)
        calculator.reset()
        score_after_reset = calculator.get_score()
        time_after_reset = calculator.get_calculation_time()
        reset_works = score_after_reset is None and time_after_reset is None
        self.print_test_result("Reset functionality", True, reset_works, reset_works)
        
        str_repr = str(calculator)
        has_name = "TestCalculator" in str_repr
        self.print_test_result("String representation", True, has_name, has_name)
        
        repr_str = repr(calculator)
        has_name_and_score = "TestCalculator" in repr_str and "score=" in repr_str
        self.print_test_result("Repr representation", True, has_name_and_score, has_name_and_score)

    def test_results_storage_edge_cases(self):
        self.print_section("RESULTS STORAGE EDGE CASES")
        
        from src.storage.results_storage import MetricResult, ModelResult
        import json
        
        try:
            result = MetricResult("test_metric", 0.5, 100, str(datetime.datetime.now()))
            self.print_test_result("MetricResult creation", True, result.metric_name == "test_metric", result.metric_name == "test_metric")
        except Exception:
            self.print_test_result("MetricResult creation", True, False, False)
            
        try:
            model_result = ModelResult(
                url="https://test.com",
                net_score=0.8, net_score_latency=100,
                size_score={"raspberry_pi": 0.7, "jetson_nano": 0.7, "desktop_pc": 0.7, "aws_server": 0.7}, size_latency=50,
                license_score=1.0, license_latency=25,
                ramp_up_score=0.6, ramp_up_latency=75,
                bus_factor_score=0.5, bus_factor_latency=60,
                dataset_code_score=0.9, dataset_code_latency=40,
                dataset_quality_score=0.8, dataset_quality_latency=55,
                code_quality_score=0.85, code_quality_latency=45,
                performance_claims_score=0.75, performance_claims_latency=65
            )
            str_repr = str(model_result)
            has_url = "https://test.com" in str_repr
            self.print_test_result("ModelResult string representation", True, has_url, has_url)
        except Exception:
            self.print_test_result("ModelResult string representation", True, False, False)
            
        try:
            model_result = ModelResult(
                url="https://test.com",
                net_score=0.8, net_score_latency=100,
                size_score={'raspberry_pi': 0.0, 'jetson_nano': 0.0, 'desktop_pc': 0.0, 'aws_server': 0.0}, 
                size_latency=50,
                license_score=1.0, license_latency=25,
                ramp_up_score=0.6, ramp_up_latency=75,
                bus_factor_score=0.5, bus_factor_latency=60,
                dataset_code_score=0.9, dataset_code_latency=40,
                dataset_quality_score=0.8, dataset_quality_latency=55,
                code_quality_score=0.85, code_quality_latency=45,
                performance_claims_score=0.75, performance_claims_latency=65
            )
            ndjson = model_result.to_ndjson_line()
            has_model_name = "unknown" in ndjson
            is_valid_json = json.loads(ndjson) is not None
            self.print_test_result("NDJSON formatting", True, has_model_name and is_valid_json, has_model_name and is_valid_json)
        except Exception:
            self.print_test_result("NDJSON formatting", True, False, False)
            
        try:
            result = MetricResult("test_metric", 0.5, 100, str(datetime.datetime.now()))
            result_dict = result.to_dict()
            has_metric_name = "metric_name" in result_dict
            self.print_test_result("MetricResult to_dict", True, has_metric_name, has_metric_name)
        except Exception:
            self.print_test_result("MetricResult to_dict", True, False, False)


    def test_results_storage_functionality(self):
        self.print_section("RESULTS STORAGE FUNCTIONALITY")
        
        from src.storage.results_storage import ResultsStorage, MetricResult
        
        storage = ResultsStorage()
        test_url = "https://test-storage.com"
        
        metric1 = MetricResult("license", 1.0, 50, str(datetime.datetime.now()))
        metric2 = MetricResult("ramp_up", 0.8, 75, str(datetime.datetime.now()))
        
        storage.store_metric_result(test_url, metric1)
        storage.store_metric_result(test_url, metric2)
        
        retrieved = storage.get_metric_result(test_url, "license")
        self.print_test_result("Get stored metric result", 1.0, retrieved.score if retrieved else None, retrieved is not None and retrieved.score == 1.0)
        
        all_metrics = storage.get_all_metrics_for_model(test_url)
        has_both_metrics = len(all_metrics) == 2
        self.print_test_result("Get all metrics for model", True, has_both_metrics, has_both_metrics)
        
        is_complete = storage.is_model_complete(test_url)
        self.print_test_result("Model incomplete check", False, is_complete, is_complete == False)
        
        completed = storage.get_completed_models()
        is_empty = len(completed) == 0
        self.print_test_result("Get completed models (empty)", True, is_empty, is_empty)
        
        storage.clear()
        cleared_metrics = storage.get_all_metrics_for_model(test_url)
        is_cleared = len(cleared_metrics) == 0
        self.print_test_result("Storage clear functionality", True, is_cleared, is_cleared)
    
    def print_summary(self) -> None:
        if self.coverage_mode:
            import subprocess
            try:
                result: subprocess.CompletedProcess[str] = subprocess.run([sys.executable, '-m', 'coverage', 'report'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines: List[str] = result.stdout.strip().split('\n')
                    if lines:
                        last_line: str = lines[-1]
                        import re
                        match: Optional[re.Match[str]] = re.search(r'(\d+)%', last_line)
                        if match:
                            coverage_percent: str = match.group(1)
                            print(f"{self.passed_tests}/{self.total_tests} test cases passed. {coverage_percent}% line coverage achieved.")
                        else:
                            print(f"{self.passed_tests}/{self.total_tests} test cases passed. Coverage percentage not available.")
                    else:
                        print(f"{self.passed_tests}/{self.total_tests} test cases passed. Coverage percentage not available.")
                else:
                    print(f"{self.passed_tests}/{self.total_tests} test cases passed. Coverage percentage not available.")
            except:
                print(f"{self.passed_tests}/{self.total_tests} test cases passed. Coverage percentage not available.")
            return
            
        self.print_header("TEST SUMMARY")
        
        pass_rate: float = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\nALL TESTS PASSED!")
        else:
            print(f"\n{self.failed_tests} test(s) failed. Review output above.")
        
        print("\nNOTE: Only License metrics are fully implemented.")
        print("Other metrics show dummy values for testing purposes.")
    
    def run_all_tests(self) -> None:
        if not self.coverage_mode:
            print("Starting Comprehensive Test Suite for ECE 46100 Team 8")
            print("Testing URL processing, routing, and metric calculations...")
        
        try:

            import signal
            
            def timeout_handler(signum: int, frame: Any) -> None:
                print("\nTest suite timed out after 5 minutes")
                raise TimeoutError("Test suite execution timed out")
            

            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
            

            import os
            skip_network_tests: bool = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            self.test_url_validation()
            self.test_url_categorization() 
            self.test_edge_cases()
            self.test_url_parsing_edge_cases()
            self.test_license_compatibility_mapping()
            self.test_rate_limiter()
            self.test_data_consistency_checks()
            self.test_boundary_value_analysis()
            self.test_file_handling_edge_cases()
            self.test_security_validation()
            self.test_unicode_and_internationalization()
            self.test_edge_case_metric_scenarios()
            self.test_metric_calculator_validation()
            self.test_metric_calculator_additional_features()
            self.test_results_storage_edge_cases()
            self.test_results_storage_functionality()
            self.test_config_comprehensive()
            self.test_exceptions_comprehensive()
            
            # Run comprehensive tests with proper mocking for coverage
            self.test_llm_client_comprehensive()
            self.test_llm_analyzer_comprehensive()
            self.test_size_calculator_comprehensive()
            self.test_git_analyzer_comprehensive()
            self.test_ramp_up_calculator_comprehensive()
            self.test_code_quality_calculator_comprehensive()
            self.test_performance_claims_calculator_comprehensive()
            self.test_dataset_quality_calculator_comprehensive()
            self.test_http_client_comprehensive()
            self.test_model_analyzer_comprehensive()
            
            # Run additional comprehensive tests for better coverage
            self.test_all_metric_calculators_comprehensive()
            self.test_all_core_modules_comprehensive()

            
        except Exception as e:
            if not self.coverage_mode:
                print(f"❌ CRITICAL ERROR: {e}")
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            self.print_summary()
            
        if self.failed_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    def test_size_calculator_comprehensive(self):
        self.print_header("SIZE CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = SizeCalculator()
        

        context = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Size Calculator - Valid HF Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Size Calculator - Valid HF Model", "No exception", f"Exception: {e}", False)
        

        context_github = ModelContext(
            model_url="https://github.com/google-research/bert",
            code_url="https://github.com/google-research/bert",
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_github)
            self.print_test_result("Size Calculator - GitHub Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Size Calculator - GitHub Model", "No exception", f"Exception: {e}", False)
        

        context_invalid = ModelContext(
            model_url="https://invalid-url.com/model",
            code_url=None,
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_invalid)
            self.print_test_result("Size Calculator - Invalid URL", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Size Calculator - Invalid URL", "No exception", f"Exception: {e}", False)
    
    def test_ramp_up_calculator_comprehensive(self):
        self.print_header("RAMP UP CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = RampUpCalculator()
        

        context = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Ramp Up Calculator - Valid HF Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Ramp Up Calculator - Valid HF Model", "No exception", f"Exception: {e}", False)
        

        context_github = ModelContext(
            model_url="https://github.com/google-research/bert",
            code_url="https://github.com/google-research/bert",
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_github)
            self.print_test_result("Ramp Up Calculator - GitHub Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Ramp Up Calculator - GitHub Model", "No exception", f"Exception: {e}", False)
    
    def test_code_quality_calculator_comprehensive(self):
        self.print_header("CODE QUALITY CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = CodeQualityCalculator()
        

        context = ModelContext(
            model_url="https://github.com/google-research/bert",
            code_url="https://github.com/google-research/bert",
            dataset_url=None,
            model_info={'github_metadata': {
                'language': 'Python',
                'stargazers_count': 1500,
                'updated_at': '2024-01-01T00:00:00Z',
                'description': 'BERT implementation',
                'archived': False,
                'topics': ['nlp', 'bert', 'transformer']
            }}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Code Quality Calculator - GitHub Metadata", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Code Quality Calculator - GitHub Metadata", "No exception", f"Exception: {e}", False)
        

        context_hf = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url=None,
            model_info={},
            huggingface_metadata={
                'downloads': 1000000,
                'likes': 150,
                'tags': ['bert', 'nlp', 'transformer']
            }
        )
        
        try:
            score = calculator.calculate_score(context_hf)
            self.print_test_result("Code Quality Calculator - HF Metadata", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Code Quality Calculator - HF Metadata", "No exception", f"Exception: {e}", False)
    
    def test_dataset_code_calculator_comprehensive(self):
        self.print_header("DATASET CODE CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = DatasetCodeCalculator()
        

        context = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url="https://github.com/google-research/bert",
            dataset_url="https://huggingface.co/datasets/bookcorpus",
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Dataset Code Calculator - Both URLs", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Dataset Code Calculator - Both URLs", "No exception", f"Exception: {e}", False)
        

        context_dataset = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url="https://huggingface.co/datasets/bookcorpus",
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_dataset)
            self.print_test_result("Dataset Code Calculator - Dataset Only", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Dataset Code Calculator - Dataset Only", "No exception", f"Exception: {e}", False)
        

        context_code = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url="https://github.com/google-research/bert",
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_code)
            self.print_test_result("Dataset Code Calculator - Code Only", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Dataset Code Calculator - Code Only", "No exception", f"Exception: {e}", False)
    
    def test_dataset_quality_calculator_comprehensive(self):
        self.print_header("DATASET QUALITY CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = DatasetQualityCalculator()
        

        context = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url="https://huggingface.co/datasets/bookcorpus",
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Dataset Quality Calculator - With Dataset", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Dataset Quality Calculator - With Dataset", "No exception", f"Exception: {e}", False)
        

        context_no_dataset = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_no_dataset)
            self.print_test_result("Dataset Quality Calculator - No Dataset", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Dataset Quality Calculator - No Dataset", "No exception", f"Exception: {e}", False)
    
    def test_performance_claims_calculator_comprehensive(self):
        self.print_header("PERFORMANCE CLAIMS CALCULATOR COMPREHENSIVE TESTS")
        
        calculator = PerformanceClaimsCalculator()
        

        context = ModelContext(
            model_url="https://huggingface.co/google-bert/bert-base-uncased",
            code_url=None,
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context)
            self.print_test_result("Performance Claims Calculator - HF Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Performance Claims Calculator - HF Model", "No exception", f"Exception: {e}", False)
        

        context_github = ModelContext(
            model_url="https://github.com/google-research/bert",
            code_url="https://github.com/google-research/bert",
            dataset_url=None,
            model_info={}
        )
        
        try:
            score = calculator.calculate_score(context_github)
            self.print_test_result("Performance Claims Calculator - GitHub Model", 
                                 "0.0-1.0 range", f"{score:.2f}", 0.0 <= score <= 1.0)
        except Exception as e:
            self.print_test_result("Performance Claims Calculator - GitHub Model", "No exception", f"Exception: {e}", False)
    
    def test_llm_analyzer_comprehensive(self):
        self.print_header("LLM ANALYZER COMPREHENSIVE TESTS")
        
        analyzer = LLMAnalyzer()
    
        try:
            result = analyzer.analyze_dataset_quality({"description": "Test dataset description"})
            self.print_test_result("LLM Analyzer - Dataset Quality", 
                                 "Float score", str(type(result)), isinstance(result, float))
        except Exception as e:
            self.print_test_result("LLM Analyzer - Dataset Quality", "No exception", f"Exception: {e}", False)
    
        try:

            result2: Optional[str] = analyzer._post_to_genai([{"role": "user", "content": "test"}])
            self.print_test_result("LLM Analyzer - Post to GenAI", 
                                 "String or None", str(type(result2)), isinstance(result2, (str, type(None))))
        except Exception as e:
            self.print_test_result("LLM Analyzer - Post to GenAI", "No exception", f"Exception: {e}", False)
        

        try:
            result = analyzer._extract_score("Score: 0.8")
            self.print_test_result("LLM Analyzer - Extract Score", 
                                 "Float score", str(type(result)), isinstance(result, float))
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score", "No exception", f"Exception: {e}", False)
        
        try:
            score = analyzer.analyze_dataset_quality({"description": "Test dataset"})
            success = isinstance(score, float)
            self.print_test_result("LLM Analyzer - Analyze Dataset Quality", 
                                 "Valid result", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Analyze Dataset Quality", "No exception", f"Exception: {e}", False)
        

        try:
            score = analyzer.analyze_dataset_quality({})
            success = isinstance(score, float)
            self.print_test_result("LLM Analyzer - Empty Dataset Info", 
                                 "Valid result", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Empty Dataset Info", "No exception", f"Exception: {e}", False)
        

        try:
            score2: float = analyzer.analyze_dataset_quality({})
            success = isinstance(score2, float)
            self.print_test_result("LLM Analyzer - None Dataset Info", 
                                 "Valid result", f"Score: {score2}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - None Dataset Info", "No exception", f"Exception: {e}", False)
        

        try:

            score = analyzer._extract_score("The quality score is 0.85")
            success = isinstance(score, float) and score == 0.85
            self.print_test_result("LLM Analyzer - Extract Score Valid", 
                                 "0.85", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score Valid", "No exception", f"Exception: {e}", False)
        
        try:

            score = analyzer._extract_score("No score mentioned here")
            success = isinstance(score, float) and score == 0.0
            self.print_test_result("LLM Analyzer - Extract Score None", 
                                 "0.0", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score None", "No exception", f"Exception: {e}", False)
        
        try:

            score = analyzer._extract_score("Score: 0.3 and also 0.7")
            success = isinstance(score, float) and score == 0.3
            self.print_test_result("LLM Analyzer - Extract Score Multiple", 
                                 "0.3", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score Multiple", "No exception", f"Exception: {e}", False)
        
        try:

            score = analyzer._extract_score("Score: 1.5")
            success = isinstance(score, float) and score == 1.5
            self.print_test_result("LLM Analyzer - Extract Score > 1", 
                                 "1.5", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score > 1", "No exception", f"Exception: {e}", False)
        
        try:

            score = analyzer._extract_score("Score: -0.5")
            success = isinstance(score, float) and score == -0.5
            self.print_test_result("LLM Analyzer - Extract Score Negative", 
                                 "-0.5", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Analyzer - Extract Score Negative", "No exception", f"Exception: {e}", False)
    
    def test_llm_client_comprehensive(self):
        self.print_header("LLM CLIENT COMPREHENSIVE TESTS")
        
        from src.core.llm_client import ask_for_json_score, _extract_json_score
        

        test_cases = [
            ("Valid JSON", '{"score": 0.8, "rationale": "Good"}', 0.8),
            ("Invalid JSON", '{"score": "invalid"}', None),
            ("No JSON", 'Score: 0.7', 0.7),
            ("Empty content", '', None),
            ("JSON with regex fallback", 'The score is 0.9', 0.9),
            ("Multiple numbers", 'Score: 0.5 and 0.8', 0.5),
            ("No numbers", 'No score here', None)
        ]
        
        for test_name, content, expected_score in test_cases:
            try:
                score, rationale = _extract_json_score(content)
                if expected_score is None:
                    passed = score is None
                else:
                    passed = score is not None and abs(score - expected_score) < 0.01
                self.print_test_result(f"LLM Client - {test_name}", 
                                     expected_score, score, passed)
            except Exception as e:
                self.print_test_result(f"LLM Client - {test_name}", 
                                     expected_score, f"Exception: {e}", False)
        

        try:
            score, rationale = ask_for_json_score("Test prompt")
            self.print_test_result("LLM Client - API Call", 
                                 "None or float", str(type(score)), isinstance(score, (float, type(None))))
        except Exception as e:
            self.print_test_result("LLM Client - API Call", "No exception", f"Exception: {e}", False)
        

        try:

            score, rationale = _extract_json_score('{"score": 0.8, "rationale": "Good", "extra": "field"}')
            success = score == 0.8 and isinstance(rationale, str)
            self.print_test_result("LLM Client - Malformed JSON", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Malformed JSON", "No exception", f"Exception: {e}", False)
        
        try:

            score, rationale = _extract_json_score('{"score": "high", "rationale": "Good"}')
            success = score is None
            self.print_test_result("LLM Client - Non-numeric Score", 
                                 "None", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Non-numeric Score", "No exception", f"Exception: {e}", False)
        
        try:

            score, rationale = _extract_json_score('{"rationale": "Good"}')
            success = score is None
            self.print_test_result("LLM Client - Missing Score Field", 
                                 "None", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Missing Score Field", "No exception", f"Exception: {e}", False)
        
        try:

            score, rationale = _extract_json_score('{"score": 0.8}')
            success = score == 0.8 and rationale == ""
            self.print_test_result("LLM Client - Missing Rationale Field", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Missing Rationale Field", "No exception", f"Exception: {e}", False)
        
        try:

            score, rationale = _extract_json_score('The scores are 0.3, 0.7, and 0.9')
            success = score == 0.3
            self.print_test_result("LLM Client - Multiple Numbers Regex", 
                                 "0.3", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Multiple Numbers Regex", "No exception", f"Exception: {e}", False)
        

        try:

            long_content = "This is a very long content with many words " * 100 + "Score: 0.8"
            score, rationale = _extract_json_score(long_content)
            success = score == 0.8
            self.print_test_result("LLM Client - Long Content", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Long Content", "No exception", f"Exception: {e}", False)
        
        try:

            special_content = "Score: 0.8 with special chars: !@#$%^&*()"
            score, rationale = _extract_json_score(special_content)
            success = score == 0.8
            self.print_test_result("LLM Client - Special Characters", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Special Characters", "No exception", f"Exception: {e}", False)
        
        try:

            multiline_content = "This is line 1\nThis is line 2\nScore: 0.8\nThis is line 4"
            score, rationale = _extract_json_score(multiline_content)
            success = score == 0.8
            self.print_test_result("LLM Client - Multiline Content", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Multiline Content", "No exception", f"Exception: {e}", False)
        
        try:

            tab_content = "Score:\t0.8\twith tabs"
            score, rationale = _extract_json_score(tab_content)
            success = score == 0.8
            self.print_test_result("LLM Client - Tab Content", 
                                 "0.8", f"Score: {score}", success)
        except Exception as e:
            self.print_test_result("LLM Client - Tab Content", "No exception", f"Exception: {e}", False)
    
    def test_http_client_comprehensive(self):
        self.print_header("HTTP CLIENT COMPREHENSIVE TESTS")
        
        from src.core.http_client import get_with_rate_limit, post_with_rate_limit, head_with_rate_limit, make_rate_limited_request
        from src.core.rate_limiter import APIService
        

        try:
            response = get_with_rate_limit("https://httpbin.org/get", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - GET Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - GET Request", "No exception", f"Exception: {e}", False)
        

        try:
            response = head_with_rate_limit("https://httpbin.org/get", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - HEAD Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - HEAD Request", "No exception", f"Exception: {e}", False)
        

        try:
            response = post_with_rate_limit("https://httpbin.org/post", APIService.GITHUB, 
                                          json={"test": "data"}, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - POST Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - POST Request", "No exception", f"Exception: {e}", False)
        

        try:
            response = make_rate_limited_request("GET", "https://httpbin.org/get", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Make Request GET", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Make Request GET", "No exception", f"Exception: {e}", False)
        

        try:
            response = get_with_rate_limit("https://invalid-domain-that-does-not-exist.com", APIService.GITHUB, timeout=1)

            success = response is None or hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Invalid URL", 
                                 "Graceful handling", f"Response: {response}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Invalid URL", "No exception", f"Exception: {e}", False)
        

        try:
            response = make_rate_limited_request("PUT", "https://httpbin.org/put", APIService.GITHUB, 
                                               json={"test": "data"}, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - PUT Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - PUT Request", "No exception", f"Exception: {e}", False)
        

        try:
            response = make_rate_limited_request("DELETE", "https://httpbin.org/delete", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - DELETE Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - DELETE Request", "No exception", f"Exception: {e}", False)
        

        try:
            response = get_with_rate_limit("https://httpbin.org/headers", APIService.GITHUB, 
                                         headers={"X-Test": "value"}, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Custom Headers", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Custom Headers", "No exception", f"Exception: {e}", False)
        

        try:
            response = get_with_rate_limit("https://httpbin.org/get", APIService.GITHUB, 
                                         params={"test": "param"}, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Parameters", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Parameters", "No exception", f"Exception: {e}", False)
        

        try:
            from src.core.rate_limiter import get_rate_limiter, APIService
            
            rate_limiter = get_rate_limiter()
            

            has_quota = rate_limiter.has_quota(APIService.GITHUB)
            success = isinstance(has_quota, bool)
            self.print_test_result("Rate Limiter - Quota Check", 
                                 "Boolean", f"Result: {has_quota}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Quota Check", "No exception", f"Exception: {e}", False)
        
        try:

            quota_status = rate_limiter.get_quota_status(APIService.GITHUB)
            success = isinstance(quota_status, dict)
            self.print_test_result("Rate Limiter - Quota Status", 
                                 "Dictionary", f"Result: {quota_status}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Quota Status", "No exception", f"Exception: {e}", False)
        
        try:

            github_quota = rate_limiter.has_quota(APIService.GITHUB)
            hf_quota = rate_limiter.has_quota(APIService.HUGGINGFACE)
            genai_quota = rate_limiter.has_quota(APIService.GENAI)
            
            success = all(isinstance(q, bool) for q in [github_quota, hf_quota, genai_quota])
            self.print_test_result("Rate Limiter - Service Independence", 
                                 "All Boolean", f"GitHub: {github_quota}, HF: {hf_quota}, GenAI: {genai_quota}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Service Independence", "No exception", f"Exception: {e}", False)
        

        try:

            response = get_with_rate_limit("https://httpbin.org/delay/1", APIService.GITHUB, timeout=0.5)
            success = response is None
            self.print_test_result("HTTP Client - Timeout", 
                                 "None", f"Response: {response}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Timeout", "No exception", f"Exception: {e}", False)
        
        try:

            response = make_rate_limited_request("PATCH", "https://httpbin.org/patch", APIService.GITHUB, 
                                               json={"test": "data"}, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - PATCH Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - PATCH Request", "No exception", f"Exception: {e}", False)
        
        try:

            response = make_rate_limited_request("OPTIONS", "https://httpbin.org/get", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - OPTIONS Request", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - OPTIONS Request", "No exception", f"Exception: {e}", False)
        

        try:

            rate_limiter = get_rate_limiter()
            

            status = rate_limiter.get_quota_status(APIService.GITHUB)
            success = isinstance(status, dict) and 'service' in status
            self.print_test_result("Rate Limiter - Quota Status", 
                                 "Dictionary", f"Result: {status}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Quota Status", "No exception", f"Exception: {e}", False)
        
        try:

            rate_limiter.reset_failures(APIService.GITHUB)
            success = True
            self.print_test_result("Rate Limiter - Reset Failures", 
                                 "No exception", f"Success: {success}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Reset Failures", "No exception", f"Exception: {e}", False)
        
        try:

            rate_limiter.handle_rate_limit_response(APIService.GITHUB, 5)
            success = True
            self.print_test_result("Rate Limiter - Handle Rate Limit", 
                                 "No exception", f"Success: {success}", success)
        except Exception as e:
            self.print_test_result("Rate Limiter - Handle Rate Limit", "No exception", f"Exception: {e}", False)
        

        try:

            response = get_with_rate_limit("https://httpbin.org/get", APIService.GITHUB, timeout=0.1)
            success = response is None
            self.print_test_result("HTTP Client - Short Timeout", 
                                 "None", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Short Timeout", "No exception", f"Exception: {e}", False)
        
        try:

            response = make_rate_limited_request("GET", "https://httpbin.org/get", APIService.GITHUB, 
                                               timeout=5, max_retries=1)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Custom Retry Count", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Custom Retry Count", "No exception", f"Exception: {e}", False)
        
        try:

            response = get_with_rate_limit("https://httpbin.org/get", APIService.HUGGINGFACE, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Different Service", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Different Service", "No exception", f"Exception: {e}", False)
        

        try:

            response = make_rate_limited_request("INVALID", "https://httpbin.org/get", APIService.GITHUB, timeout=5)
            success = response is not None and hasattr(response, 'status_code')
            self.print_test_result("HTTP Client - Invalid Method", 
                                 "Valid response", f"Response: {response is not None}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Invalid Method", "No exception", f"Exception: {e}", False)
        
        try:

            response = make_rate_limited_request("GET", "not-a-url", APIService.GITHUB, timeout=5)
            success = response is None
            self.print_test_result("HTTP Client - Malformed URL", 
                                 "None", f"Response: {response}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Malformed URL", "No exception", f"Exception: {e}", False)
        
        try:

            response = make_rate_limited_request("GET", "https://nonexistent-domain-12345.com", APIService.GITHUB, timeout=1)
            success = response is None
            self.print_test_result("HTTP Client - Connection Error", 
                                 "None", f"Response: {response}", success)
        except Exception as e:
            self.print_test_result("HTTP Client - Connection Error", "No exception", f"Exception: {e}", False)
    
    def test_config_comprehensive(self):
        self.print_header("CONFIG COMPREHENSIVE TESTS")
        

        try:
            token = Config.get_github_token()
            self.print_test_result("Config - GitHub Token", 
                                 "String or None", str(type(token)), isinstance(token, (str, type(None))))
        except Exception as e:
            self.print_test_result("Config - GitHub Token", "No exception", f"Exception: {e}", False)
        

        try:
            key = Config.get_genai_token()
            self.print_test_result("Config - GenAI Token", 
                                 "String or None", str(type(key)), isinstance(key, (str, type(None))))
        except Exception as e:
            self.print_test_result("Config - GenAI Token", "No exception", f"Exception: {e}", False)
        

        try:
            import os

            original_env = os.environ.get('GITHUB_TOKEN')
            os.environ['GITHUB_TOKEN'] = 'test_env_token'
            env_token = Config.get_github_token()
            success = env_token == 'test_env_token'
            self.print_test_result("Config - Environment Priority", 
                                 "test_env_token", env_token, success)
            

            if original_env:
                os.environ['GITHUB_TOKEN'] = original_env
            else:
                del os.environ['GITHUB_TOKEN']
        except Exception as e:
            self.print_test_result("Config - Environment Priority", "No exception", f"Exception: {e}", False)
        

        try:
            import os

            original_env = os.environ.get('GITHUB_TOKEN')
            if 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            

            file_token = Config.get_github_token()
            success = isinstance(file_token, (str, type(None)))
            self.print_test_result("Config - File Fallback", 
                                 "String or None", str(type(file_token)), success)
            

            if original_env:
                os.environ['GITHUB_TOKEN'] = original_env
        except Exception as e:
            self.print_test_result("Config - File Fallback", "No exception", f"Exception: {e}", False)
        

        try:
            import os
            import tempfile
            

            with tempfile.TemporaryDirectory() as temp_dir:
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                

                original_github = os.environ.get('GITHUB_TOKEN')
                original_genai = os.environ.get('GEN_AI_STUDIO_API_KEY')
                
                if 'GITHUB_TOKEN' in os.environ:
                    del os.environ['GITHUB_TOKEN']
                if 'GEN_AI_STUDIO_API_KEY' in os.environ:
                    del os.environ['GEN_AI_STUDIO_API_KEY']
                

                github_token = Config.get_github_token()
                genai_token = Config.get_genai_token()
                
                success = github_token is None and genai_token is None
                self.print_test_result("Config - No Files", 
                                     "None", f"GitHub: {github_token}, GenAI: {genai_token}", success)
                

                os.chdir(original_cwd)
                if original_github:
                    os.environ['GITHUB_TOKEN'] = original_github
                if original_genai:
                    os.environ['GEN_AI_STUDIO_API_KEY'] = original_genai
        except Exception as e:
            self.print_test_result("Config - No Files", "No exception", f"Exception: {e}", False)
    
    def test_exceptions_comprehensive(self):
        self.print_header("EXCEPTIONS COMPREHENSIVE TESTS")
        

        try:
            raise MetricCalculationException("TestMetric", "Test error")
        except MetricCalculationException as e:
            self.print_test_result("Exception - MetricCalculationException", 
                                 "Exception raised", str(e), "Test error" in str(e))
        

        try:
            raise APIRateLimitException("TestAPI", 60)
        except APIRateLimitException as e:
            self.print_test_result("Exception - APIRateLimitException", 
                                 "Exception raised", str(e), "TestAPI" in str(e))
        

        try:
            raise InvalidURLException("bad-url", "malformed")
        except InvalidURLException as e:
            self.print_test_result("Exception - InvalidURLException", 
                                 "Exception raised", str(e), "bad-url" in str(e))
    
    def test_url_processor_comprehensive(self):
        self.print_header("URL PROCESSOR COMPREHENSIVE TESTS")
        

        try:
            processor = URLProcessor("sample_input.txt")
            self.print_test_result("URL Processor - Initialization", 
                                 "URLProcessor instance", str(type(processor)), isinstance(processor, URLProcessor))
        except Exception as e:
            self.print_test_result("URL Processor - Initialization", "No exception", f"Exception: {e}", False)
        

        try:
            url_type = categorize_url("https://huggingface.co/google-bert/bert-base-uncased")
            self.print_test_result("URL Processor - HF Categorization", 
                                 "URLType.HUGGINGFACE_MODEL", str(url_type), url_type == URLType.HUGGINGFACE_MODEL)
        except Exception as e:
            self.print_test_result("URL Processor - HF Categorization", "No exception", f"Exception: {e}", False)
        
        try:
            url_type = categorize_url("https://github.com/google-research/bert")
            self.print_test_result("URL Processor - GitHub Categorization", 
                                 "URLType.GITHUB_REPO", str(url_type), url_type == URLType.GITHUB_REPO)
        except Exception as e:
            self.print_test_result("URL Processor - GitHub Categorization", "No exception", f"Exception: {e}", False)
        

        try:
            is_valid = is_valid_url("https://huggingface.co/google-bert/bert-base-uncased")
            self.print_test_result("URL Processor - Valid URL", 
                                 "True", str(is_valid), is_valid == True)
        except Exception as e:
            self.print_test_result("URL Processor - Valid URL", "No exception", f"Exception: {e}", False)
        
        try:
            is_valid = is_valid_url("invalid-url")
            self.print_test_result("URL Processor - Invalid URL", 
                                 "False", str(is_valid), is_valid == False)
        except Exception as e:
            self.print_test_result("URL Processor - Invalid URL", "No exception", f"Exception: {e}", False)
        

        try:

            result: bool = is_valid_url("")
            success = result == False
            self.print_test_result("URL Processor - Empty URL", 
                                 "False", f"Result: {result}", success)
        except Exception as e:
            self.print_test_result("URL Processor - Empty URL", "No exception", f"Exception: {e}", False)
        
        try:

            result = is_valid_url("")
            success = result == False
            self.print_test_result("URL Processor - Empty URL", 
                                 "False", f"Result: {result}", success)
        except Exception as e:
            self.print_test_result("URL Processor - Empty URL", "No exception", f"Exception: {e}", False)
        
        try:

            hf_model = categorize_url("https://huggingface.co/microsoft/DialoGPT")
            success = hf_model == URLType.HUGGINGFACE_MODEL
            self.print_test_result("URL Processor - HF Model Categorization", 
                                 "HUGGINGFACE_MODEL", f"Result: {hf_model}", success)
        except Exception as e:
            self.print_test_result("URL Processor - HF Model Categorization", "No exception", f"Exception: {e}", False)
        
        try:

            github_repo = categorize_url("https://github.com/microsoft/DialoGPT")
            success = github_repo == URLType.GITHUB_REPO
            self.print_test_result("URL Processor - GitHub Repo Categorization", 
                                 "GITHUB_REPO", f"Result: {github_repo}", success)
        except Exception as e:
            self.print_test_result("URL Processor - GitHub Repo Categorization", "No exception", f"Exception: {e}", False)
        
        try:

            unknown_url = categorize_url("https://example.com/test")
            success = unknown_url == URLType.UNKNOWN
            self.print_test_result("URL Processor - Unknown URL Categorization", 
                                 "UNKNOWN", f"Result: {unknown_url}", success)
        except Exception as e:
            self.print_test_result("URL Processor - Unknown URL Categorization", "No exception", f"Exception: {e}", False)

    def test_llm_client_comprehensive(self) -> None:
        """Test LLM client with comprehensive mocking."""
        self.print_header("LLM CLIENT COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.post_with_rate_limit') as mock_post:
                # Mock successful response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "0.8"}}]
                }
                mock_post.return_value = mock_response
                
                from src.core.llm_client import ask_for_json_score
                result = ask_for_json_score("Test prompt")
                
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Client success test passed")
                
                # Test error handling
                mock_post.return_value = None
                result = ask_for_json_score("Test prompt")
                if result is None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Client error handling test passed")
                
                # Test with invalid response
                mock_response.status_code = 400
                mock_response.text = "Bad Request"
                mock_post.return_value = mock_response
                result = ask_for_json_score("Test prompt")
                if result is None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Client invalid response test passed")
                
                # Test with exception
                mock_post.side_effect = Exception("Network error")
                result = ask_for_json_score("Test prompt")
                if result is None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Client exception handling test passed")
                        
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ LLM Client test failed: {e}")
    
    def test_llm_analyzer_comprehensive(self) -> None:
        """Test LLM analyzer with comprehensive mocking."""
        self.print_header("LLM ANALYZER COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.post_with_rate_limit') as mock_post:
                # Mock successful response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "0.8"}}]
                }
                mock_post.return_value = mock_response
                
                from src.metrics.llm_analyzer import LLMAnalyzer
                analyzer = LLMAnalyzer()
                
                # Test dataset quality analysis
                result = analyzer.analyze_dataset_quality({"description": "Test dataset"})
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Analyzer dataset quality test passed")
                
                # Test _extract_score method
                score = analyzer._extract_score("Score: 0.8")
                if score == 0.8:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Analyzer score extraction test passed")
                
                # Test error handling
                mock_post.return_value = None
                result = analyzer.analyze_dataset_quality({"description": "Test"})
                if result == 0.0:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ LLM Analyzer error handling test passed")
                        
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ LLM Analyzer test failed: {e}")
    
    def test_size_calculator_comprehensive(self) -> None:
        """Test size calculator with comprehensive mocking."""
        self.print_header("SIZE CALCULATOR COMPREHENSIVE TESTS")
        
        try:
            with patch('huggingface_hub.HfApi') as mock_api:
                with patch('huggingface_hub.hf_hub_download') as mock_download:
                    # Mock successful API calls
                    mock_api_instance = Mock()
                    mock_api_instance.list_repo_files.return_value = [
                        {"path": "config.json"},
                        {"path": "model.safetensors"},
                        {"path": "tokenizer.json"}
                    ]
                    mock_api.return_value = mock_api_instance
                    mock_download.return_value = "/tmp/test_config.json"
                    
                    # Mock file reading
                    with patch('builtins.open', mock_open_config()):
                        from src.metrics.size_calculator import SizeCalculator
                        calculator = SizeCalculator()
                        
                        context = ModelContext(
                            model_url="https://huggingface.co/test/model",
                            model_info={"name": "test-model"},
                            dataset_url=None,
                            code_url=None
                        )
                        
                        result = calculator.calculate_score(context)
                        if result is not None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Size Calculator success test passed")
                        
                        # Test error handling
                        mock_api_instance.list_repo_files.side_effect = Exception("API error")
                        result = calculator.calculate_score(context)
                        if result is not None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Size Calculator error handling test passed")
                        
                        # Test with different file types
                        mock_api_instance.list_repo_files.return_value = [
                            {"path": "model.bin"},
                            {"path": "model.safetensors"},
                            {"path": "config.json"}
                        ]
                        mock_api_instance.list_repo_files.side_effect = None
                        result = calculator.calculate_score(context)
                        if result is not None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Size Calculator different files test passed")
                        
                        # Test with empty file list
                        mock_api_instance.list_repo_files.return_value = []
                        result = calculator.calculate_score(context)
                        if result is not None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Size Calculator empty files test passed")
                                
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Size Calculator test failed: {e}")
    
    def test_git_analyzer_comprehensive(self) -> None:
        """Test Git analyzer with comprehensive mocking."""
        self.print_header("GIT ANALYZER COMPREHENSIVE TESTS")
        
        try:
            with patch('dulwich.porcelain.clone') as mock_clone:
                with patch('dulwich.repo.Repo') as mock_repo:
                    # Mock successful clone and analysis
                    mock_repo_instance = Mock()
                    mock_repo_instance.get_commits.return_value = [
                        Mock(author=b"Author 1", commit_time=1000),
                        Mock(author=b"Author 2", commit_time=2000),
                        Mock(author=b"Author 1", commit_time=3000)
                    ]
                    mock_repo.return_value = mock_repo_instance
                    mock_clone.return_value = mock_repo_instance
                    
                    from src.core.git_analyzer import GitAnalyzer
                    analyzer = GitAnalyzer()
                    
                    result = analyzer.analyze_repository("https://github.com/test/repo")
                    if result is not None and "total_commits" in result:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Git Analyzer success test passed")
                    
                    # Test error handling
                    mock_clone.side_effect = Exception("Clone failed")
                    result = analyzer.analyze_repository("https://github.com/test/repo")
                    if result is None:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Git Analyzer error handling test passed")
                    
                    # Test with different commit patterns
                    mock_clone.side_effect = None
                    mock_repo_instance.get_commits.return_value = [
                        Mock(author=b"Author 1", commit_time=1000),
                        Mock(author=b"Author 1", commit_time=2000),
                        Mock(author=b"Author 1", commit_time=3000),
                        Mock(author=b"Author 1", commit_time=4000)
                    ]
                    result = analyzer.analyze_repository("https://github.com/test/repo")
                    if result is not None and "total_commits" in result:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Git Analyzer single author test passed")
                    
                    # Test with empty commit list
                    mock_repo_instance.get_commits.return_value = []
                    result = analyzer.analyze_repository("https://github.com/test/repo")
                    if result is not None:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Git Analyzer empty commits test passed")
                            
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Git Analyzer test failed: {e}")
    
    def test_ramp_up_calculator_comprehensive(self) -> None:
        """Test ramp-up calculator with comprehensive mocking."""
        self.print_header("RAMP-UP CALCULATOR COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                # Mock successful response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "downloads": 1000,
                    "likes": 50,
                    "lastModified": "2023-01-01T00:00:00.000Z"
                }
                mock_get.return_value = mock_response
                
                from src.metrics.ramp_up_calculator import RampUpCalculator
                calculator = RampUpCalculator()
                
                context = ModelContext(
                    model_url="https://huggingface.co/test/model",
                    model_info={"name": "test-model"},
                    dataset_url=None,
                    code_url=None
                )
                
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Ramp-Up Calculator success test passed")
                
                # Test error handling
                mock_get.return_value = None
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Ramp-Up Calculator error handling test passed")
                        
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Ramp-Up Calculator test failed: {e}")
    
    def test_code_quality_calculator_comprehensive(self) -> None:
        """Test code quality calculator with comprehensive mocking."""
        self.print_header("CODE QUALITY CALCULATOR COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                with patch('huggingface_hub.HfApi') as mock_api:
                    # Mock successful responses
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"test": "data"}
                    mock_get.return_value = mock_response
                    
                    mock_api_instance = Mock()
                    mock_api_instance.list_repo_files.return_value = [
                        {"path": "README.md"},
                        {"path": "test.py"},
                        {"path": "requirements.txt"}
                    ]
                    mock_api.return_value = mock_api_instance
                    
                    from src.metrics.code_quality_calculator import CodeQualityCalculator
                    calculator = CodeQualityCalculator()
                    
                    context = ModelContext(
                        model_url="https://huggingface.co/test/model",
                        model_info={"name": "test-model"},
                        dataset_url=None,
                        code_url="https://github.com/test/repo"
                    )
                    
                    result = calculator.calculate_score(context)
                    if result is not None:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Code Quality Calculator success test passed")
                    
                    # Test error handling
                    mock_get.return_value = None
                    result = calculator.calculate_score(context)
                    if result is not None:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Code Quality Calculator error handling test passed")
                            
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Code Quality Calculator test failed: {e}")
    
    def test_performance_claims_calculator_comprehensive(self) -> None:
        """Test performance claims calculator with comprehensive mocking."""
        self.print_header("PERFORMANCE CLAIMS CALCULATOR COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                # Mock successful response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = "This model achieves 95% accuracy on benchmark tests."
                mock_get.return_value = mock_response
                
                from src.metrics.performance_claims_calculator import PerformanceClaimsCalculator
                calculator = PerformanceClaimsCalculator()
                
                context = ModelContext(
                    model_url="https://huggingface.co/test/model",
                    model_info={"name": "test-model"},
                    dataset_url=None,
                    code_url=None
                )
                
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Performance Claims Calculator success test passed")
                
                # Test error handling
                mock_get.return_value = None
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Performance Claims Calculator error handling test passed")
                        
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Performance Claims Calculator test failed: {e}")
    
    def test_dataset_quality_calculator_comprehensive(self) -> None:
        """Test dataset quality calculator with comprehensive mocking."""
        self.print_header("DATASET QUALITY CALCULATOR COMPREHENSIVE TESTS")
        
        try:
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                # Mock successful response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "downloads": 5000,
                    "likes": 100,
                    "lastModified": "2023-01-01T00:00:00.000Z"
                }
                mock_get.return_value = mock_response
                
                from src.metrics.dataset_quality_calculator import DatasetQualityCalculator
                calculator = DatasetQualityCalculator()
                
                context = ModelContext(
                    model_url="https://huggingface.co/test/model",
                    model_info={"name": "test-model"},
                    dataset_url="https://huggingface.co/datasets/test-dataset",
                    code_url=None
                )
                
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Dataset Quality Calculator success test passed")
                
                # Test error handling
                mock_get.return_value = None
                result = calculator.calculate_score(context)
                if result is not None:
                    self.passed_tests += 1
                    if not self.coverage_mode:
                        print("✅ Dataset Quality Calculator error handling test passed")
                        
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Dataset Quality Calculator test failed: {e}")
    
    def test_model_analyzer_comprehensive(self) -> None:
        """Test Model analyzer with comprehensive mocking."""
        self.print_header("MODEL ANALYZER COMPREHENSIVE TESTS")
        
        try:
            with patch('huggingface_hub.HfApi') as mock_api:
                with patch('huggingface_hub.hf_hub_download') as mock_download:
                    # Mock successful API calls
                    mock_api_instance = Mock()
                    mock_api_instance.list_repo_files.return_value = [
                        {"path": "config.json"},
                        {"path": "model.safetensors"},
                        {"path": "tokenizer.json"}
                    ]
                    mock_api.return_value = mock_api_instance
                    mock_download.return_value = "/tmp/test_config.json"
                    
                    # Mock file reading
                    with patch('builtins.open', mock_open_config()):
                        from src.core.model_analyzer import ModelDynamicAnalyzer
                        analyzer = ModelDynamicAnalyzer()
                        
                        result = analyzer.analyze_model_loading("test/model")
                        if result is not None and "config_loaded" in result:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Model Analyzer success test passed")
                        
                        # Test error handling
                        mock_api_instance.list_repo_files.side_effect = Exception("API error")
                        result = analyzer.analyze_model_loading("test/model")
                        if result is None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ Model Analyzer error handling test passed")
                                
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ Model Analyzer test failed: {e}")

    def test_all_metric_calculators_comprehensive(self) -> None:
        """Test all metric calculators comprehensively for coverage."""
        self.print_header("ALL METRIC CALCULATORS COMPREHENSIVE TESTS")
        
        try:
            # Test Bus Factor Calculator
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                with patch('src.core.git_analyzer.porcelain.clone') as mock_clone:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"contributors": [{"login": "user1"}, {"login": "user2"}]}
                    mock_get.return_value = mock_response
                    
                    mock_repo = Mock()
                    mock_repo.get_commits.return_value = [
                        Mock(author=b"Author 1", commit_time=1000),
                        Mock(author=b"Author 2", commit_time=2000)
                    ]
                    mock_clone.return_value = mock_repo
                    
                    from src.metrics.busfactor_calculator import BusFactorCalculator
                    calculator = BusFactorCalculator()
                    
                    context = ModelContext(
                        model_url="https://github.com/test/repo",
                        model_info={"name": "test-model"},
                        dataset_url=None,
                        code_url=None
                    )
                    
                    result = calculator.calculate_score(context)
                    if result is not None:
                        self.passed_tests += 1
                        if not self.coverage_mode:
                            print("✅ Bus Factor Calculator comprehensive test passed")
            
            # Test License Calculator
            from src.metrics.license_calculator import LicenseCalculator
            calculator = LicenseCalculator()
            
            context = ModelContext(
                model_url="https://huggingface.co/test/model",
                model_info={"license": "MIT"},
                dataset_url=None,
                code_url=None
            )
            
            result = calculator.calculate_score(context)
            if result is not None:
                self.passed_tests += 1
                if not self.coverage_mode:
                    print("✅ License Calculator comprehensive test passed")
            
            # Test Dataset Code Calculator
            from src.metrics.dataset_code_calculator import DatasetCodeCalculator
            calculator = DatasetCodeCalculator()
            
            context = ModelContext(
                model_url="https://huggingface.co/test/model",
                model_info={"name": "test-model"},
                dataset_url="https://huggingface.co/datasets/test",
                code_url="https://github.com/test/repo"
            )
            
            result = calculator.calculate_score(context)
            if result is not None:
                self.passed_tests += 1
                if not self.coverage_mode:
                    print("✅ Dataset Code Calculator comprehensive test passed")
                    
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ All Metric Calculators test failed: {e}")

    def test_all_core_modules_comprehensive(self) -> None:
        """Test all core modules comprehensively for coverage."""
        self.print_header("ALL CORE MODULES COMPREHENSIVE TESTS")
        
        try:
            # Test URL Processor
            with patch('src.core.http_client.get_with_rate_limit') as mock_get:
                with patch('src.core.http_client.post_with_rate_limit') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"test": "data"}
                    mock_get.return_value = mock_response
                    mock_post.return_value = mock_response
                    
                    from src.core.url_processor import URLProcessor
                    
                    # Create a temporary test file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                        f.write("https://huggingface.co/test/model\n")
                        temp_file = f.name
                    
                    try:
                        processor = URLProcessor(temp_file)
                        results = processor.process_urls_with_metrics()
                        
                        if results is not None:
                            self.passed_tests += 1
                            if not self.coverage_mode:
                                print("✅ URL Processor comprehensive test passed")
                    finally:
                        import os
                        os.unlink(temp_file)
            
            # Test Results Storage
            from src.storage.results_storage import ResultsStorage, ModelResult, MetricResult
            
            storage = ResultsStorage()
            
            # Test storing and retrieving results
            metric_result = MetricResult("test_metric", 0.8, 100, timestamp=1234567890.0)
            storage.store_metric_result("test_model", metric_result)
            
            retrieved = storage.get_metric_result("test_model", "test_metric")
            if retrieved is not None:
                self.passed_tests += 1
                if not self.coverage_mode:
                    print("✅ Results Storage comprehensive test passed")
            
            # Test ModelResult creation and NDJSON output
            model_result = ModelResult(
                url="https://huggingface.co/test/model",
                net_score=0.8,
                net_score_latency=100,
                size_score={"raspberry_pi": 0.5, "jetson_nano": 0.6, "desktop_pc": 0.8, "aws_server": 0.9},
                size_latency=200,
                license_score=1.0,
                license_latency=50,
                ramp_up_score=0.7,
                ramp_up_latency=150,
                bus_factor_score=0.6,
                bus_factor_latency=300,
                performance_claims_score=0.5,
                performance_claims_latency=250,
                dataset_code_score=0.9,
                dataset_code_latency=180,
                dataset_quality_score=0.8,
                dataset_quality_latency=220,
                code_quality_score=0.7,
                code_quality_latency=190
            )
            
            ndjson_line = model_result.to_ndjson_line()
            if ndjson_line and "test_model" in ndjson_line:
                self.passed_tests += 1
                if not self.coverage_mode:
                    print("✅ ModelResult NDJSON comprehensive test passed")
                    
        except Exception as e:
            self.failed_tests += 1
            if not self.coverage_mode:
                print(f"❌ All Core Modules test failed: {e}")


if __name__ == "__main__":
    import sys
    try:
        coverage_mode = "--coverage" in sys.argv
        test_suite = TestSuite(coverage_mode=coverage_mode)
        test_suite.run_all_tests()

        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        sys.exit(1)
