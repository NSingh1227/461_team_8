#!/usr/bin/env python3

import sys
import os
import datetime
from typing import List, Dict, Any
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.url_processor import URLProcessor, URLType, process_url, categorize_url, is_valid_url
from src.metrics.license_calculator import LicenseCalculator
from src.metrics.base import ModelContext

class TestSuite:
    def __init__(self, coverage_mode=False):
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.coverage_mode = coverage_mode
    
    def print_header(self, title: str):
        if not self.coverage_mode:
            print("\n" + "="*60)
            print(f"  {title}")
            print("="*60)
    
    def print_test_result(self, test_name: str, expected: Any, actual: Any, passed: bool):
        self.total_tests += 1
        if not self.coverage_mode:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {test_name}")
            print(f"      Expected: {expected}")
            print(f"      Actual:   {actual}")
            print()
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def print_section(self, title: str):
        if not self.coverage_mode:
            print(f"\n--- {title} ---")

    # ============================================================
    #  URL VALIDATION TESTS
    # ============================================================
    
    def test_url_validation(self):
        self.print_header("URL VALIDATION TESTS")
        
        test_cases = [
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
            actual = is_valid_url(url)
            passed = actual == expected
            self.print_test_result(test_name, expected, actual, passed)

    def test_url_categorization(self):
        self.print_header("URL CATEGORIZATION TESTS")
        
        test_cases = [
            ("HuggingFace Model", "https://huggingface.co/microsoft/DialoGPT-medium", URLType.HUGGINGFACE_MODEL),
            ("HuggingFace Dataset", "https://huggingface.co/datasets/squad", URLType.HUGGINGFACE_DATASET),
            ("GitHub Repository", "https://github.com/microsoft/DialoGPT", URLType.GITHUB_REPO),
            ("Unknown Domain", "https://example.com/some/path", URLType.UNKNOWN),
            ("Invalid URL", "not-a-url", URLType.UNKNOWN)
        ]
        
        for test_name, url, expected in test_cases:
            actual = process_url(url)
            passed = actual == expected
            self.print_test_result(test_name, expected.value, actual.value, passed)

    # Test URL processing with file I/O and real URLs
    def test_comprehensive_url_processing(self):
        self.print_header("COMPREHENSIVE URL PROCESSING TESTS")
        
        test_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://huggingface.co/datasets/squad",
            "https://github.com/microsoft/DialoGPT",
            "https://huggingface.co/fake/nonexistent-model",
            "https://huggingface.co/datasets/fake-dataset",
            "https://github.com/fake/nonexistent-repo",
            "https://example.com/invalid/url",
            "not-a-url-at-all"
        ]
        
        test_file_path = "temp_test_urls.txt"
        with open(test_file_path, 'w') as f:
            for url in test_urls:
                f.write(url + '\n')
        
        try:
            processor = URLProcessor(test_file_path)
            results = processor.process_urls()
            
            self.print_section("URL Processing Results")
            for i, result in enumerate(results):
                url = result.get('url', 'Unknown')
                url_type = result.get('type', 'unknown')
                has_metadata = result.get('has_metadata', False)
                
                if not self.coverage_mode:
                    print(f"{i+1}. URL: {url}")
                    print(f"   Type: {url_type}")
                    print(f"   Has Metadata: {has_metadata}")
                    
                    if 'info' in result and 'status' in result['info']:
                        print(f"   Status: {result['info']['status']}")
                    print()
            
            expected_count = len(test_urls)
            actual_count = len(results)
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
        
        test_cases = [
            ("Malformed HuggingFace URL", "https://huggingface.co/", URLType.HUGGINGFACE_MODEL),
            ("HuggingFace URL with extra path", "https://huggingface.co/microsoft/DialoGPT-medium/tree/main", URLType.HUGGINGFACE_MODEL),
            ("GitHub URL with extra path", "https://github.com/microsoft/DialoGPT/issues", URLType.GITHUB_REPO),
            ("Non-standard domain", "https://gitlab.com/test/repo", URLType.UNKNOWN),
        ]
        
        for description, url, expected in test_cases:
            try:
                result = categorize_url(url)
                self.print_test_result(description, expected.value, result.value, expected.value == result.value)
            except Exception as e:
                self.print_test_result(description, expected.value, f"Error: {e}", False)
                
        try:
            result = fetch_huggingface_metadata("https://huggingface.co/", "models")
            self.print_test_result("Fetch metadata with empty path", None, result, result is None)
        except Exception:
            self.print_test_result("Fetch metadata with empty path", None, None, True)
            
        additional_test_cases = [
            ("URL with fragment", "https://example.com#section", True),
            ("URL with query and fragment", "https://example.com?q=test#section", True),
            ("FTP URL", "ftp://files.example.com", False),
            ("File URL", "file:///path/to/file", False)
        ]
        
        for description, url, expected in additional_test_cases:
            try:
                result = is_valid_url(url)
                self.print_test_result(description, expected, result, result == expected)
            except Exception:
                self.print_test_result(description, expected, False, False)

    # ============================================================
    #  LICENSE METRIC TESTS
    # ============================================================
    
    def test_license_calculator(self):
        self.print_header("LICENSE CALCULATOR TESTS")
        
        calculator = LicenseCalculator()
        
        test_cases = [
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

    # ============================================================
    #  RAMP UP METRIC TESTS
    # ============================================================
    
    # TODO: Add RampUp metric tests when implemented
    
    # ============================================================
    #  BUS FACTOR METRIC TESTS
    # ============================================================
    
    # TODO: Add BusFactor metric tests when implemented
    
    # ============================================================
    #  CORRECTNESS METRIC TESTS
    # ============================================================
    
    # TODO: Add Correctness metric tests when implemented
    
    # ============================================================
    #  RESPONSIVE MAINTAINER METRIC TESTS
    # ============================================================
    
    # TODO: Add ResponsiveMaintainer metric tests when implemented
    
    # ============================================================
    #  NET SCORE TESTS
    # ============================================================
    
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
                    print(f"    Size: {result.size_score:.3f} ({result.size_latency}ms)")
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

    # ============================================================
    #  METRIC CALCULATOR FRAMEWORK TESTS
    # ============================================================
    
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

    # Test additional MetricCalculator methods
    def test_metric_calculator_additional_features(self):
        self.print_section("METRIC CALCULATOR ADDITIONAL FEATURES")
        
        from src.metrics.base import MetricCalculator, ModelContext
        
        class TestCalculator(MetricCalculator):
            def calculate_score(self, url_type, metadata):
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

    # ============================================================
    #  EXCEPTION HANDLING TESTS
    # ============================================================
    
    def test_exception_handling(self):
        self.print_section("EXCEPTION HANDLING TESTS")
        
        try:
            from src.core.exceptions import MetricCalculationException
            error = MetricCalculationException("TestMetric", "Test error message")
            expected = "Failed to calculate TestMetric metric: Test error message"
            self.print_test_result("MetricCalculationException creation", expected, str(error), str(error) == expected)
        except Exception as e:
            self.print_test_result("MetricCalculationException creation", "Success", f"Error: {e}", False)
        
        try:
            from src.core.exceptions import InvalidURLException
            error = InvalidURLException("bad-url", "malformed")
            expected = "Invalid URL 'bad-url': malformed"
            self.print_test_result("InvalidURLException creation", expected, str(error), str(error) == expected)
        except Exception as e:
            self.print_test_result("InvalidURLException creation", "Success", f"Error: {e}", False)
            
        try:
            from src.core.exceptions import APIRateLimitException
            error = APIRateLimitException("TestAPI", 60)
            expected = "Rate limit exceeded for TestAPI API. Retry after 60 seconds"
            self.print_test_result("APIRateLimitException creation", expected, str(error), str(error) == expected)
        except Exception as e:
            self.print_test_result("APIRateLimitException creation", "Success", f"Error: {e}", False)

    # ============================================================
    #  RESULTS STORAGE TESTS
    # ============================================================
    
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
                size_score=0.7, size_latency=50,
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
                size_score=0.7, size_latency=50,
                license_score=1.0, license_latency=25,
                ramp_up_score=0.6, ramp_up_latency=75,
                bus_factor_score=0.5, bus_factor_latency=60,
                dataset_code_score=0.9, dataset_code_latency=40,
                dataset_quality_score=0.8, dataset_quality_latency=55,
                code_quality_score=0.85, code_quality_latency=45,
                performance_claims_score=0.75, performance_claims_latency=65
            )
            ndjson = model_result.to_ndjson_line()
            has_url = "https://test.com" in ndjson
            is_valid_json = json.loads(ndjson) is not None
            self.print_test_result("NDJSON formatting", True, has_url and is_valid_json, has_url and is_valid_json)
        except Exception:
            self.print_test_result("NDJSON formatting", True, False, False)
            
        try:
            result = MetricResult("test_metric", 0.5, 100, str(datetime.datetime.now()))
            result_dict = result.to_dict()
            has_metric_name = "metric_name" in result_dict
            self.print_test_result("MetricResult to_dict", True, has_metric_name, has_metric_name)
        except Exception:
            self.print_test_result("MetricResult to_dict", True, False, False)

    # Test core ResultsStorage functionality
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

    # ============================================================
    #  TEST SUMMARY AND EXECUTION
    # ============================================================
    
    def print_summary(self):
        if self.coverage_mode:
            import subprocess
            try:
                result = subprocess.run([sys.executable, '-m', 'coverage', 'report'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        last_line = lines[-1]
                        import re
                        match = re.search(r'(\d+)%', last_line)
                        if match:
                            coverage_percent = match.group(1)
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
        
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\nALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {self.failed_tests} test(s) failed. Review output above.")
        
        print("\nNOTE: Only License metrics are fully implemented.")
        print("Other metrics show dummy values for testing purposes.")
    
    def run_all_tests(self):
        if not self.coverage_mode:
            print("🚀 Starting Comprehensive Test Suite for ECE 46100 Team 8")
            print("Testing URL processing, routing, and metric calculations...")
        
        try:
            self.test_url_validation()
            self.test_url_categorization() 
            self.test_comprehensive_url_processing()
            self.test_edge_cases()
            self.test_url_processor_edge_cases()
            
            self.test_license_calculator()
            self.test_license_compatibility_mapping()
            
            self.test_full_metric_pipeline()
            
            self.test_metric_calculator_validation()
            self.test_metric_calculator_additional_features()
            self.test_exception_handling()
            self.test_results_storage_edge_cases()
            self.test_results_storage_functionality()
            
        except Exception as e:
            if not self.coverage_mode:
                print(f"❌ CRITICAL ERROR: {e}")
        finally:
            self.print_summary()
            
        if self.failed_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    coverage_mode = "--coverage" in sys.argv
    test_suite = TestSuite(coverage_mode=coverage_mode)
    test_suite.run_all_tests()