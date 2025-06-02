#!/usr/bin/env python3
"""
Test Enhanced Benchmarking System

Tests for the enhanced benchmarking system that evaluates model performance
with cosmetic procedure questions through API endpoints.
"""
import pytest
import os
import sys
import time
import tempfile
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beautyai_inference.benchmarks.enhanced_benchmarking import (
    EnhancedBenchmarkingService,
    BenchmarkConfig,
    BenchmarkResult,
    ModelPerformanceMetrics
)
from beautyai_inference.benchmarks.analyze_results import BenchmarkAnalyzer
from beautyai_inference.benchmarks.demo_benchmarking import DemoBenchmarkingService


class TestBenchmarkConfig:
    """Test benchmark configuration functionality."""
    
    def test_default_config_creation(self):
        """Test that default configuration is created correctly."""
        config = BenchmarkConfig()
        
        # Verify default values
        assert config.base_url == "http://localhost:8000"
        assert config.test_questions_count == 50
        assert config.concurrent_requests == 1
        assert config.timeout_seconds == 30
        assert config.include_content_filter_tests is True
        assert config.export_format == "both"
        assert config.output_directory == "./benchmark_results"
    
    def test_custom_config_creation(self):
        """Test creation of custom configuration."""
        custom_config = BenchmarkConfig(
            base_url="http://test.example.com",
            test_questions_count=100,
            concurrent_requests=5,
            timeout_seconds=60,
            include_content_filter_tests=False,
            export_format="json",
            output_directory="/tmp/benchmarks"
        )
        
        assert custom_config.base_url == "http://test.example.com"
        assert custom_config.test_questions_count == 100
        assert custom_config.concurrent_requests == 5
        assert custom_config.timeout_seconds == 60
        assert custom_config.include_content_filter_tests is False
        assert custom_config.export_format == "json"
        assert custom_config.output_directory == "/tmp/benchmarks"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid values
        with pytest.raises(ValueError):
            BenchmarkConfig(test_questions_count=0)
        
        with pytest.raises(ValueError):
            BenchmarkConfig(concurrent_requests=0)
        
        with pytest.raises(ValueError):
            BenchmarkConfig(timeout_seconds=-1)


class TestBenchmarkResult:
    """Test benchmark result data structures."""
    
    def test_benchmark_result_creation(self):
        """Test creation of benchmark results."""
        metrics = ModelPerformanceMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_response_time=1.5,
            min_response_time=0.8,
            max_response_time=3.2,
            requests_per_second=10.5,
            content_filter_blocks=25,
            content_filter_rate=0.25,
            error_rate=0.05
        )
        
        result = BenchmarkResult(
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            config=BenchmarkConfig(),
            performance_metrics=metrics,
            sample_responses=[],
            errors=[]
        )
        
        assert result.model_name == "test-model"
        assert result.performance_metrics.total_requests == 100
        assert result.performance_metrics.successful_requests == 95
        assert result.performance_metrics.error_rate == 0.05
    
    def test_metrics_calculations(self):
        """Test that metrics are calculated correctly."""
        metrics = ModelPerformanceMetrics(
            total_requests=100,
            successful_requests=80,
            failed_requests=20,
            avg_response_time=2.0,
            min_response_time=1.0,
            max_response_time=5.0,
            requests_per_second=8.0,
            content_filter_blocks=30,
            content_filter_rate=0.3,
            error_rate=0.2
        )
        
        # Verify consistency
        assert metrics.total_requests == metrics.successful_requests + metrics.failed_requests
        assert metrics.error_rate == metrics.failed_requests / metrics.total_requests
        assert 0.0 <= metrics.content_filter_rate <= 1.0
        assert metrics.avg_response_time > 0


class TestEnhancedBenchmarkingService:
    """Test the main benchmarking service."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a test configuration."""
        return BenchmarkConfig(
            base_url="http://localhost:8000",
            test_questions_count=10,
            concurrent_requests=1,
            timeout_seconds=5,
            output_directory=tempfile.mkdtemp()
        )
    
    @pytest.fixture
    def benchmarking_service(self, mock_config):
        """Create benchmarking service with test config."""
        return EnhancedBenchmarkingService(mock_config)
    
    def test_service_initialization(self, benchmarking_service, mock_config):
        """Test that service initializes correctly."""
        assert benchmarking_service.config == mock_config
        assert benchmarking_service.session is not None
        assert benchmarking_service.results == []
    
    @patch('requests.Session.get')
    def test_api_health_check(self, mock_get, benchmarking_service):
        """Test API health check functionality."""
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        
        is_healthy = benchmarking_service._check_api_health()
        assert is_healthy is True
        
        # Mock failed health check
        mock_get.side_effect = Exception("Connection failed")
        is_healthy = benchmarking_service._check_api_health()
        assert is_healthy is False
    
    @patch('requests.Session.get')
    def test_get_available_models(self, mock_get, benchmarking_service):
        """Test getting available models from API."""
        # Mock successful models response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": ["model1", "model2", "model3"]
        }
        mock_get.return_value = mock_response
        
        models = benchmarking_service._get_available_models()
        assert models == ["model1", "model2", "model3"]
        
        # Mock failed response
        mock_get.side_effect = Exception("API failed")
        models = benchmarking_service._get_available_models()
        assert models == []
    
    def test_load_test_questions(self, benchmarking_service):
        """Test loading test questions from CSV."""
        questions = benchmarking_service._load_test_questions()
        
        # Should load questions (may be empty if CSV doesn't exist in test env)
        assert isinstance(questions, list)
        
        # If questions are loaded, verify format
        if questions:
            for question in questions[:5]:  # Check first 5
                assert isinstance(question, str)
                assert len(question.strip()) > 0
    
    def test_question_sampling(self, benchmarking_service):
        """Test question sampling logic."""
        # Test with mock questions
        mock_questions = [f"Question {i}" for i in range(100)]
        
        # Test sampling smaller subset
        with patch.object(benchmarking_service, '_load_test_questions', return_value=mock_questions):
            benchmarking_service.config.test_questions_count = 20
            sampled = benchmarking_service._load_test_questions()
            
            # Should return all questions (actual sampling happens in benchmark logic)
            assert len(sampled) == 100
    
    @patch('requests.Session.post')
    def test_single_request_benchmark(self, mock_post, benchmarking_service):
        """Test benchmarking a single request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "filtered": False
        }
        mock_response.elapsed.total_seconds.return_value = 1.5
        mock_post.return_value = mock_response
        
        start_time = time.time()
        result = benchmarking_service._benchmark_single_request(
            "test-model", "Test question", start_time
        )
        
        assert result is not None
        assert "response_time" in result
        assert "success" in result
        assert result["success"] is True
    
    def test_results_export_json(self, benchmarking_service):
        """Test exporting results to JSON format."""
        # Create mock result
        mock_metrics = ModelPerformanceMetrics(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            avg_response_time=1.5,
            min_response_time=1.0,
            max_response_time=2.0,
            requests_per_second=5.0,
            content_filter_blocks=3,
            content_filter_rate=0.3,
            error_rate=0.2
        )
        
        mock_result = BenchmarkResult(
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            config=benchmarking_service.config,
            performance_metrics=mock_metrics,
            sample_responses=[],
            errors=[]
        )
        
        benchmarking_service.results = [mock_result]
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            benchmarking_service._export_results_json(filename)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(filename)
            with open(filename, 'r') as f:
                data = json.load(f)
                assert "results" in data
                assert len(data["results"]) == 1
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestBenchmarkAnalyzer:
    """Test the benchmark results analyzer."""
    
    @pytest.fixture
    def sample_results_file(self):
        """Create a sample results file for testing."""
        sample_data = {
            "metadata": {
                "benchmark_version": "1.0.0",
                "total_models_tested": 2,
                "timestamp": "2024-01-01T00:00:00Z"
            },
            "results": [
                {
                    "model_name": "model1",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "performance_metrics": {
                        "total_requests": 100,
                        "successful_requests": 90,
                        "failed_requests": 10,
                        "avg_response_time": 1.5,
                        "min_response_time": 1.0,
                        "max_response_time": 3.0,
                        "requests_per_second": 10.0,
                        "content_filter_blocks": 25,
                        "content_filter_rate": 0.25,
                        "error_rate": 0.1
                    }
                },
                {
                    "model_name": "model2", 
                    "timestamp": "2024-01-01T00:00:00Z",
                    "performance_metrics": {
                        "total_requests": 100,
                        "successful_requests": 95,
                        "failed_requests": 5,
                        "avg_response_time": 2.0,
                        "min_response_time": 1.2,
                        "max_response_time": 4.0,
                        "requests_per_second": 8.0,
                        "content_filter_blocks": 30,
                        "content_filter_rate": 0.3,
                        "error_rate": 0.05
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            return f.name
    
    def test_analyzer_initialization(self, sample_results_file):
        """Test analyzer initialization."""
        analyzer = BenchmarkAnalyzer(sample_results_file)
        assert analyzer.results_file == sample_results_file
        assert analyzer.data is not None
    
    def test_load_results(self, sample_results_file):
        """Test loading results from file."""
        analyzer = BenchmarkAnalyzer(sample_results_file)
        analyzer.load_results()
        
        assert "metadata" in analyzer.data
        assert "results" in analyzer.data
        assert len(analyzer.data["results"]) == 2
    
    def test_performance_comparison(self, sample_results_file):
        """Test performance comparison analysis."""
        analyzer = BenchmarkAnalyzer(sample_results_file)
        analyzer.load_results()
        
        comparison = analyzer.generate_performance_comparison()
        
        assert "summary" in comparison
        assert "model_rankings" in comparison
        assert "detailed_comparison" in comparison
        
        # Verify model rankings
        rankings = comparison["model_rankings"]
        assert len(rankings) == 2
        
        # Best performing model should be ranked first (lowest error rate)
        assert rankings[0]["model_name"] == "model2"  # Lower error rate (0.05 vs 0.1)
    
    def test_statistics_generation(self, sample_results_file):
        """Test statistics generation."""
        analyzer = BenchmarkAnalyzer(sample_results_file)
        analyzer.load_results()
        
        stats = analyzer.generate_statistics()
        
        required_stats = [
            "total_models",
            "avg_response_time", 
            "avg_success_rate",
            "avg_filter_rate",
            "best_performing_model",
            "fastest_model"
        ]
        
        for stat in required_stats:
            assert stat in stats
    
    def tearDown(self, sample_results_file):
        """Clean up test files."""
        if os.path.exists(sample_results_file):
            os.unlink(sample_results_file)


class TestDemoBenchmarkingService:
    """Test the demo benchmarking functionality."""
    
    @patch('requests.Session.get')
    @patch('requests.Session.post')
    def test_demo_quick_test(self, mock_post, mock_get):
        """Test demo quick test functionality."""
        # Mock API health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_health_response
        
        # Mock inference response
        mock_inference_response = Mock()
        mock_inference_response.status_code = 200
        mock_inference_response.json.return_value = {
            "response": "Test response", 
            "filtered": False
        }
        mock_inference_response.elapsed.total_seconds.return_value = 1.0
        mock_post.return_value = mock_inference_response
        
        demo_service = DemoBenchmarkingService()
        
        # Test quick demo
        with patch.object(demo_service, '_load_sample_questions', 
                         return_value=["Test question 1", "Test question 2"]):
            results = demo_service.run_quick_demo("test-model")
            
            assert results is not None
            assert "model_name" in results
            assert "test_results" in results


class TestIntegration:
    """Integration tests for the benchmarking system."""
    
    def test_end_to_end_workflow(self):
        """Test complete benchmarking workflow."""
        # Test that all components can be imported and initialized
        try:
            config = BenchmarkConfig(test_questions_count=5)
            service = EnhancedBenchmarkingService(config)
            demo_service = DemoBenchmarkingService()
            
            assert config is not None
            assert service is not None  
            assert demo_service is not None
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def test_file_system_operations(self):
        """Test file system operations work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(output_directory=temp_dir)
            service = EnhancedBenchmarkingService(config)
            
            # Verify output directory is set correctly
            assert service.config.output_directory == temp_dir
            
            # Test that we can create files in the directory
            test_file = os.path.join(temp_dir, "test.json")
            with open(test_file, 'w') as f:
                json.dump({"test": "data"}, f)
            
            assert os.path.exists(test_file)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
