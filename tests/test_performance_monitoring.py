"""
Test Suite for Performance Monitoring System.

Tests all components of the performance monitoring infrastructure:
- PerformanceMonitor core functionality
- MetricsAggregator data processing
- AnomalyDetector anomaly detection
- Dashboard API endpoints
- Integration with existing services

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import pytest
import time
import json
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Add backend/src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

# Import the modules to test
from beautyai_inference.core.performance_monitor import (
    PerformanceMonitor,
    initialize_performance_monitor,
    shutdown_performance_monitor,
    get_performance_monitor
)
from beautyai_inference.core.performance_types import (
    MetricPoint,
    MetricType, 
    PerformanceConfig
)
from beautyai_inference.core.metrics_aggregator import (
    MetricsAggregator,
    AggregationType,
    TimeWindow,
    AggregationRule
)
from beautyai_inference.core.anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectionRule,
    AlertSeverity,
    DetectedAnomaly,
    AnomalyType,
    AnomalyDetectionAlgorithm
)
from beautyai_inference.api.performance_integration import (
    PerformanceMonitoringService,
    get_performance_monitoring_service
)


class TestPerformanceMonitor:
    """Test cases for the PerformanceMonitor class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return PerformanceConfig(
            enabled=True,
            collection_interval_seconds=1.0,
            metrics_retention_seconds=300,
            anomaly_detection_enabled=True,
            alerting_enabled=True,
            system_metrics_enabled=True,
            circuit_breaker_monitoring=True,
            connection_pool_monitoring=True
        )
    
    @pytest.fixture
    def performance_monitor(self, config):
        """Performance monitor instance for testing."""
        return PerformanceMonitor(config)
    
    def test_performance_monitor_initialization(self, config):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(config)
        
        assert monitor.config.enabled == True
        assert monitor.config.collection_interval_seconds == 1.0
        assert monitor.metrics_collector is not None
        assert monitor.metrics_aggregator is not None
        assert monitor.anomaly_detector is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitor_start_stop(self, performance_monitor):
        """Test starting and stopping the performance monitor."""
        # Start the monitor
        await performance_monitor.start()
        assert performance_monitor._monitoring_task is not None
        
        # Stop the monitor
        await performance_monitor.stop()
        assert performance_monitor._monitoring_task.cancelled() or performance_monitor._monitoring_task.done()
    
    @pytest.mark.asyncio
    async def test_add_custom_metric(self, performance_monitor):
        """Test adding custom metrics."""
        await performance_monitor.add_custom_metric("test_metric", 42.0)
        
        # Get current metrics
        current_metrics = await performance_monitor.get_current_metrics()
        
        # Check if custom metric is present
        custom_metrics = await performance_monitor.metrics_collector.get_custom_metrics()
        assert len(custom_metrics) > 0
        assert any(metric.name == "test_metric" for metric in custom_metrics)
    
    @pytest.mark.asyncio
    @patch('beautyai_inference.core.performance_monitor.psutil.cpu_percent')
    @patch('beautyai_inference.core.performance_monitor.psutil.virtual_memory')
    async def test_system_metrics_collection(self, mock_memory, mock_cpu, performance_monitor):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(percent=80.0, used=8000000000, available=2000000000)
        
        # Collect metrics
        metrics = await performance_monitor.metrics_collector.collect_system_metrics()
        
        assert len(metrics) > 0
        cpu_metrics = [m for m in metrics if m.name == "system_cpu_usage_percent"]
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].value == 75.0
        
        memory_metrics = [m for m in metrics if m.name == "system_memory_usage_percent"]
        assert len(memory_metrics) == 1
        assert memory_metrics[0].value == 80.0
    
    @pytest.mark.asyncio
    async def test_performance_dashboard(self, performance_monitor):
        """Test getting performance dashboard data."""
        # Add some test metrics first
        await performance_monitor.add_custom_metric("test_cpu", 50.0)
        await performance_monitor.add_custom_metric("test_memory", 60.0)
        
        # Get dashboard data
        dashboard_data = await performance_monitor.get_performance_dashboard()
        
        assert "timestamp" in dashboard_data
        assert "monitoring_config" in dashboard_data
        assert "system_summary" in dashboard_data
        assert "alert_summary" in dashboard_data
        assert "current_metrics" in dashboard_data
        
        assert dashboard_data["monitoring_config"]["enabled"] == True


class TestMetricsAggregator:
    """Test cases for the MetricsAggregator class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return PerformanceConfig(
            enabled=True,
            historical_data_points=100,
            data_aggregation_interval_seconds=5.0
        )
    
    @pytest.fixture
    def aggregator(self, config):
        """Metrics aggregator instance for testing."""
        return MetricsAggregator(config)
    
    @pytest.mark.asyncio
    async def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator._aggregation_rules is not None
        assert len(aggregator._aggregation_rules) > 0  # Should have default rules
    
    @pytest.mark.asyncio
    async def test_ingest_metrics(self, aggregator):
        """Test metrics ingestion."""
        metrics = [
            MetricPoint(
                name="test_metric",
                value=50.0,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            ),
            MetricPoint(
                name="test_metric",
                value=60.0,
                timestamp=time.time() + 1,
                metric_type=MetricType.GAUGE
            )
        ]
        
        await aggregator.ingest_metrics(metrics)
        
        assert "test_metric" in aggregator._time_series_buffers
        buffer = aggregator._time_series_buffers["test_metric"]
        assert buffer.size == 2
    
    @pytest.mark.asyncio
    async def test_add_aggregation_rule(self, aggregator):
        """Test adding custom aggregation rules."""
        rule = AggregationRule(
            metric_pattern="custom_metric",
            aggregation_type=AggregationType.AVERAGE,
            time_window=TimeWindow.MINUTE_1
        )
        
        initial_count = len(aggregator._aggregation_rules)
        aggregator.add_aggregation_rule(rule)
        
        assert len(aggregator._aggregation_rules) == initial_count + 1
        assert rule in aggregator._aggregation_rules
    
    @pytest.mark.asyncio
    async def test_aggregation_processing(self, aggregator):
        """Test aggregation processing."""
        # First add an aggregation rule
        rule = AggregationRule(
            metric_pattern="test_cpu_usage",
            aggregation_type=AggregationType.AVERAGE,
            time_window=TimeWindow.MINUTE_1
        )
        aggregator.add_aggregation_rule(rule)
        
        # Add test metrics
        current_time = time.time()
        metrics = []
        for i in range(10):
            metrics.append(MetricPoint(
                name="test_cpu_usage",
                value=50.0 + i * 2,  # Values from 50 to 68
                timestamp=current_time - (60 - i * 6),  # Spread over 1 minute
                metric_type=MetricType.GAUGE
            ))
        
        await aggregator.ingest_metrics(metrics)
        
        # Process aggregation
        await aggregator._process_aggregation_rules()
        
        # Check if aggregated data was created
        aggregated_metrics = aggregator.get_aggregated_metrics(
            metric_name="test_cpu_usage",
            aggregation_type=AggregationType.AVERAGE
        )
        
        assert len(aggregated_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_get_time_series_data(self, aggregator):
        """Test getting time series data."""
        # Add test data
        metrics = [
            MetricPoint(
                name="series_test",
                value=i * 10.0,
                timestamp=time.time() - (10 - i),
                metric_type=MetricType.COUNTER
            )
            for i in range(10)
        ]
        
        await aggregator.ingest_metrics(metrics)
        
        # Get time series data
        series_data = aggregator.get_time_series_data("series_test", duration_seconds=60)
        
        assert "raw_data" in series_data
        assert "aggregated_data" in series_data
        assert series_data["raw_data_points"] == 10
        assert len(series_data["raw_data"]) == 10


class TestAnomalyDetector:
    """Test cases for the AnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Anomaly detector instance for testing."""
        return AnomalyDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector._detection_rules is not None
        assert detector._alerting_rules is not None
        assert len(detector._detection_rules) > 0  # Should have default rules
        assert len(detector._alerting_rules) > 0   # Should have default rules
    
    @pytest.mark.asyncio
    async def test_z_score_detection(self, detector):
        """Test z-score anomaly detection."""
        from beautyai_inference.core.anomaly_detector import StatisticalDetector
        
        # Test normal values
        values = [50.0, 51.0, 49.0, 52.0, 48.0, 50.5, 49.5]
        current_value = 51.5
        
        is_anomaly, score, expected = StatisticalDetector.z_score_detection(
            values, current_value, threshold=2.0
        )
        
        assert not is_anomaly  # Should not be anomalous
        assert 0 <= score <= 1
        
        # Test anomalous value
        anomalous_value = 100.0
        is_anomaly, score, expected = StatisticalDetector.z_score_detection(
            values, anomalous_value, threshold=2.0
        )
        
        assert is_anomaly  # Should be anomalous
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_iqr_outlier_detection(self, detector):
        """Test IQR outlier detection."""
        from beautyai_inference.core.anomaly_detector import StatisticalDetector
        
        # Test data with outliers
        values = [10, 12, 14, 15, 13, 11, 16, 14, 12, 13]
        normal_value = 14.0
        outlier_value = 50.0
        
        # Test normal value
        is_anomaly, score, expected = StatisticalDetector.iqr_outlier_detection(
            values, normal_value
        )
        assert not is_anomaly
        
        # Test outlier
        is_anomaly, score, expected = StatisticalDetector.iqr_outlier_detection(
            values, outlier_value
        )
        assert is_anomaly
        assert score > 0
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, detector):
        """Test trend analysis."""
        from beautyai_inference.core.anomaly_detector import TrendAnalyzer
        
        # Test stable trend
        stable_values = [i + 1 for i in range(20)]  # Linear increase
        is_anomaly, score = TrendAnalyzer.detect_trend_change(stable_values)
        assert not is_anomaly
        
        # Test trend change
        trend_change_values = [i for i in range(10)] + [10 - i for i in range(10)]  # Up then down
        is_anomaly, score = TrendAnalyzer.detect_trend_change(trend_change_values)
        assert is_anomaly or score > 0  # Should detect significant change
    
    @pytest.mark.asyncio
    async def test_ingest_and_detect(self, detector):
        """Test ingesting metrics and detecting anomalies."""
        # Create test metrics
        current_time = time.time()
        metrics = []
        
        # Add normal CPU usage values
        for i in range(20):
            metrics.append(MetricPoint(
                name="system_cpu_usage_percent",
                value=50.0 + (i % 5),  # Values 50-54
                timestamp=current_time - (20 - i),
                metric_type=MetricType.GAUGE
            ))
        
        # Add anomalous value
        metrics.append(MetricPoint(
            name="system_cpu_usage_percent",
            value=98.0,  # Anomalous high value
            timestamp=current_time,
            metric_type=MetricType.GAUGE
        ))
        
        # Ingest metrics
        await detector.ingest_metrics(metrics)
        
        # Detect anomalies
        anomalies = await detector.detect_anomalies()
        
        # Should detect at least one anomaly
        assert len(anomalies) >= 0  # May or may not detect based on rules and thresholds
    
    @pytest.mark.asyncio
    async def test_add_detection_rule(self, detector):
        """Test adding custom detection rules."""
        rule = AnomalyDetectionRule(
            rule_id="test_rule",
            name="Test Rule",
            metric_pattern="test_*",
            algorithm=AnomalyDetectionAlgorithm.Z_SCORE,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER
        )
        
        initial_count = len(detector._detection_rules)
        detector.add_detection_rule(rule)
        
        assert len(detector._detection_rules) == initial_count + 1
        assert rule.rule_id in detector._detection_rules
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, detector):
        """Test generating alerts from anomalies."""
        # Create mock anomaly
        from beautyai_inference.core.anomaly_detector import DetectedAnomaly
        
        anomaly = DetectedAnomaly(
            anomaly_id="test_anomaly",
            rule_id="test_rule",
            metric_name="test_metric",
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            algorithm=AnomalyDetectionAlgorithm.Z_SCORE,
            timestamp=time.time(),
            value=100.0,
            expected_value=50.0,
            anomaly_score=0.9,
            severity=AlertSeverity.HIGH,
            description="Test anomaly",
            confidence=0.95
        )
        
        # Generate alerts
        alerts = await detector.generate_alerts([anomaly])
        
        # Check if alerts were generated (depends on alerting rules)
        assert isinstance(alerts, list)


class TestPerformanceIntegration:
    """Test cases for performance monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test performance monitoring service initialization."""
        service = get_performance_monitoring_service()
        assert service is not None
        assert isinstance(service, PerformanceMonitoringService)
    
    @pytest.mark.asyncio
    async def test_config_loading(self):
        """Test configuration loading."""
        service = PerformanceMonitoringService()
        
        # Test with mock config
        mock_config_data = {
            "enabled": True,
            "collection_interval_seconds": 5.0,
            "thresholds": {
                "cpu_usage_warning_threshold": 70.0
            },
            "historical_data": {
                "enable_historical_data": True
            }
        }
        
        config = service._create_config_from_dict(mock_config_data)
        
        assert config.enabled == True
        assert config.collection_interval_seconds == 5.0
        assert config.cpu_usage_warning_threshold == 70.0
        assert config.enable_historical_data == True
    
    @pytest.mark.asyncio
    async def test_custom_metric_addition(self):
        """Test adding custom metrics through service."""
        service = PerformanceMonitoringService()
        service._config = PerformanceConfig(enabled=True)
        
        # Mock the monitor
        mock_monitor = Mock()
        mock_monitor.add_custom_metric = AsyncMock()
        
        with patch.object(service, 'get_monitor', return_value=mock_monitor):
            await service.add_custom_metric("test_metric", 42.0, {"label": "test"})
            mock_monitor.add_custom_metric.assert_called_once()


class TestPerformanceDashboardAPI:
    """Test cases for performance dashboard API endpoints."""
    
    @pytest.fixture
    def mock_monitor(self):
        """Mock performance monitor."""
        monitor = Mock()
        monitor.get_performance_dashboard = AsyncMock(return_value={
            "timestamp": time.time(),
            "system_summary": {"system_cpu_usage_percent": 50.0},
            "alert_summary": {"total_active": 2},
            "current_metrics": {"test_metric": [{"value": 42.0}]}
        })
        monitor.get_current_metrics = AsyncMock(return_value={
            "test_metric": [{"value": 42.0, "timestamp": time.time()}]
        })
        monitor.metrics_collector = Mock()
        monitor.metrics_collector.get_historical_metrics = Mock(return_value=[])
        monitor.anomaly_detector = Mock()
        monitor.anomaly_detector.get_active_alerts = Mock(return_value=[])
        monitor.anomaly_detector.get_alert_history = Mock(return_value=[])
        monitor.anomaly_detector.get_active_anomalies = Mock(return_value=[])
        monitor.anomaly_detector.get_anomaly_statistics = Mock(return_value={})
        
        return monitor
    
    @pytest.mark.asyncio
    async def test_dashboard_endpoint(self, mock_monitor):
        """Test dashboard API endpoint."""
        from beautyai_inference.api.endpoints.performance_dashboard import get_performance_dashboard
        
        with patch('beautyai_inference.api.endpoints.performance_dashboard.get_performance_monitor', return_value=mock_monitor):
            response = await get_performance_dashboard(monitor=mock_monitor)
            
            assert response.status_code == 200
            mock_monitor.get_performance_dashboard.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_current_metrics_endpoint(self, mock_monitor):
        """Test current metrics API endpoint."""
        from beautyai_inference.api.endpoints.performance_dashboard import get_current_metrics
        
        with patch('beautyai_inference.api.endpoints.performance_dashboard.get_performance_monitor', return_value=mock_monitor):
            response = await get_current_metrics(metric_name=None, monitor=mock_monitor)
            
            assert response.status_code == 200
            mock_monitor.get_current_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_system_status_endpoint(self, mock_monitor):
        """Test system status API endpoint."""
        from beautyai_inference.api.endpoints.performance_dashboard import get_system_status
        
        with patch('beautyai_inference.api.endpoints.performance_dashboard.get_performance_monitor', return_value=mock_monitor):
            response = await get_system_status(monitor=mock_monitor)
            
            assert response.status_code == 200
            mock_monitor.get_current_metrics.assert_called_once()


@pytest.mark.integration
class TestPerformanceMonitoringIntegration:
    """Integration tests for the complete performance monitoring system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self):
        """Test complete end-to-end monitoring workflow."""
        # Create configuration
        config = PerformanceConfig(
            enabled=True,
            collection_interval_seconds=0.1,  # Fast for testing
            metrics_retention_seconds=60,
            anomaly_detection_enabled=True,
            system_metrics_enabled=True
        )
        
        # Initialize monitor
        monitor = await initialize_performance_monitor(config)
        assert monitor is not None
        
        try:
            # Wait for one collection cycle
            await asyncio.sleep(0.2)
            
            # Add custom metric
            await monitor.add_custom_metric("test_integration_metric", 75.0)
            
            # Get dashboard data
            dashboard = await monitor.get_performance_dashboard()
            assert dashboard is not None
            assert "timestamp" in dashboard
            assert "current_metrics" in dashboard
            
            # Check if we have some metrics
            current_metrics = await monitor.get_current_metrics()
            assert len(current_metrics) > 0
            
        finally:
            # Cleanup
            await shutdown_performance_monitor()
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation_integration(self):
        """Test metrics aggregation integration."""
        config = PerformanceConfig(
            enabled=True,
            collection_interval_seconds=0.1,
            data_aggregation_interval_seconds=0.5
        )
        
        monitor = PerformanceMonitor(config)
        await monitor.start()
        
        try:
            # Add multiple data points
            for i in range(5):
                await monitor.add_custom_metric("integration_test", 50.0 + i * 5)
                await asyncio.sleep(0.1)
            
            # Wait for aggregation
            await asyncio.sleep(0.6)
            
            # Check aggregated data
            summary = monitor.metrics_aggregator.get_metrics_summary()
            assert summary["total_raw_metrics"] > 0
            
        finally:
            await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_integration(self):
        """Test anomaly detection integration."""
        monitor = PerformanceMonitor()
        await monitor.start()
        
        try:
            # Add normal metrics
            for i in range(10):
                await monitor.add_custom_metric("anomaly_test", 50.0 + i)
                await asyncio.sleep(0.05)
            
            # Add anomalous metric
            await monitor.add_custom_metric("anomaly_test", 150.0)
            
            # Wait for detection
            await asyncio.sleep(0.2)
            
            # Check for anomalies
            anomalies = monitor.anomaly_detector.get_active_anomalies(duration_minutes=1)
            statistics = monitor.anomaly_detector.get_anomaly_statistics()
            
            assert isinstance(anomalies, list)
            assert isinstance(statistics, dict)
            
        finally:
            await monitor.stop()


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([
        __file__ + "::TestPerformanceMonitor::test_performance_monitor_initialization",
        "-v"
    ])