"""
Performance Monitoring Integration for BeautyAI API.

Initializes and manages the performance monitoring system integration
with the main API server, WebSocket endpoints, and service layer.

This module provides:
- Startup and shutdown hooks for performance monitoring
- Configuration loading from config system
- Integration with API router registration
- Service layer integration for custom metrics

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from ..core.performance_monitor import (
    PerformanceMonitor,
    initialize_performance_monitor,
    shutdown_performance_monitor,
    get_performance_monitor
)
from ..core.performance_types import PerformanceConfig
from ..core.config_manager import ConfigManager
from .endpoints.performance_dashboard import get_performance_router

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    """
    Service for managing performance monitoring integration with the API.
    
    Handles initialization, configuration, and lifecycle management
    of the performance monitoring system within the BeautyAI API.
    """
    
    def __init__(self):
        self._monitor: Optional[PerformanceMonitor] = None
        self._config: Optional[PerformanceConfig] = None
        self._initialized = False
    
    async def initialize(self, config_manager: Optional[ConfigManager] = None) -> bool:
        """
        Initialize performance monitoring from configuration.
        
        Args:
            config_manager: Configuration manager instance
            
        Returns:
            True if successfully initialized, False otherwise
        """
        try:
            logger.info("Initializing performance monitoring service")
            
            # Load configuration
            if config_manager:
                config_data = await config_manager.get_config_section("performance_monitoring")
            else:
                # Load from default config
                from ..core.config_management import get_config_manager
                config_manager = get_config_manager()
                if config_manager:
                    config_data = await config_manager.get_config_section("performance_monitoring")
                else:
                    config_data = {}
            
            # Create configuration object
            self._config = self._create_config_from_dict(config_data)
            
            # Initialize performance monitor
            if self._config.enabled:
                self._monitor = await initialize_performance_monitor(self._config)
                logger.info("Performance monitoring initialized successfully")
            else:
                logger.info("Performance monitoring disabled by configuration")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring: {e}")
            return False
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> PerformanceConfig:
        """Create PerformanceConfig from dictionary."""
        # Extract main config
        main_config = {
            "enabled": config_data.get("enabled", True),
            "collection_interval_seconds": config_data.get("collection_interval_seconds", 10.0),
            "metrics_retention_seconds": config_data.get("metrics_retention_seconds", 3600),
            "anomaly_detection_enabled": config_data.get("anomaly_detection_enabled", True),
            "alerting_enabled": config_data.get("alerting_enabled", True),
            "system_metrics_enabled": config_data.get("system_metrics_enabled", True),
            "circuit_breaker_monitoring": config_data.get("circuit_breaker_monitoring", True),
            "connection_pool_monitoring": config_data.get("connection_pool_monitoring", True),
            "custom_metrics_enabled": config_data.get("custom_metrics_enabled", True)
        }
        
        # Extract thresholds
        thresholds = config_data.get("thresholds", {})
        threshold_config = {
            "cpu_usage_warning_threshold": thresholds.get("cpu_usage_warning_threshold", 80.0),
            "cpu_usage_critical_threshold": thresholds.get("cpu_usage_critical_threshold", 95.0),
            "memory_usage_warning_threshold": thresholds.get("memory_usage_warning_threshold", 85.0),
            "memory_usage_critical_threshold": thresholds.get("memory_usage_critical_threshold", 95.0),
            "response_time_warning_threshold_ms": thresholds.get("response_time_warning_threshold_ms", 1000.0),
            "response_time_critical_threshold_ms": thresholds.get("response_time_critical_threshold_ms", 5000.0),
            "error_rate_warning_threshold": thresholds.get("error_rate_warning_threshold", 0.05),
            "error_rate_critical_threshold": thresholds.get("error_rate_critical_threshold", 0.10)
        }
        
        # Extract historical data config
        historical_data = config_data.get("historical_data", {})
        historical_config = {
            "enable_historical_data": historical_data.get("enable_historical_data", True),
            "historical_data_points": historical_data.get("historical_data_points", 1000),
            "data_aggregation_interval_seconds": historical_data.get("data_aggregation_interval_seconds", 60.0)
        }
        
        # Combine all configuration
        full_config = {**main_config, **threshold_config, **historical_config}
        
        return PerformanceConfig(**full_config)
    
    async def shutdown(self):
        """Shutdown performance monitoring service."""
        try:
            logger.info("Shutting down performance monitoring service")
            
            if self._monitor:
                await shutdown_performance_monitor()
                self._monitor = None
            
            self._initialized = False
            logger.info("Performance monitoring service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down performance monitoring: {e}")
    
    def get_monitor(self) -> Optional[PerformanceMonitor]:
        """Get the performance monitor instance."""
        return get_performance_monitor()
    
    def is_initialized(self) -> bool:
        """Check if performance monitoring is initialized."""
        return self._initialized and self.get_monitor() is not None
    
    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self._config is not None and self._config.enabled
    
    async def add_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a custom application metric."""
        monitor = self.get_monitor()
        if monitor:
            from ..core.performance_monitor import MetricType
            await monitor.add_custom_metric(name, value, MetricType.GAUGE, labels)
    
    async def get_dashboard_data(self) -> Optional[Dict[str, Any]]:
        """Get performance dashboard data."""
        monitor = self.get_monitor()
        if monitor:
            return await monitor.get_performance_dashboard()
        return None
    
    def get_api_router(self):
        """Get the performance monitoring API router."""
        return get_performance_router()


# Global service instance
_performance_monitoring_service: Optional[PerformanceMonitoringService] = None


def get_performance_monitoring_service() -> PerformanceMonitoringService:
    """Get the global performance monitoring service instance."""
    global _performance_monitoring_service
    
    if _performance_monitoring_service is None:
        _performance_monitoring_service = PerformanceMonitoringService()
    
    return _performance_monitoring_service


async def initialize_performance_monitoring(config_manager: Optional[ConfigManager] = None) -> bool:
    """Initialize performance monitoring service."""
    service = get_performance_monitoring_service()
    return await service.initialize(config_manager)


async def shutdown_performance_monitoring():
    """Shutdown performance monitoring service."""
    global _performance_monitoring_service
    
    if _performance_monitoring_service:
        await _performance_monitoring_service.shutdown()
        _performance_monitoring_service = None