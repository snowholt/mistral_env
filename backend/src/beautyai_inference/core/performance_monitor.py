"""
Performance Monitoring System for BeautyAI Framework.

Provides comprehensive monitoring of system resources, application metrics,
and anomaly detection with configurable thresholds and alerting.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import time
import psutil
import logging
import json
import statistics
import threading
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict, deque
from enum import Enum, auto

# Import shared types
from .performance_types import (
    MetricPoint,
    PerformanceAlert,
    MetricType,
    AnomalyType,
    AlertSeverity,
    AlertLevel,
    AggregationType,
    TimeWindow,
    PerformanceConfig
)

from .circuit_breaker import CircuitBreakerRegistry
from .websocket_connection_pool import get_websocket_pool
from .metrics_aggregator import MetricsAggregator
from .anomaly_detector import AnomalyDetector as CoreAnomalyDetector

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()


@dataclass
class MetricPoint:
    """Individual metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: Union[int, float]
    threshold_value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


class MetricsCollector:
    """Collects metrics from various sources."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._metrics_buffer = deque(maxlen=config.historical_data_points)
        self._custom_metrics: Dict[str, MetricPoint] = {}
        self._lock = asyncio.Lock()
        
    async def collect_system_metrics(self) -> List[MetricPoint]:
        """Collect system resource metrics."""
        if not self.config.system_metrics_enabled:
            return []
        
        metrics = []
        current_time = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(MetricPoint(
                name="system_cpu_usage_percent",
                value=cpu_percent,
                timestamp=current_time,
                metric_type=MetricType.GAUGE
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(MetricPoint(
                name="system_memory_usage_percent",
                value=memory.percent,
                timestamp=current_time,
                metric_type=MetricType.GAUGE
            ))
            
            metrics.append(MetricPoint(
                name="system_memory_used_bytes",
                value=memory.used,
                timestamp=current_time,
                metric_type=MetricType.GAUGE
            ))
            
            metrics.append(MetricPoint(
                name="system_memory_available_bytes",
                value=memory.available,
                timestamp=current_time,
                metric_type=MetricType.GAUGE
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(MetricPoint(
                name="system_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                timestamp=current_time,
                metric_type=MetricType.GAUGE
            ))
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                metrics.extend([
                    MetricPoint(
                        name="system_network_bytes_sent",
                        value=net_io.bytes_sent,
                        timestamp=current_time,
                        metric_type=MetricType.COUNTER
                    ),
                    MetricPoint(
                        name="system_network_bytes_recv",
                        value=net_io.bytes_recv,
                        timestamp=current_time,
                        metric_type=MetricType.COUNTER
                    )
                ])
            except Exception as e:
                logger.warning(f"Failed to collect network metrics: {e}")
            
            # Process-specific metrics
            process = psutil.Process()
            metrics.extend([
                MetricPoint(
                    name="process_memory_rss_bytes",
                    value=process.memory_info().rss,
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE
                ),
                MetricPoint(
                    name="process_cpu_percent",
                    value=process.cpu_percent(),
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE
                ),
                MetricPoint(
                    name="process_open_files_count",
                    value=len(process.open_files()),
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE
                )
            ])
            
            # Python GC metrics
            gc_stats = gc.get_stats()
            for i, stat in enumerate(gc_stats):
                metrics.append(MetricPoint(
                    name=f"python_gc_collections_generation_{i}",
                    value=stat['collections'],
                    timestamp=current_time,
                    metric_type=MetricType.COUNTER
                ))
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def collect_circuit_breaker_metrics(self) -> List[MetricPoint]:
        """Collect metrics from circuit breakers."""
        if not self.config.circuit_breaker_monitoring:
            return []
        
        metrics = []
        current_time = time.time()
        
        try:
            from .circuit_breaker import circuit_breaker_registry
            cb_metrics = circuit_breaker_registry.get_all_metrics()
            
            for cb_name, cb_data in cb_metrics.items():
                labels = {"circuit_breaker": cb_name}
                
                # State metrics
                state_value = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}.get(cb_data["state"], -1)
                metrics.append(MetricPoint(
                    name="circuit_breaker_state",
                    value=state_value,
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ))
                
                # Request metrics
                metrics.extend([
                    MetricPoint(
                        name="circuit_breaker_requests_total",
                        value=cb_data["total_requests"],
                        timestamp=current_time,
                        metric_type=MetricType.COUNTER,
                        labels=labels
                    ),
                    MetricPoint(
                        name="circuit_breaker_requests_successful",
                        value=cb_data["successful_requests"],
                        timestamp=current_time,
                        metric_type=MetricType.COUNTER,
                        labels=labels
                    ),
                    MetricPoint(
                        name="circuit_breaker_requests_failed",
                        value=cb_data["failed_requests"],
                        timestamp=current_time,
                        metric_type=MetricType.COUNTER,
                        labels=labels
                    ),
                    MetricPoint(
                        name="circuit_breaker_failure_rate",
                        value=cb_data["failure_rate"],
                        timestamp=current_time,
                        metric_type=MetricType.GAUGE,
                        labels=labels
                    ),
                    MetricPoint(
                        name="circuit_breaker_average_response_time_ms",
                        value=cb_data["average_response_time_ms"],
                        timestamp=current_time,
                        metric_type=MetricType.GAUGE,
                        labels=labels
                    ),
                    MetricPoint(
                        name="circuit_breaker_uptime_percentage",
                        value=cb_data["uptime_percentage"],
                        timestamp=current_time,
                        metric_type=MetricType.GAUGE,
                        labels=labels
                    )
                ])
                
        except Exception as e:
            logger.error(f"Error collecting circuit breaker metrics: {e}")
        
        return metrics
    
    async def collect_connection_pool_metrics(self) -> List[MetricPoint]:
        """Collect metrics from connection pools."""
        if not self.config.connection_pool_monitoring:
            return []
        
        metrics = []
        current_time = time.time()
        
        try:
            pool = get_websocket_pool()
            pool_metrics = pool.get_metrics()
            
            pool_data = pool_metrics["pool"]
            labels = {"pool": pool_data["name"]}
            
            # Connection metrics
            metrics.extend([
                MetricPoint(
                    name="connection_pool_total_connections",
                    value=pool_data["total_connections"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_active_connections",
                    value=pool_data["active_connections"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_idle_connections",
                    value=pool_data["idle_connections"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_unhealthy_connections",
                    value=pool_data["unhealthy_connections"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_utilization",
                    value=pool_data["pool_utilization"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_peak_usage",
                    value=pool_data["peak_usage"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_total_created",
                    value=pool_data["total_created"],
                    timestamp=current_time,
                    metric_type=MetricType.COUNTER,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_total_destroyed",
                    value=pool_data["total_destroyed"],
                    timestamp=current_time,
                    metric_type=MetricType.COUNTER,
                    labels=labels
                ),
                MetricPoint(
                    name="connection_pool_average_acquisition_time_ms",
                    value=pool_data["avg_acquisition_time_ms"],
                    timestamp=current_time,
                    metric_type=MetricType.GAUGE,
                    labels=labels
                )
            ])
            
        except Exception as e:
            logger.error(f"Error collecting connection pool metrics: {e}")
        
        return metrics
    
    async def add_custom_metric(self, metric: MetricPoint):
        """Add a custom application metric."""
        if not self.config.custom_metrics_enabled:
            return
        
        async with self._lock:
            self._custom_metrics[metric.name] = metric
    
    async def get_custom_metrics(self) -> List[MetricPoint]:
        """Get all custom metrics."""
        async with self._lock:
            return list(self._custom_metrics.values())
    
    async def collect_all_metrics(self) -> List[MetricPoint]:
        """Collect all metrics from all sources."""
        all_metrics = []
        
        # Collect system metrics
        system_metrics = await self.collect_system_metrics()
        all_metrics.extend(system_metrics)
        
        # Collect circuit breaker metrics
        cb_metrics = await self.collect_circuit_breaker_metrics()
        all_metrics.extend(cb_metrics)
        
        # Collect connection pool metrics
        pool_metrics = await self.collect_connection_pool_metrics()
        all_metrics.extend(pool_metrics)
        
        # Add custom metrics
        custom_metrics = await self.get_custom_metrics()
        all_metrics.extend(custom_metrics)
        
        # Store in buffer for historical tracking
        if self.config.enable_historical_data:
            self._metrics_buffer.extend(all_metrics)
        
        return all_metrics
    
    def get_historical_metrics(self, metric_name: Optional[str] = None, 
                              duration_seconds: Optional[int] = None) -> List[MetricPoint]:
        """Get historical metrics data."""
        if not self.config.enable_historical_data:
            return []
        
        metrics = list(self._metrics_buffer)
        
        # Filter by metric name if provided
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        # Filter by time duration if provided
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics


class AnomalyDetector:
    """Detects performance anomalies and generates alerts."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: List[PerformanceAlert] = []
        
    async def analyze_metrics(self, metrics: List[MetricPoint]) -> List[PerformanceAlert]:
        """Analyze metrics for anomalies and generate alerts."""
        if not self.config.anomaly_detection_enabled:
            return []
        
        alerts = []
        current_time = time.time()
        
        for metric in metrics:
            # Update baseline
            self._baseline_metrics[metric.name].append(metric.value)
            
            # Check for threshold-based alerts
            threshold_alerts = self._check_threshold_alerts(metric, current_time)
            alerts.extend(threshold_alerts)
            
            # Check for statistical anomalies
            if len(self._baseline_metrics[metric.name]) >= 10:
                statistical_alerts = self._check_statistical_anomalies(metric, current_time)
                alerts.extend(statistical_alerts)
        
        # Update active alerts
        for alert in alerts:
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
        
        return alerts
    
    def _check_threshold_alerts(self, metric: MetricPoint, current_time: float) -> List[PerformanceAlert]:
        """Check for threshold-based alerts."""
        alerts = []
        
        # CPU usage thresholds
        if metric.name == "system_cpu_usage_percent":
            if metric.value >= self.config.cpu_usage_critical_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"cpu_critical_{int(current_time)}",
                    name="High CPU Usage - Critical",
                    level=AlertLevel.CRITICAL,
                    message=f"CPU usage is {metric.value:.1f}% (critical threshold: {self.config.cpu_usage_critical_threshold}%)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.cpu_usage_critical_threshold,
                    labels=metric.labels
                ))
            elif metric.value >= self.config.cpu_usage_warning_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"cpu_warning_{int(current_time)}",
                    name="High CPU Usage - Warning",
                    level=AlertLevel.WARNING,
                    message=f"CPU usage is {metric.value:.1f}% (warning threshold: {self.config.cpu_usage_warning_threshold}%)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.cpu_usage_warning_threshold,
                    labels=metric.labels
                ))
        
        # Memory usage thresholds
        elif metric.name == "system_memory_usage_percent":
            if metric.value >= self.config.memory_usage_critical_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"memory_critical_{int(current_time)}",
                    name="High Memory Usage - Critical",
                    level=AlertLevel.CRITICAL,
                    message=f"Memory usage is {metric.value:.1f}% (critical threshold: {self.config.memory_usage_critical_threshold}%)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.memory_usage_critical_threshold,
                    labels=metric.labels
                ))
            elif metric.value >= self.config.memory_usage_warning_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"memory_warning_{int(current_time)}",
                    name="High Memory Usage - Warning",
                    level=AlertLevel.WARNING,
                    message=f"Memory usage is {metric.value:.1f}% (warning threshold: {self.config.memory_usage_warning_threshold}%)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.memory_usage_warning_threshold,
                    labels=metric.labels
                ))
        
        # Response time thresholds
        elif "response_time" in metric.name or "acquisition_time" in metric.name:
            if metric.value >= self.config.response_time_critical_threshold_ms:
                alerts.append(PerformanceAlert(
                    alert_id=f"response_time_critical_{int(current_time)}",
                    name="High Response Time - Critical",
                    level=AlertLevel.CRITICAL,
                    message=f"Response time is {metric.value:.1f}ms (critical threshold: {self.config.response_time_critical_threshold_ms}ms)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.response_time_critical_threshold_ms,
                    labels=metric.labels
                ))
            elif metric.value >= self.config.response_time_warning_threshold_ms:
                alerts.append(PerformanceAlert(
                    alert_id=f"response_time_warning_{int(current_time)}",
                    name="High Response Time - Warning",
                    level=AlertLevel.WARNING,
                    message=f"Response time is {metric.value:.1f}ms (warning threshold: {self.config.response_time_warning_threshold_ms}ms)",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.response_time_warning_threshold_ms,
                    labels=metric.labels
                ))
        
        # Error rate thresholds
        elif "failure_rate" in metric.name or "error_rate" in metric.name:
            if metric.value >= self.config.error_rate_critical_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"error_rate_critical_{int(current_time)}",
                    name="High Error Rate - Critical",
                    level=AlertLevel.CRITICAL,
                    message=f"Error rate is {metric.value:.1%} (critical threshold: {self.config.error_rate_critical_threshold:.1%})",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.error_rate_critical_threshold,
                    labels=metric.labels
                ))
            elif metric.value >= self.config.error_rate_warning_threshold:
                alerts.append(PerformanceAlert(
                    alert_id=f"error_rate_warning_{int(current_time)}",
                    name="High Error Rate - Warning",
                    level=AlertLevel.WARNING,
                    message=f"Error rate is {metric.value:.1%} (warning threshold: {self.config.error_rate_warning_threshold:.1%})",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=self.config.error_rate_warning_threshold,
                    labels=metric.labels
                ))
        
        return alerts
    
    def _check_statistical_anomalies(self, metric: MetricPoint, current_time: float) -> List[PerformanceAlert]:
        """Check for statistical anomalies using baseline data."""
        alerts = []
        baseline_data = list(self._baseline_metrics[metric.name])
        
        if len(baseline_data) < 10:
            return alerts
        
        try:
            # Calculate statistics
            mean_value = statistics.mean(baseline_data[:-1])  # Exclude current value
            std_dev = statistics.stdev(baseline_data[:-1])
            
            # Z-score anomaly detection (3-sigma rule)
            z_score = abs(metric.value - mean_value) / std_dev if std_dev > 0 else 0
            
            if z_score >= 3.0:  # 3-sigma anomaly
                alerts.append(PerformanceAlert(
                    alert_id=f"anomaly_critical_{metric.name}_{int(current_time)}",
                    name=f"Statistical Anomaly Detected - {metric.name}",
                    level=AlertLevel.WARNING,
                    message=f"Metric {metric.name} value {metric.value} deviates significantly from baseline (z-score: {z_score:.2f})",
                    timestamp=current_time,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=mean_value + (3 * std_dev),
                    labels=metric.labels,
                    metadata={
                        "z_score": z_score,
                        "baseline_mean": mean_value,
                        "baseline_std_dev": std_dev
                    }
                ))
        except statistics.StatisticsError:
            pass  # Not enough data for statistical analysis
        
        return alerts
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        alerts = list(self._active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return [alert for alert in alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
    
    def get_alert_history(self, duration_seconds: Optional[int] = None) -> List[PerformanceAlert]:
        """Get alert history."""
        alerts = self._alert_history
        
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            alerts = [alert for alert in alerts if alert.timestamp >= cutoff_time]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)


class PerformanceMonitor:
    """
    Main performance monitoring system for BeautyAI.
    
    Aggregates metrics from all sources, detects anomalies, and provides
    comprehensive monitoring capabilities with real-time dashboards.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.metrics_aggregator = MetricsAggregator(self.config)
        self.anomaly_detector = CoreAnomalyDetector()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self._metric_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Historical data tracking
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info(f"Initialized PerformanceMonitor with config: enabled={self.config.enabled}")
    
    async def start(self):
        """Start the performance monitoring system."""
        if not self.config.enabled:
            logger.info("Performance monitoring disabled by configuration")
            return
        
        logger.info("Starting performance monitoring system")
        
        # Start metrics aggregator
        await self.metrics_aggregator.start()
        
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring system started")
    
    async def stop(self):
        """Stop the performance monitoring system."""
        logger.info("Stopping performance monitoring system")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop metrics aggregator
        await self.metrics_aggregator.stop()
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect all metrics
                metrics = await self.metrics_collector.collect_all_metrics()
                
                # Feed metrics to aggregator
                await self.metrics_aggregator.ingest_metrics(metrics)
                
                # Feed metrics to anomaly detector
                await self.anomaly_detector.ingest_metrics(metrics)
                
                # Detect anomalies
                detected_anomalies = await self.anomaly_detector.detect_anomalies()
                
                # Generate alerts from anomalies
                alerts = await self.anomaly_detector.generate_alerts(detected_anomalies)
                
                # Store performance history
                await self._update_performance_history(metrics, detected_anomalies, alerts)
                
                # Execute callbacks
                await self._execute_callbacks(metrics, alerts)
                
                # Log summary
                if metrics or detected_anomalies or alerts:
                    logger.debug(f"Collected {len(metrics)} metrics, detected {len(detected_anomalies)} anomalies, generated {len(alerts)} alerts")
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief delay on error
    
    async def _execute_callbacks(self, metrics: List[MetricPoint], alerts: List[PerformanceAlert]):
        """Execute registered callbacks."""
        # Metric callbacks
        for callback in self._metric_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.warning(f"Metric callback failed: {e}")
        
        # Alert callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")
    
    async def _update_performance_history(self, metrics: List[MetricPoint], 
                                        anomalies: List, alerts: List[PerformanceAlert]):
        """Update performance history tracking."""
        if not self.config.enable_historical_data:
            return
        
        current_time = time.time()
        
        # Create history entry
        history_entry = {
            "timestamp": current_time,
            "metrics_count": len(metrics),
            "anomalies_count": len(anomalies),
            "alerts_count": len(alerts),
            "system_status": await self._get_system_status_summary(metrics),
            "performance_summary": await self._get_performance_summary(metrics)
        }
        
        # Store in history
        self._performance_history["daily"].append(history_entry)
        
        # Limit history size
        if len(self._performance_history["daily"]) > 1440:  # 24 hours of minute-level data
            self._performance_history["daily"] = self._performance_history["daily"][-720:]
    
    async def _get_system_status_summary(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Get system status summary from current metrics."""
        status_summary = {"overall_health": "healthy"}
        
        for metric in metrics:
            if metric.name == "system_cpu_usage_percent":
                if metric.value > 95:
                    status_summary["cpu_status"] = "critical"
                elif metric.value > 80:
                    status_summary["cpu_status"] = "warning"
                else:
                    status_summary["cpu_status"] = "healthy"
                status_summary["cpu_usage"] = metric.value
            
            elif metric.name == "system_memory_usage_percent":
                if metric.value > 95:
                    status_summary["memory_status"] = "critical"
                elif metric.value > 85:
                    status_summary["memory_status"] = "warning"
                else:
                    status_summary["memory_status"] = "healthy"
                status_summary["memory_usage"] = metric.value
        
        # Determine overall health
        if any(status.endswith("critical") for key, status in status_summary.items() if key.endswith("_status")):
            status_summary["overall_health"] = "critical"
        elif any(status.endswith("warning") for key, status in status_summary.items() if key.endswith("_status")):
            status_summary["overall_health"] = "warning"
        
        return status_summary
    
    async def _get_performance_summary(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Get performance summary from current metrics."""
        performance_summary = {}
        
        # Group metrics by type
        system_metrics = [m for m in metrics if m.name.startswith("system_")]
        pool_metrics = [m for m in metrics if "connection_pool" in m.name]
        cb_metrics = [m for m in metrics if "circuit_breaker" in m.name]
        
        performance_summary["system_metrics_count"] = len(system_metrics)
        performance_summary["connection_pool_metrics_count"] = len(pool_metrics)
        performance_summary["circuit_breaker_metrics_count"] = len(cb_metrics)
        
        return performance_summary
    
    def add_metric_callback(self, callback: Callable):
        """Add callback for metric collection events."""
        self._metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert events."""
        self._alert_callbacks.append(callback)
    
    async def add_custom_metric(self, name: str, value: Union[int, float], 
                               metric_type: MetricType = MetricType.GAUGE,
                               labels: Optional[Dict[str, str]] = None):
        """Add a custom application metric."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            labels=labels or {}
        )
        await self.metrics_collector.add_custom_metric(metric)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current snapshot of all metrics."""
        metrics = await self.metrics_collector.collect_all_metrics()
        
        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric.name].append({
                "value": metric.value,
                "timestamp": metric.timestamp,
                "metric_type": metric.metric_type.name,
                "labels": metric.labels,
                "metadata": metric.metadata
            })
        
        return dict(grouped_metrics)
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        current_time = time.time()
        
        # Current metrics
        current_metrics = await self.get_current_metrics()
        
        # Active alerts
        active_anomalies = self.anomaly_detector.get_active_anomalies()
        
        # Alert summary  
        alert_summary = {
            "total_active": len(active_anomalies),
            "critical": len([a for a in active_anomalies if hasattr(a, 'severity') and a.severity.name == 'CRITICAL']),
            "warning": len([a for a in active_anomalies if hasattr(a, 'severity') and a.severity.name == 'WARNING']),
            "info": len([a for a in active_anomalies if hasattr(a, 'severity') and a.severity.name == 'INFO'])
        }
        
        # Historical data (last hour)
        historical_metrics = self.metrics_collector.get_historical_metrics(duration_seconds=3600)
        
        # Aggregated data summary
        aggregated_summary = self.metrics_aggregator.get_metrics_summary()
        
        # Anomaly statistics
        anomaly_statistics = self.anomaly_detector.get_anomaly_statistics()
        
        # System summary
        system_summary = {}
        for metric_name in ["system_cpu_usage_percent", "system_memory_usage_percent"]:
            if metric_name in current_metrics and current_metrics[metric_name]:
                latest_metric = current_metrics[metric_name][-1]
                system_summary[metric_name] = latest_metric["value"]
        
        # Performance history
        performance_history = self.get_performance_history(duration_hours=24)
        
        return {
            "timestamp": current_time,
            "monitoring_config": {
                "enabled": self.config.enabled,
                "collection_interval_seconds": self.config.collection_interval_seconds,
                "anomaly_detection_enabled": self.config.anomaly_detection_enabled,
                "alerting_enabled": self.config.alerting_enabled
            },
            "system_summary": system_summary,
            "alert_summary": alert_summary,
            "anomaly_statistics": anomaly_statistics,
            "aggregation_summary": aggregated_summary,
            "performance_history_summary": {
                "entries_count": len(performance_history),
                "latest_entry": performance_history[-1] if performance_history else None
            },
            "active_alerts": [
                {
                    "anomaly_id": getattr(anomaly, 'anomaly_id', 'unknown'),
                    "metric_name": getattr(anomaly, 'metric_name', 'unknown'),
                    "anomaly_type": getattr(anomaly, 'anomaly_type', 'unknown'),
                    "severity": getattr(anomaly, 'severity', 'unknown'),
                    "timestamp": getattr(anomaly, 'timestamp', 0),
                    "value": getattr(anomaly, 'value', 0),
                    "description": getattr(anomaly, 'description', ''),
                    "labels": getattr(anomaly, 'labels', {})
                }
                for anomaly in active_anomalies
            ],
            "current_metrics": current_metrics,
            "metrics_count": len(historical_metrics),
            "uptime_seconds": current_time - (self._monitoring_task.get_loop().time() if self._monitoring_task else current_time)
        }
    
    def get_performance_history(self, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for specified duration."""
        if not self.config.enable_historical_data:
            return []
        
        cutoff_time = time.time() - (duration_hours * 3600)
        
        # Filter history by time
        filtered_history = [
            entry for entry in self._performance_history.get("daily", [])
            if entry["timestamp"] >= cutoff_time
        ]
        
        return sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)


# Global performance monitor instance
_performance_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _performance_monitor_instance


async def initialize_performance_monitor(config: Optional[PerformanceConfig] = None) -> PerformanceMonitor:
    """Initialize the global performance monitor."""
    global _performance_monitor_instance
    
    if _performance_monitor_instance is not None:
        await _performance_monitor_instance.stop()
    
    _performance_monitor_instance = PerformanceMonitor(config)
    await _performance_monitor_instance.start()
    
    return _performance_monitor_instance


async def shutdown_performance_monitor():
    """Shutdown the global performance monitor."""
    global _performance_monitor_instance
    
    if _performance_monitor_instance is not None:
        await _performance_monitor_instance.stop()
        _performance_monitor_instance = None