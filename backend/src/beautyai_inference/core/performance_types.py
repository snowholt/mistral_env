"""
Common types and data structures for performance monitoring.

This module contains shared types, enums, and data classes used across
the performance monitoring system to avoid circular imports.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"      # Monotonically increasing values
    GAUGE = "gauge"          # Point-in-time measurements  
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"          # Timing measurements


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_CHANGE = "trend_change"
    THRESHOLD_VIOLATION = "threshold_violation"
    PATTERN_DEVIATION = "pattern_deviation"
    CORRELATION_ANOMALY = "correlation_anomaly"


class AnomalyDetectionAlgorithm(Enum):
    """Algorithms used for anomaly detection."""
    Z_SCORE = "z_score"
    IQR_OUTLIER = "iqr_outlier"
    ISOLATION_FOREST = "isolation_forest"
    MOVING_AVERAGE = "moving_average"
    STATISTICAL_THRESHOLD = "statistical_threshold"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertLevel(Enum):
    """Alert levels for system notifications."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AggregationType(Enum):
    """Types of metric aggregations."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_90 = "p90" 
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class TimeWindow(Enum):
    """Time windows for aggregation."""
    SECOND_1 = 1
    SECOND_5 = 5
    SECOND_10 = 10
    SECOND_30 = 30
    MINUTE_1 = 60
    MINUTE_5 = 300
    MINUTE_15 = 900
    HOUR_1 = 3600
    HOUR_6 = 21600
    DAY_1 = 86400


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    labels: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class PerformanceAlert:
    """A performance alert generated from anomaly detection."""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    timestamp: float
    message: str
    value: float
    threshold: Optional[float] = None
    acknowledged: bool = False
    labels: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    collection_interval_seconds: float = 10.0
    retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    anomaly_detection_enabled: bool = True
    dashboard_enabled: bool = True
    custom_metrics_enabled: bool = True
    
    # Additional detailed configuration
    metrics_retention_seconds: int = 3600  # 1 hour
    alerting_enabled: bool = True
    system_metrics_enabled: bool = True
    circuit_breaker_monitoring: bool = True
    connection_pool_monitoring: bool = True
    
    # Alerting thresholds
    cpu_usage_warning_threshold: float = 80.0
    cpu_usage_critical_threshold: float = 95.0
    memory_usage_warning_threshold: float = 85.0
    memory_usage_critical_threshold: float = 95.0
    response_time_warning_threshold_ms: float = 1000.0
    response_time_critical_threshold_ms: float = 5000.0
    error_rate_warning_threshold: float = 0.05  # 5%
    error_rate_critical_threshold: float = 0.10  # 10%
    
    # Historical data settings
    enable_historical_data: bool = True
    historical_data_points: int = 1000
    data_aggregation_interval_seconds: float = 60.0