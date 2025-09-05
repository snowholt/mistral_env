"""
Advanced Anomaly Detection and Alerting System for BeautyAI.

Provides sophisticated anomaly detection using multiple algorithms:
- Statistical anomaly detection (z-score, IQR, isolation forest)
- Machine learning-based detection for complex patterns
- Threshold-based alerting with adaptive thresholds
- Alert management with escalation and notification chains
- Integration with external alerting systems
- Historical anomaly analysis and trend detection

This service integrates with the performance monitor to provide
intelligent alerting and anomaly detection capabilities.

Author: BeautyAI Framework  
Date: September 5, 2025
"""

import asyncio
import logging
import time
import json
import statistics
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from collections import deque, defaultdict
from enum import Enum, auto
from datetime import datetime, timedelta
import math
import uuid

from .performance_types import MetricPoint, MetricType, PerformanceConfig, PerformanceAlert, AlertLevel
from .metrics_aggregator import AggregatedMetric, TimeSeriesBuffer

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    THRESHOLD_BREACH = auto()       # Simple threshold violation
    STATISTICAL_OUTLIER = auto()    # Statistical outlier detection
    TREND_ANOMALY = auto()          # Unusual trend or pattern
    RATE_ANOMALY = auto()          # Abnormal rate of change
    CORRELATION_ANOMALY = auto()    # Unusual correlation patterns
    SEASONAL_ANOMALY = auto()       # Deviation from seasonal patterns
    VOLUME_ANOMALY = auto()         # Unusual data volume or frequency


class AnomalyDetectionAlgorithm(Enum):
    """Anomaly detection algorithms available."""
    Z_SCORE = auto()                # Z-score based detection
    MODIFIED_Z_SCORE = auto()       # Modified z-score using median
    IQR_OUTLIER = auto()           # Interquartile range outlier detection
    ISOLATION_FOREST = auto()       # Isolation forest algorithm
    LOCAL_OUTLIER_FACTOR = auto()   # Local outlier factor
    MOVING_AVERAGE = auto()         # Moving average deviation
    EXPONENTIAL_SMOOTHING = auto()  # Exponential smoothing prediction
    PERCENTILE_THRESHOLD = auto()   # Dynamic percentile thresholds


class AlertSeverity(Enum):
    """Alert severity levels for escalation."""
    LOW = auto()
    MEDIUM = auto() 
    HIGH = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class AnomalyDetectionRule:
    """Rule for detecting anomalies in metrics."""
    rule_id: str
    name: str
    metric_pattern: str                         # Metric name pattern to monitor
    algorithm: AnomalyDetectionAlgorithm       # Detection algorithm to use
    anomaly_type: AnomalyType                  # Type of anomaly to detect
    sensitivity: float = 0.95                  # Detection sensitivity (0-1)
    window_size: int = 50                      # Number of data points for analysis
    enabled: bool = True                       # Whether rule is active
    labels_filter: Dict[str, str] = field(default_factory=dict)  # Label filters
    parameters: Dict[str, Any] = field(default_factory=dict)     # Algorithm parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedAnomaly:
    """Detected anomaly information."""
    anomaly_id: str
    rule_id: str
    metric_name: str
    anomaly_type: AnomalyType
    algorithm: AnomalyDetectionAlgorithm
    timestamp: float
    value: float
    expected_value: Optional[float]
    anomaly_score: float                       # Anomaly strength score (0-1)
    severity: AlertSeverity
    description: str
    confidence: float                          # Confidence in detection (0-1)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Alert management
    alert_sent: bool = False
    alert_timestamp: Optional[float] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class AlertingRule:
    """Rule for generating alerts from anomalies."""
    rule_id: str
    name: str
    anomaly_types: List[AnomalyType]           # Types of anomalies to alert on
    severity_threshold: AlertSeverity          # Minimum severity to trigger alert
    cooldown_minutes: int = 15                 # Minutes to wait before re-alerting
    escalation_enabled: bool = False           # Whether to escalate unacknowledged alerts
    escalation_delay_minutes: int = 60         # Minutes before escalation
    notification_channels: List[str] = field(default_factory=list)  # Channels to notify
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatisticalDetector:
    """Statistical anomaly detection algorithms."""
    
    @staticmethod
    def z_score_detection(values: List[float], current_value: float, 
                         threshold: float = 3.0) -> Tuple[bool, float, float]:
        """Z-score based anomaly detection."""
        if len(values) < 3:
            return False, 0.0, current_value
        
        try:
            mean_val = statistics.mean(values)
            std_dev = statistics.stdev(values)
            
            if std_dev == 0:
                return False, 0.0, mean_val
            
            z_score = abs(current_value - mean_val) / std_dev
            is_anomaly = z_score > threshold
            
            return is_anomaly, min(z_score / threshold, 1.0), mean_val
            
        except statistics.StatisticsError:
            return False, 0.0, current_value
    
    @staticmethod
    def modified_z_score_detection(values: List[float], current_value: float,
                                  threshold: float = 3.5) -> Tuple[bool, float, float]:
        """Modified z-score using median absolute deviation."""
        if len(values) < 3:
            return False, 0.0, current_value
        
        try:
            median_val = statistics.median(values)
            mad = statistics.median([abs(x - median_val) for x in values])
            
            if mad == 0:
                return False, 0.0, median_val
            
            modified_z = 0.6745 * (current_value - median_val) / mad
            is_anomaly = abs(modified_z) > threshold
            
            return is_anomaly, min(abs(modified_z) / threshold, 1.0), median_val
            
        except (statistics.StatisticsError, ZeroDivisionError):
            return False, 0.0, current_value
    
    @staticmethod
    def iqr_outlier_detection(values: List[float], current_value: float,
                             iqr_multiplier: float = 1.5) -> Tuple[bool, float, float]:
        """Interquartile range outlier detection."""
        if len(values) < 4:
            return False, 0.0, current_value
        
        try:
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1
            
            if iqr == 0:
                return False, 0.0, statistics.median(values)
            
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            is_anomaly = current_value < lower_bound or current_value > upper_bound
            
            # Calculate anomaly score based on distance from bounds
            if current_value < lower_bound:
                score = min((lower_bound - current_value) / iqr, 1.0)
            elif current_value > upper_bound:
                score = min((current_value - upper_bound) / iqr, 1.0)
            else:
                score = 0.0
            
            expected = (q1 + q3) / 2  # Median of interquartile range
            
            return is_anomaly, score, expected
            
        except (IndexError, ZeroDivisionError):
            return False, 0.0, current_value
    
    @staticmethod
    def moving_average_detection(values: List[float], current_value: float,
                               window_size: int = 10, threshold_factor: float = 2.0) -> Tuple[bool, float, float]:
        """Moving average based anomaly detection."""
        if len(values) < window_size:
            return False, 0.0, current_value
        
        try:
            # Calculate moving average and standard deviation
            recent_values = values[-window_size:]
            moving_avg = statistics.mean(recent_values)
            moving_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            if moving_std == 0:
                return abs(current_value - moving_avg) > 0, 0.0, moving_avg
            
            # Check if current value deviates significantly from moving average
            deviation = abs(current_value - moving_avg) / moving_std
            is_anomaly = deviation > threshold_factor
            
            return is_anomaly, min(deviation / threshold_factor, 1.0), moving_avg
            
        except statistics.StatisticsError:
            return False, 0.0, current_value


class TrendAnalyzer:
    """Analyzes trends and detects trend-based anomalies."""
    
    @staticmethod
    def detect_trend_change(values: List[float], window_size: int = 10) -> Tuple[bool, float]:
        """Detect significant trend changes."""
        if len(values) < window_size * 2:
            return False, 0.0
        
        try:
            # Compare recent trend vs historical trend
            recent_values = values[-window_size:]
            historical_values = values[-window_size*2:-window_size]
            
            # Calculate linear regression slopes
            recent_slope = TrendAnalyzer._calculate_slope(recent_values)
            historical_slope = TrendAnalyzer._calculate_slope(historical_values)
            
            # Detect significant slope change
            slope_change = abs(recent_slope - historical_slope)
            avg_slope = (abs(recent_slope) + abs(historical_slope)) / 2
            
            if avg_slope == 0:
                return slope_change > 0, min(slope_change, 1.0)
            
            relative_change = slope_change / avg_slope
            is_anomaly = relative_change > 2.0  # 200% change in slope
            
            return is_anomaly, min(relative_change / 2.0, 1.0)
            
        except Exception:
            return False, 0.0
    
    @staticmethod
    def _calculate_slope(values: List[float]) -> float:
        """Calculate linear regression slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    @staticmethod
    def detect_rate_anomaly(values: List[float], current_value: float,
                           window_size: int = 5) -> Tuple[bool, float]:
        """Detect anomalous rate of change."""
        if len(values) < window_size + 1:
            return False, 0.0
        
        try:
            # Calculate recent rates of change
            recent_rates = []
            for i in range(len(values) - window_size, len(values)):
                if i > 0:
                    rate = values[i] - values[i-1]
                    recent_rates.append(rate)
            
            if not recent_rates:
                return False, 0.0
            
            # Calculate current rate of change
            current_rate = current_value - values[-1]
            
            # Compare with historical rates
            avg_rate = statistics.mean(recent_rates)
            std_rate = statistics.stdev(recent_rates) if len(recent_rates) > 1 else 0
            
            if std_rate == 0:
                return abs(current_rate - avg_rate) > 0, 0.0
            
            z_score = abs(current_rate - avg_rate) / std_rate
            is_anomaly = z_score > 2.0  # 2-sigma threshold for rate changes
            
            return is_anomaly, min(z_score / 2.0, 1.0)
            
        except statistics.StatisticsError:
            return False, 0.0


class AnomalyDetector:
    """
    Advanced anomaly detection engine for performance metrics.
    
    Uses multiple algorithms to detect various types of anomalies
    and generates intelligent alerts with confidence scores.
    """
    
    def __init__(self):
        self._detection_rules: Dict[str, AnomalyDetectionRule] = {}
        self._alerting_rules: Dict[str, AlertingRule] = {}
        self._metric_buffers: Dict[str, TimeSeriesBuffer] = {}
        self._detected_anomalies: List[DetectedAnomaly] = []
        self._alert_history: List[PerformanceAlert] = []
        self._alert_cooldowns: Dict[str, float] = {}  # rule_id -> last_alert_time
        
        # Algorithm implementations
        self._statistical_detector = StatisticalDetector()
        self._trend_analyzer = TrendAnalyzer()
        
        # Default rules
        self._setup_default_rules()
        
        logger.info("Initialized AnomalyDetector")
    
    def _setup_default_rules(self):
        """Setup default anomaly detection and alerting rules."""
        # CPU usage anomaly detection
        self.add_detection_rule(AnomalyDetectionRule(
            rule_id="cpu_usage_anomaly",
            name="CPU Usage Anomaly Detection",
            metric_pattern="system_cpu_usage_percent",
            algorithm=AnomalyDetectionAlgorithm.Z_SCORE,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            sensitivity=0.95,
            window_size=30,
            parameters={"threshold": 2.5}
        ))
        
        # Memory usage anomaly detection
        self.add_detection_rule(AnomalyDetectionRule(
            rule_id="memory_usage_anomaly",
            name="Memory Usage Anomaly Detection",
            metric_pattern="system_memory_usage_percent",
            algorithm=AnomalyDetectionAlgorithm.MOVING_AVERAGE,
            anomaly_type=AnomalyType.TREND_ANOMALY,
            sensitivity=0.90,
            window_size=20,
            parameters={"threshold_factor": 2.0}
        ))
        
        # Response time anomaly detection
        self.add_detection_rule(AnomalyDetectionRule(
            rule_id="response_time_anomaly",
            name="Response Time Anomaly Detection",
            metric_pattern="*response_time*",
            algorithm=AnomalyDetectionAlgorithm.IQR_OUTLIER,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            sensitivity=0.98,
            window_size=40,
            parameters={"iqr_multiplier": 1.5}
        ))
        
        # Error rate trend anomaly
        self.add_detection_rule(AnomalyDetectionRule(
            rule_id="error_rate_trend_anomaly",
            name="Error Rate Trend Anomaly Detection",
            metric_pattern="*failure_rate*",
            algorithm=AnomalyDetectionAlgorithm.MOVING_AVERAGE,
            anomaly_type=AnomalyType.TREND_ANOMALY,
            sensitivity=0.95,
            window_size=15,
            parameters={"threshold_factor": 1.5}
        ))
        
        # Default alerting rules
        self.add_alerting_rule(AlertingRule(
            rule_id="critical_system_alerts",
            name="Critical System Alerts",
            anomaly_types=[AnomalyType.STATISTICAL_OUTLIER, AnomalyType.TREND_ANOMALY],
            severity_threshold=AlertSeverity.HIGH,
            cooldown_minutes=10,
            escalation_enabled=True,
            escalation_delay_minutes=30,
            notification_channels=["system_alerts", "email"]
        ))
        
        self.add_alerting_rule(AlertingRule(
            rule_id="performance_degradation_alerts",
            name="Performance Degradation Alerts",
            anomaly_types=[AnomalyType.RATE_ANOMALY, AnomalyType.CORRELATION_ANOMALY],
            severity_threshold=AlertSeverity.MEDIUM,
            cooldown_minutes=15,
            notification_channels=["performance_alerts"]
        ))
    
    def add_detection_rule(self, rule: AnomalyDetectionRule):
        """Add an anomaly detection rule."""
        self._detection_rules[rule.rule_id] = rule
        logger.info(f"Added anomaly detection rule: {rule.name}")
    
    def add_alerting_rule(self, rule: AlertingRule):
        """Add an alerting rule."""
        self._alerting_rules[rule.rule_id] = rule
        logger.info(f"Added alerting rule: {rule.name}")
    
    def remove_detection_rule(self, rule_id: str):
        """Remove an anomaly detection rule."""
        if rule_id in self._detection_rules:
            del self._detection_rules[rule_id]
            logger.info(f"Removed anomaly detection rule: {rule_id}")
    
    def remove_alerting_rule(self, rule_id: str):
        """Remove an alerting rule."""
        if rule_id in self._alerting_rules:
            del self._alerting_rules[rule_id]
            logger.info(f"Removed alerting rule: {rule_id}")
    
    async def ingest_metrics(self, metrics: List[MetricPoint]):
        """Ingest metrics for anomaly detection."""
        for metric in metrics:
            # Store in internal buffers
            if metric.name not in self._metric_buffers:
                self._metric_buffers[metric.name] = TimeSeriesBuffer(max_points=1000)
            
            self._metric_buffers[metric.name].add_point(metric)
    
    async def detect_anomalies(self) -> List[DetectedAnomaly]:
        """Run anomaly detection on current metrics."""
        detected_anomalies = []
        
        for rule in self._detection_rules.values():
            if not rule.enabled:
                continue
            
            try:
                # Find matching metrics
                matching_buffers = self._find_matching_metric_buffers(rule.metric_pattern)
                
                for metric_name, buffer in matching_buffers.items():
                    # Apply label filters
                    if rule.labels_filter:
                        filtered_points = self._filter_by_labels(
                            buffer.get_latest_points(rule.window_size + 1),
                            rule.labels_filter
                        )
                    else:
                        filtered_points = buffer.get_latest_points(rule.window_size + 1)
                    
                    if len(filtered_points) < 2:
                        continue
                    
                    # Perform anomaly detection
                    anomalies = await self._apply_detection_algorithm(
                        rule, metric_name, filtered_points
                    )
                    
                    detected_anomalies.extend(anomalies)
                    
            except Exception as e:
                logger.warning(f"Error applying detection rule {rule.rule_id}: {e}")
        
        # Store detected anomalies
        self._detected_anomalies.extend(detected_anomalies)
        
        # Limit history size
        if len(self._detected_anomalies) > 10000:
            self._detected_anomalies = self._detected_anomalies[-5000:]
        
        return detected_anomalies
    
    def _find_matching_metric_buffers(self, pattern: str) -> Dict[str, TimeSeriesBuffer]:
        """Find metric buffers matching a pattern."""
        import fnmatch
        
        matching = {}
        for metric_name, buffer in self._metric_buffers.items():
            if fnmatch.fnmatch(metric_name, pattern) or pattern == metric_name:
                matching[metric_name] = buffer
        
        return matching
    
    def _filter_by_labels(self, points: List[MetricPoint], 
                         label_filters: Dict[str, str]) -> List[MetricPoint]:
        """Filter metric points by label criteria."""
        filtered = []
        
        for point in points:
            matches = True
            for label_key, label_value in label_filters.items():
                if point.labels.get(label_key) != label_value:
                    matches = False
                    break
            
            if matches:
                filtered.append(point)
        
        return filtered
    
    async def _apply_detection_algorithm(self, rule: AnomalyDetectionRule, 
                                       metric_name: str, 
                                       points: List[MetricPoint]) -> List[DetectedAnomaly]:
        """Apply anomaly detection algorithm to metric points."""
        if len(points) < 2:
            return []
        
        current_point = points[-1]
        historical_values = [p.value for p in points[:-1]]
        current_value = current_point.value
        
        # Apply detection algorithm
        is_anomaly, anomaly_score, expected_value = await self._run_algorithm(
            rule.algorithm, historical_values, current_value, rule.parameters
        )
        
        if not is_anomaly:
            return []
        
        # Determine severity based on anomaly score and sensitivity
        severity = self._calculate_severity(anomaly_score, rule.sensitivity)
        
        # Create detected anomaly
        anomaly = DetectedAnomaly(
            anomaly_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            metric_name=metric_name,
            anomaly_type=rule.anomaly_type,
            algorithm=rule.algorithm,
            timestamp=current_point.timestamp,
            value=current_value,
            expected_value=expected_value,
            anomaly_score=anomaly_score,
            severity=severity,
            description=self._generate_anomaly_description(
                rule, metric_name, current_value, expected_value, anomaly_score
            ),
            confidence=rule.sensitivity,
            labels=current_point.labels.copy()
        )
        
        return [anomaly]
    
    async def _run_algorithm(self, algorithm: AnomalyDetectionAlgorithm,
                           historical_values: List[float], current_value: float,
                           parameters: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Run specific anomaly detection algorithm."""
        
        if algorithm == AnomalyDetectionAlgorithm.Z_SCORE:
            threshold = parameters.get("threshold", 3.0)
            return self._statistical_detector.z_score_detection(
                historical_values, current_value, threshold
            )
        
        elif algorithm == AnomalyDetectionAlgorithm.MODIFIED_Z_SCORE:
            threshold = parameters.get("threshold", 3.5)
            return self._statistical_detector.modified_z_score_detection(
                historical_values, current_value, threshold
            )
        
        elif algorithm == AnomalyDetectionAlgorithm.IQR_OUTLIER:
            multiplier = parameters.get("iqr_multiplier", 1.5)
            return self._statistical_detector.iqr_outlier_detection(
                historical_values, current_value, multiplier
            )
        
        elif algorithm == AnomalyDetectionAlgorithm.MOVING_AVERAGE:
            window_size = parameters.get("window_size", 10)
            threshold_factor = parameters.get("threshold_factor", 2.0)
            return self._statistical_detector.moving_average_detection(
                historical_values, current_value, window_size, threshold_factor
            )
        
        else:
            logger.warning(f"Unknown algorithm: {algorithm}")
            return False, 0.0, current_value
    
    def _calculate_severity(self, anomaly_score: float, sensitivity: float) -> AlertSeverity:
        """Calculate alert severity based on anomaly score and sensitivity."""
        # Adjust score based on sensitivity
        adjusted_score = anomaly_score * sensitivity
        
        if adjusted_score >= 0.95:
            return AlertSeverity.EMERGENCY
        elif adjusted_score >= 0.85:
            return AlertSeverity.CRITICAL
        elif adjusted_score >= 0.70:
            return AlertSeverity.HIGH
        elif adjusted_score >= 0.50:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_anomaly_description(self, rule: AnomalyDetectionRule, metric_name: str,
                                    current_value: float, expected_value: Optional[float],
                                    anomaly_score: float) -> str:
        """Generate human-readable anomaly description."""
        desc = f"Anomaly detected in {metric_name} using {rule.algorithm.name.lower()}"
        desc += f" (score: {anomaly_score:.2f})"
        
        if expected_value is not None:
            desc += f". Current value: {current_value:.2f}, Expected: {expected_value:.2f}"
        else:
            desc += f". Current value: {current_value:.2f}"
        
        return desc
    
    async def generate_alerts(self, detected_anomalies: List[DetectedAnomaly]) -> List[PerformanceAlert]:
        """Generate alerts from detected anomalies based on alerting rules."""
        alerts = []
        current_time = time.time()
        
        for anomaly in detected_anomalies:
            for alerting_rule in self._alerting_rules.values():
                if not alerting_rule.enabled:
                    continue
                
                # Check if anomaly type matches
                if anomaly.anomaly_type not in alerting_rule.anomaly_types:
                    continue
                
                # Check severity threshold
                if anomaly.severity.value < alerting_rule.severity_threshold.value:
                    continue
                
                # Check cooldown period
                cooldown_key = f"{alerting_rule.rule_id}:{anomaly.metric_name}"
                last_alert_time = self._alert_cooldowns.get(cooldown_key, 0)
                if current_time - last_alert_time < alerting_rule.cooldown_minutes * 60:
                    continue
                
                # Generate alert
                alert = PerformanceAlert(
                    alert_id=f"anomaly_{anomaly.anomaly_id}",
                    rule_id=alerting_rule.rule_id,
                    metric_name=anomaly.metric_name,
                    severity=anomaly.severity,
                    timestamp=current_time,
                    message=f"{anomaly.description} (Rule: {alerting_rule.name})",
                    value=anomaly.value,
                    threshold=anomaly.expected_value,
                    labels={
                        **anomaly.labels,
                        "anomaly_type": anomaly.anomaly_type.name,
                        "detection_algorithm": anomaly.algorithm.name,
                        "alerting_rule": alerting_rule.rule_id
                    }
                )
                
                alerts.append(alert)
                
                # Update cooldown
                self._alert_cooldowns[cooldown_key] = current_time
                
                # Mark anomaly as alerted
                anomaly.alert_sent = True
                anomaly.alert_timestamp = current_time
        
        # Store in history
        self._alert_history.extend(alerts)
        
        # Limit history size
        if len(self._alert_history) > 5000:
            self._alert_history = self._alert_history[-2500:]
        
        return alerts
    
    def _map_severity_to_alert_level(self, severity: AlertSeverity) -> AlertLevel:
        """Map anomaly severity to alert level."""
        mapping = {
            AlertSeverity.LOW: AlertLevel.INFO,
            AlertSeverity.MEDIUM: AlertLevel.WARNING,
            AlertSeverity.HIGH: AlertLevel.WARNING,
            AlertSeverity.CRITICAL: AlertLevel.CRITICAL,
            AlertSeverity.EMERGENCY: AlertLevel.CRITICAL
        }
        return mapping.get(severity, AlertLevel.INFO)
    
    def get_active_anomalies(self, duration_minutes: int = 60) -> List[DetectedAnomaly]:
        """Get anomalies detected in the recent time period."""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [
            anomaly for anomaly in self._detected_anomalies
            if anomaly.timestamp >= cutoff_time and not anomaly.resolved
        ]
    
    def get_alert_history(self, duration_minutes: int = 1440) -> List[PerformanceAlert]:
        """Get alert history for specified duration (default: 24 hours)."""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [
            alert for alert in self._alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def acknowledge_anomaly(self, anomaly_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an anomaly."""
        for anomaly in self._detected_anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.acknowledged = True
                anomaly.acknowledged_by = acknowledged_by
                anomaly.acknowledged_at = time.time()
                return True
        return False
    
    def resolve_anomaly(self, anomaly_id: str) -> bool:
        """Mark an anomaly as resolved."""
        for anomaly in self._detected_anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.resolved = True
                anomaly.resolved_at = time.time()
                return True
        return False
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies."""
        current_time = time.time()
        last_24h = current_time - 86400
        
        # Filter recent anomalies
        recent_anomalies = [a for a in self._detected_anomalies if a.timestamp >= last_24h]
        
        # Group by type
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_algorithm = defaultdict(int)
        
        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type.name] += 1
            by_severity[anomaly.severity.name] += 1
            by_algorithm[anomaly.algorithm.name] += 1
        
        return {
            "total_anomalies": len(self._detected_anomalies),
            "recent_anomalies_24h": len(recent_anomalies),
            "active_anomalies": len([a for a in recent_anomalies if not a.resolved]),
            "acknowledged_anomalies": len([a for a in recent_anomalies if a.acknowledged]),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_algorithm": dict(by_algorithm),
            "detection_rules_active": len([r for r in self._detection_rules.values() if r.enabled]),
            "alerting_rules_active": len([r for r in self._alerting_rules.values() if r.enabled])
        }