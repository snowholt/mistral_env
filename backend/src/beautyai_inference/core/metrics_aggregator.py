"""
Metrics Aggregation System for Performance Monitoring.

Provides advanced metrics aggregation, time series analysis, and data processing
for the performance monitoring system.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import time
import statistics
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable

# Import shared types
from .performance_types import (
    MetricPoint, MetricType, PerformanceConfig,
    AggregationType, TimeWindow
)

import asyncio
import logging
import time
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque, defaultdict
from enum import Enum, auto
import bisect
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of metric aggregation."""
    SUM = auto()            # Sum of values
    AVERAGE = auto()        # Average value
    MIN = auto()            # Minimum value
    MAX = auto()            # Maximum value
    COUNT = auto()          # Count of data points
    RATE = auto()           # Rate of change
    PERCENTILE_50 = auto()  # Median (50th percentile)
    PERCENTILE_90 = auto()  # 90th percentile
    PERCENTILE_95 = auto()  # 95th percentile
    PERCENTILE_99 = auto()  # 99th percentile
    STD_DEV = auto()        # Standard deviation
    VARIANCE = auto()       # Variance
    FIRST = auto()          # First value in period
    LAST = auto()           # Last value in period


class TimeWindow(Enum):
    """Time window sizes for aggregation."""
    SECONDS_10 = 10
    SECONDS_30 = 30
    MINUTE_1 = 60
    MINUTE_5 = 300
    MINUTE_15 = 900
    HOUR_1 = 3600
    HOUR_6 = 21600
    DAY_1 = 86400


@dataclass
class AggregationRule:
    """Rule for metric aggregation."""
    metric_pattern: str                    # Pattern to match metric names (regex or glob)
    aggregation_type: AggregationType      # Type of aggregation to perform
    time_window: TimeWindow               # Time window for aggregation
    labels_to_group_by: List[str] = field(default_factory=list)  # Labels to group by
    enabled: bool = True                   # Whether rule is active
    retain_raw_data: bool = True          # Keep original data points
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AggregatedMetric:
    """Aggregated metric data point."""
    name: str
    aggregation_type: AggregationType
    value: Union[int, float]
    timestamp_start: float                # Start of aggregation window
    timestamp_end: float                  # End of aggregation window
    time_window: TimeWindow              # Aggregation window size
    data_points_count: int               # Number of original data points
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def window_duration_seconds(self) -> float:
        """Duration of the aggregation window in seconds."""
        return self.timestamp_end - self.timestamp_start


@dataclass
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: float
    raw_metrics: List[MetricPoint]
    aggregated_metrics: List[AggregatedMetric]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesBuffer:
    """Buffer for storing time-series metrics data."""
    
    def __init__(self, max_points: int = 10000, retention_seconds: int = 86400):
        self.max_points = max_points
        self.retention_seconds = retention_seconds
        self._data: deque = deque(maxlen=max_points)
        self._timestamps: deque = deque(maxlen=max_points)
    
    def add_point(self, metric: MetricPoint):
        """Add a data point to the buffer."""
        self._data.append(metric)
        self._timestamps.append(metric.timestamp)
        self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove data points older than retention period."""
        cutoff_time = time.time() - self.retention_seconds
        
        while self._timestamps and self._timestamps[0] < cutoff_time:
            self._timestamps.popleft()
            self._data.popleft()
    
    def get_points_in_range(self, start_time: float, end_time: float) -> List[MetricPoint]:
        """Get data points within a time range."""
        points = []
        
        for i, timestamp in enumerate(self._timestamps):
            if start_time <= timestamp <= end_time:
                points.append(self._data[i])
            elif timestamp > end_time:
                break
        
        return points
    
    def get_latest_points(self, count: int) -> List[MetricPoint]:
        """Get the latest N data points."""
        return list(self._data)[-count:] if count <= len(self._data) else list(self._data)
    
    def get_points_since(self, timestamp: float) -> List[MetricPoint]:
        """Get all points since a specific timestamp."""
        points = []
        
        for i, ts in enumerate(self._timestamps):
            if ts >= timestamp:
                points.append(self._data[i])
        
        return points
    
    @property
    def size(self) -> int:
        """Number of data points in buffer."""
        return len(self._data)
    
    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Timestamp of oldest data point."""
        return self._timestamps[0] if self._timestamps else None
    
    @property
    def newest_timestamp(self) -> Optional[float]:
        """Timestamp of newest data point."""
        return self._timestamps[-1] if self._timestamps else None


class MetricsAggregator:
    """
    Advanced metrics aggregation engine.
    
    Provides statistical aggregation, time-series processing, and trend analysis
    for performance metrics collected from various sources.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Data storage
        self._time_series_buffers: Dict[str, TimeSeriesBuffer] = {}  # metric_name -> buffer
        self._aggregated_data: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        
        # Aggregation rules
        self._aggregation_rules: List[AggregationRule] = []
        
        # Background processing
        self._aggregation_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Initialize default aggregation rules
        self._setup_default_rules()
        
        logger.info("Initialized MetricsAggregator")
    
    def _setup_default_rules(self):
        """Setup default aggregation rules."""
        # System metrics - averages over different time windows
        system_metrics_patterns = [
            "system_cpu_usage_percent",
            "system_memory_usage_percent",
            "process_cpu_percent",
            "process_memory_rss_bytes"
        ]
        
        for pattern in system_metrics_patterns:
            # 1-minute averages
            self._aggregation_rules.append(AggregationRule(
                metric_pattern=pattern,
                aggregation_type=AggregationType.AVERAGE,
                time_window=TimeWindow.MINUTE_1
            ))
            
            # 5-minute averages
            self._aggregation_rules.append(AggregationRule(
                metric_pattern=pattern,
                aggregation_type=AggregationType.AVERAGE,
                time_window=TimeWindow.MINUTE_5
            ))
        
        # Response time metrics - percentiles
        response_time_patterns = [
            "*response_time*",
            "*acquisition_time*"
        ]
        
        for pattern in response_time_patterns:
            for percentile_type in [AggregationType.PERCENTILE_50, AggregationType.PERCENTILE_90, AggregationType.PERCENTILE_95]:
                self._aggregation_rules.append(AggregationRule(
                    metric_pattern=pattern,
                    aggregation_type=percentile_type,
                    time_window=TimeWindow.MINUTE_1
                ))
        
        # Counter metrics - rates
        counter_patterns = [
            "*_requests_total",
            "*_requests_successful", 
            "*_requests_failed",
            "*_total_created",
            "*_total_destroyed"
        ]
        
        for pattern in counter_patterns:
            self._aggregation_rules.append(AggregationRule(
                metric_pattern=pattern,
                aggregation_type=AggregationType.RATE,
                time_window=TimeWindow.MINUTE_1
            ))
        
        # Connection pool metrics - grouped by pool
        pool_patterns = [
            "connection_pool_*"
        ]
        
        for pattern in pool_patterns:
            self._aggregation_rules.append(AggregationRule(
                metric_pattern=pattern,
                aggregation_type=AggregationType.AVERAGE,
                time_window=TimeWindow.MINUTE_1,
                labels_to_group_by=["pool"]
            ))
        
        # Circuit breaker metrics - grouped by circuit breaker
        cb_patterns = [
            "circuit_breaker_*"
        ]
        
        for pattern in cb_patterns:
            self._aggregation_rules.append(AggregationRule(
                metric_pattern=pattern,
                aggregation_type=AggregationType.AVERAGE,
                time_window=TimeWindow.MINUTE_1,
                labels_to_group_by=["circuit_breaker"]
            ))
    
    def add_aggregation_rule(self, rule: AggregationRule):
        """Add a custom aggregation rule."""
        self._aggregation_rules.append(rule)
        logger.info(f"Added aggregation rule: {rule.metric_pattern} -> {rule.aggregation_type.name}")
    
    def remove_aggregation_rule(self, metric_pattern: str, aggregation_type: AggregationType):
        """Remove an aggregation rule."""
        self._aggregation_rules = [
            rule for rule in self._aggregation_rules 
            if not (rule.metric_pattern == metric_pattern and rule.aggregation_type == aggregation_type)
        ]
    
    async def start(self):
        """Start the aggregation service."""
        logger.info("Starting metrics aggregation service")
        
        if not self._aggregation_task:
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        logger.info("Metrics aggregation service started")
    
    async def stop(self):
        """Stop the aggregation service."""
        logger.info("Stopping metrics aggregation service")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel aggregation task
        if self._aggregation_task and not self._aggregation_task.done():
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics aggregation service stopped")
    
    async def ingest_metrics(self, metrics: List[MetricPoint]):
        """Ingest raw metrics for aggregation."""
        for metric in metrics:
            # Get or create buffer for this metric
            if metric.name not in self._time_series_buffers:
                self._time_series_buffers[metric.name] = TimeSeriesBuffer(
                    max_points=self.config.historical_data_points,
                    retention_seconds=self.config.metrics_retention_seconds
                )
            
            # Add to buffer
            self._time_series_buffers[metric.name].add_point(metric)
    
    async def _aggregation_loop(self):
        """Main aggregation processing loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._process_aggregation_rules()
                
                # Wait for next aggregation interval
                await asyncio.sleep(self.config.data_aggregation_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(5)  # Brief delay on error
    
    async def _process_aggregation_rules(self):
        """Process all aggregation rules."""
        current_time = time.time()
        
        for rule in self._aggregation_rules:
            if not rule.enabled:
                continue
            
            try:
                # Find matching metrics
                matching_buffers = self._find_matching_buffers(rule.metric_pattern)
                
                for metric_name, buffer in matching_buffers.items():
                    aggregated_metric = await self._aggregate_metric_buffer(
                        metric_name, buffer, rule, current_time
                    )
                    
                    if aggregated_metric:
                        self._aggregated_data[metric_name].append(aggregated_metric)
                        
                        # Limit aggregated data size
                        if len(self._aggregated_data[metric_name]) > 1000:
                            self._aggregated_data[metric_name] = self._aggregated_data[metric_name][-500:]
                        
            except Exception as e:
                logger.warning(f"Error processing aggregation rule {rule.metric_pattern}: {e}")
    
    def _find_matching_buffers(self, pattern: str) -> Dict[str, TimeSeriesBuffer]:
        """Find metric buffers matching a pattern."""
        import fnmatch
        
        matching = {}
        for metric_name, buffer in self._time_series_buffers.items():
            if fnmatch.fnmatch(metric_name, pattern) or pattern == metric_name:
                matching[metric_name] = buffer
        
        return matching
    
    async def _aggregate_metric_buffer(self, metric_name: str, buffer: TimeSeriesBuffer,
                                     rule: AggregationRule, current_time: float) -> Optional[AggregatedMetric]:
        """Aggregate data from a metric buffer according to a rule."""
        # Calculate time window
        window_seconds = rule.time_window.value
        window_start = current_time - window_seconds
        window_end = current_time
        
        # Get data points in window
        data_points = buffer.get_points_in_range(window_start, window_end)
        
        if not data_points:
            return None
        
        # Group by labels if specified
        if rule.labels_to_group_by:
            grouped_data = self._group_by_labels(data_points, rule.labels_to_group_by)
            
            # For now, aggregate the first group (could be extended to handle multiple groups)
            if grouped_data:
                first_group_key = next(iter(grouped_data))
                data_points = grouped_data[first_group_key]
                group_labels = dict(zip(rule.labels_to_group_by, first_group_key))
            else:
                return None
        else:
            group_labels = {}
        
        # Extract values
        values = [point.value for point in data_points]
        
        if not values:
            return None
        
        # Perform aggregation
        aggregated_value = self._perform_aggregation(values, rule.aggregation_type)
        
        if aggregated_value is None:
            return None
        
        # Create aggregated metric
        return AggregatedMetric(
            name=f"{metric_name}_{rule.aggregation_type.name.lower()}_{rule.time_window.name.lower()}",
            aggregation_type=rule.aggregation_type,
            value=aggregated_value,
            timestamp_start=window_start,
            timestamp_end=window_end,
            time_window=rule.time_window,
            data_points_count=len(data_points),
            labels=group_labels,
            metadata={
                "original_metric": metric_name,
                "raw_data_retained": rule.retain_raw_data,
                **rule.metadata
            }
        )
    
    def _group_by_labels(self, data_points: List[MetricPoint], 
                        group_by_labels: List[str]) -> Dict[tuple, List[MetricPoint]]:
        """Group data points by specified labels."""
        groups = defaultdict(list)
        
        for point in data_points:
            # Create group key from specified labels
            key = tuple(point.labels.get(label, "") for label in group_by_labels)
            groups[key].append(point)
        
        return dict(groups)
    
    def _perform_aggregation(self, values: List[Union[int, float]], 
                           aggregation_type: AggregationType) -> Optional[Union[int, float]]:
        """Perform the specified aggregation on values."""
        if not values:
            return None
        
        try:
            if aggregation_type == AggregationType.SUM:
                return sum(values)
            
            elif aggregation_type == AggregationType.AVERAGE:
                return statistics.mean(values)
            
            elif aggregation_type == AggregationType.MIN:
                return min(values)
            
            elif aggregation_type == AggregationType.MAX:
                return max(values)
            
            elif aggregation_type == AggregationType.COUNT:
                return len(values)
            
            elif aggregation_type == AggregationType.RATE:
                # Calculate rate as change per second
                if len(values) >= 2:
                    return (values[-1] - values[0]) / len(values)
                else:
                    return 0.0
            
            elif aggregation_type == AggregationType.PERCENTILE_50:
                return statistics.median(values)
            
            elif aggregation_type == AggregationType.PERCENTILE_90:
                return self._calculate_percentile(values, 0.90)
            
            elif aggregation_type == AggregationType.PERCENTILE_95:
                return self._calculate_percentile(values, 0.95)
            
            elif aggregation_type == AggregationType.PERCENTILE_99:
                return self._calculate_percentile(values, 0.99)
            
            elif aggregation_type == AggregationType.STD_DEV:
                return statistics.stdev(values) if len(values) > 1 else 0.0
            
            elif aggregation_type == AggregationType.VARIANCE:
                return statistics.variance(values) if len(values) > 1 else 0.0
            
            elif aggregation_type == AggregationType.FIRST:
                return values[0]
            
            elif aggregation_type == AggregationType.LAST:
                return values[-1]
            
            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}")
                return None
        
        except (statistics.StatisticsError, ZeroDivisionError, ValueError) as e:
            logger.warning(f"Error performing aggregation {aggregation_type}: {e}")
            return None
    
    def _calculate_percentile(self, values: List[Union[int, float]], percentile: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    def get_aggregated_metrics(self, metric_name: Optional[str] = None,
                              aggregation_type: Optional[AggregationType] = None,
                              time_window: Optional[TimeWindow] = None,
                              duration_seconds: Optional[int] = None) -> List[AggregatedMetric]:
        """Get aggregated metrics with optional filtering."""
        all_metrics = []
        
        # Collect all aggregated metrics
        for metrics_list in self._aggregated_data.values():
            all_metrics.extend(metrics_list)
        
        # Apply filters
        if metric_name:
            all_metrics = [m for m in all_metrics if metric_name in m.name]
        
        if aggregation_type:
            all_metrics = [m for m in all_metrics if m.aggregation_type == aggregation_type]
        
        if time_window:
            all_metrics = [m for m in all_metrics if m.time_window == time_window]
        
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            all_metrics = [m for m in all_metrics if m.timestamp_end >= cutoff_time]
        
        # Sort by timestamp
        return sorted(all_metrics, key=lambda x: x.timestamp_end, reverse=True)
    
    def get_time_series_data(self, metric_name: str, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get time series data for a specific metric."""
        if metric_name not in self._time_series_buffers:
            return {"error": f"Metric {metric_name} not found"}
        
        buffer = self._time_series_buffers[metric_name]
        cutoff_time = time.time() - duration_seconds
        
        # Get raw data points
        raw_points = buffer.get_points_since(cutoff_time)
        
        # Get aggregated data
        aggregated_metrics = self.get_aggregated_metrics(
            metric_name=metric_name,
            duration_seconds=duration_seconds
        )
        
        return {
            "metric_name": metric_name,
            "duration_seconds": duration_seconds,
            "raw_data_points": len(raw_points),
            "aggregated_data_points": len(aggregated_metrics),
            "raw_data": [
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "labels": point.labels
                }
                for point in raw_points
            ],
            "aggregated_data": [
                {
                    "name": metric.name,
                    "aggregation_type": metric.aggregation_type.name,
                    "value": metric.value,
                    "timestamp_start": metric.timestamp_start,
                    "timestamp_end": metric.timestamp_end,
                    "time_window": metric.time_window.name,
                    "data_points_count": metric.data_points_count,
                    "labels": metric.labels
                }
                for metric in aggregated_metrics
            ]
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics data."""
        current_time = time.time()
        
        # Buffer statistics
        buffer_stats = {}
        total_raw_points = 0
        
        for metric_name, buffer in self._time_series_buffers.items():
            buffer_stats[metric_name] = {
                "size": buffer.size,
                "oldest_timestamp": buffer.oldest_timestamp,
                "newest_timestamp": buffer.newest_timestamp,
                "age_seconds": current_time - buffer.oldest_timestamp if buffer.oldest_timestamp else 0
            }
            total_raw_points += buffer.size
        
        # Aggregated data statistics
        total_aggregated_points = sum(len(metrics) for metrics in self._aggregated_data.values())
        
        return {
            "timestamp": current_time,
            "total_raw_metrics": len(self._time_series_buffers),
            "total_raw_data_points": total_raw_points,
            "total_aggregated_metrics": len(self._aggregated_data),
            "total_aggregated_data_points": total_aggregated_points,
            "aggregation_rules_count": len(self._aggregation_rules),
            "active_rules": len([r for r in self._aggregation_rules if r.enabled]),
            "buffer_statistics": buffer_stats,
            "memory_usage": {
                "buffers_mb": total_raw_points * 0.001,  # Rough estimate
                "aggregated_mb": total_aggregated_points * 0.001
            }
        }
    
    async def create_snapshot(self, include_raw_data: bool = True) -> MetricsSnapshot:
        """Create a complete snapshot of current metrics state."""
        current_time = time.time()
        
        # Collect raw metrics from all buffers
        raw_metrics = []
        if include_raw_data:
            for buffer in self._time_series_buffers.values():
                raw_metrics.extend(buffer.get_latest_points(10))  # Last 10 points per metric
        
        # Collect aggregated metrics
        aggregated_metrics = self.get_aggregated_metrics(duration_seconds=3600)  # Last hour
        
        return MetricsSnapshot(
            timestamp=current_time,
            raw_metrics=raw_metrics,
            aggregated_metrics=aggregated_metrics,
            metadata={
                "total_metric_types": len(self._time_series_buffers),
                "raw_points_included": len(raw_metrics),
                "aggregated_points_included": len(aggregated_metrics),
                "snapshot_duration_seconds": 3600
            }
        )
    
    def export_data(self, format_type: str = "json", duration_seconds: int = 3600) -> str:
        """Export metrics data in specified format."""
        if format_type.lower() != "json":
            raise ValueError(f"Unsupported export format: {format_type}")
        
        # Get all data for export
        export_data = {
            "export_timestamp": time.time(),
            "duration_seconds": duration_seconds,
            "metrics": {}
        }
        
        # Export time series data for each metric
        for metric_name in self._time_series_buffers.keys():
            export_data["metrics"][metric_name] = self.get_time_series_data(
                metric_name, duration_seconds
            )
        
        # Add summary information
        export_data["summary"] = self.get_metrics_summary()
        
        return json.dumps(export_data, indent=2, default=str)