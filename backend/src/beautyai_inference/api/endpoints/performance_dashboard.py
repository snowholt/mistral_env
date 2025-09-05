"""
Performance Dashboard API Endpoints for BeautyAI.

Provides REST API endpoints for accessing performance monitoring data:
- Real-time metrics dashboard
- Historical data visualization  
- Anomaly and alert management
- System health status
- Performance analytics and reports
- Export functionality for external tools

These endpoints integrate with the performance monitor, metrics aggregator,
and anomaly detector to provide comprehensive monitoring dashboards.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
import io

from ...core.performance_monitor import (
    get_performance_monitor, 
    PerformanceMonitor
)
from ...core.performance_types import (
    PerformanceConfig,
    MetricType,
    AlertLevel
)
from ...core.metrics_aggregator import (
    MetricsAggregator, 
    AggregationType, 
    TimeWindow,
    AggregationRule
)
from ...core.anomaly_detector import (
    AnomalyDetector,
    DetectedAnomaly,
    AnomalyDetectionRule,
    AlertSeverity
)

logger = logging.getLogger(__name__)

# Create API router
performance_router = APIRouter(prefix="/api/v1/performance", tags=["performance"])


def get_monitor_dependency() -> Optional[PerformanceMonitor]:
    """Dependency to get performance monitor instance."""
    return get_performance_monitor()


@performance_router.get("/dashboard")
async def get_performance_dashboard(
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get comprehensive performance dashboard data.
    
    Returns current system status, active alerts, recent metrics,
    and performance summaries.
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        dashboard_data = await monitor.get_performance_dashboard()
        return JSONResponse(content=dashboard_data)
    
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")


@performance_router.get("/metrics/current")
async def get_current_metrics(
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get current snapshot of all metrics or filtered by name.
    
    Args:
        metric_name: Optional metric name filter
    
    Returns:
        Current metrics data grouped by metric name
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        current_metrics = await monitor.get_current_metrics()
        
        # Filter by metric name if provided
        if metric_name:
            filtered_metrics = {}
            for name, data in current_metrics.items():
                if metric_name.lower() in name.lower():
                    filtered_metrics[name] = data
            current_metrics = filtered_metrics
        
        return JSONResponse(content={
            "timestamp": time.time(),
            "metric_count": len(current_metrics),
            "metrics": current_metrics
        })
    
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current metrics")


@performance_router.get("/metrics/historical")
async def get_historical_metrics(
    metric_name: str = Query(..., description="Metric name"),
    duration_seconds: int = Query(3600, description="Duration in seconds"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get historical data for a specific metric.
    
    Args:
        metric_name: Name of the metric
        duration_seconds: How far back to look (default: 1 hour)
    
    Returns:
        Historical metric data points
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Get historical data from metrics collector
        historical_data = monitor.metrics_collector.get_historical_metrics(
            metric_name=metric_name,
            duration_seconds=duration_seconds
        )
        
        # Format data for API response
        data_points = []
        for metric in historical_data:
            data_points.append({
                "timestamp": metric.timestamp,
                "value": metric.value,
                "metric_type": metric.metric_type.name,
                "labels": metric.labels,
                "metadata": metric.metadata
            })
        
        return JSONResponse(content={
            "metric_name": metric_name,
            "duration_seconds": duration_seconds,
            "data_point_count": len(data_points),
            "data_points": data_points
        })
    
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get historical metrics")


@performance_router.get("/system/status")
async def get_system_status(
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get current system health status and resource utilization.
    
    Returns:
        System status summary including CPU, memory, network, and service health
    """
    if not monitor:
        return JSONResponse(content={
            "status": "monitoring_disabled",
            "timestamp": time.time(),
            "monitoring_available": False
        })
    
    try:
        # Get current metrics
        current_metrics = await monitor.get_current_metrics()
        
        # Extract system metrics
        system_status = {
            "timestamp": time.time(),
            "monitoring_available": True,
            "status": "healthy"  # Will be updated based on metrics
        }
        
        # CPU status
        cpu_metrics = [m for name, m in current_metrics.items() if "cpu_usage_percent" in name]
        if cpu_metrics:
            latest_cpu = cpu_metrics[0][-1] if cpu_metrics[0] else None
            if latest_cpu:
                cpu_value = latest_cpu["value"]
                system_status["cpu"] = {
                    "usage_percent": cpu_value,
                    "status": "critical" if cpu_value > 95 else "warning" if cpu_value > 80 else "healthy"
                }
        
        # Memory status  
        memory_metrics = [m for name, m in current_metrics.items() if "memory_usage_percent" in name]
        if memory_metrics:
            latest_memory = memory_metrics[0][-1] if memory_metrics[0] else None
            if latest_memory:
                memory_value = latest_memory["value"]
                system_status["memory"] = {
                    "usage_percent": memory_value,
                    "status": "critical" if memory_value > 95 else "warning" if memory_value > 85 else "healthy"
                }
        
        # Connection pool status
        pool_metrics = [m for name, m in current_metrics.items() if "connection_pool" in name]
        if pool_metrics:
            # Get pool utilization
            utilization_metrics = [m for name, m in current_metrics.items() if "pool_utilization" in name]
            if utilization_metrics:
                latest_util = utilization_metrics[0][-1] if utilization_metrics[0] else None
                if latest_util:
                    util_value = latest_util["value"]
                    system_status["connection_pool"] = {
                        "utilization_percent": util_value,
                        "status": "warning" if util_value > 80 else "healthy"
                    }
        
        # Circuit breaker status
        cb_metrics = [m for name, m in current_metrics.items() if "circuit_breaker" in name]
        if cb_metrics:
            # Check for open circuit breakers
            state_metrics = [m for name, m in current_metrics.items() if "circuit_breaker_state" in name]
            open_breakers = 0
            for state_metric_list in state_metrics:
                if state_metric_list:
                    latest_state = state_metric_list[-1]
                    if latest_state["value"] == 1:  # OPEN state
                        open_breakers += 1
            
            system_status["circuit_breakers"] = {
                "total_monitored": len(state_metrics),
                "open_breakers": open_breakers,
                "status": "critical" if open_breakers > 0 else "healthy"
            }
        
        # Determine overall status
        component_statuses = []
        for component in ["cpu", "memory", "connection_pool", "circuit_breakers"]:
            if component in system_status:
                component_statuses.append(system_status[component]["status"])
        
        if "critical" in component_statuses:
            system_status["status"] = "critical"
        elif "warning" in component_statuses:
            system_status["status"] = "warning"
        else:
            system_status["status"] = "healthy"
        
        return JSONResponse(content=system_status)
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@performance_router.get("/alerts/active")
async def get_active_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get currently active performance alerts.
    
    Args:
        level: Optional filter by alert level (INFO, WARNING, CRITICAL)
    
    Returns:
        List of active alerts
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Convert level filter
        alert_level_filter = None
        if level:
            try:
                alert_level_filter = AlertLevel[level.upper()]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
        
        # Get active alerts from anomaly detector
        active_alerts = monitor.anomaly_detector.get_active_alerts(alert_level_filter)
        
        # Format alerts for API response
        formatted_alerts = []
        for alert in active_alerts:
            formatted_alerts.append({
                "alert_id": alert.alert_id,
                "name": alert.name,
                "level": alert.level.name,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "labels": alert.labels,
                "metadata": alert.metadata,
                "resolved": alert.resolved
            })
        
        return JSONResponse(content={
            "timestamp": time.time(),
            "alert_count": len(formatted_alerts),
            "level_filter": level,
            "alerts": formatted_alerts
        })
    
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active alerts")


@performance_router.get("/alerts/history")
async def get_alert_history(
    duration_minutes: int = Query(1440, description="Duration in minutes (default: 24 hours)"),
    level: Optional[str] = Query(None, description="Filter by alert level"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get alert history for specified time period.
    
    Args:
        duration_minutes: How far back to look (default: 24 hours)
        level: Optional filter by alert level
    
    Returns:
        Historical alert data
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Get alert history from anomaly detector
        alert_history = monitor.anomaly_detector.get_alert_history(duration_minutes)
        
        # Filter by level if specified
        if level:
            try:
                alert_level_filter = AlertLevel[level.upper()]
                alert_history = [a for a in alert_history if a.level == alert_level_filter]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
        
        # Format alerts for API response
        formatted_alerts = []
        for alert in alert_history:
            formatted_alerts.append({
                "alert_id": alert.alert_id,
                "name": alert.name,
                "level": alert.level.name,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at
            })
        
        return JSONResponse(content={
            "timestamp": time.time(),
            "duration_minutes": duration_minutes,
            "alert_count": len(formatted_alerts),
            "level_filter": level,
            "alerts": formatted_alerts
        })
    
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert history")


@performance_router.get("/anomalies/active")
async def get_active_anomalies(
    duration_minutes: int = Query(60, description="Duration in minutes (default: 1 hour)"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get currently active anomalies.
    
    Args:
        duration_minutes: How far back to look for anomalies
    
    Returns:
        List of active anomalies
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Get active anomalies from anomaly detector
        active_anomalies = monitor.anomaly_detector.get_active_anomalies(duration_minutes)
        
        # Format anomalies for API response
        formatted_anomalies = []
        for anomaly in active_anomalies:
            formatted_anomalies.append({
                "anomaly_id": anomaly.anomaly_id,
                "rule_id": anomaly.rule_id,
                "metric_name": anomaly.metric_name,
                "anomaly_type": anomaly.anomaly_type.name,
                "algorithm": anomaly.algorithm.name,
                "timestamp": anomaly.timestamp,
                "value": anomaly.value,
                "expected_value": anomaly.expected_value,
                "anomaly_score": anomaly.anomaly_score,
                "severity": anomaly.severity.name,
                "description": anomaly.description,
                "confidence": anomaly.confidence,
                "alert_sent": anomaly.alert_sent,
                "acknowledged": anomaly.acknowledged,
                "resolved": anomaly.resolved
            })
        
        return JSONResponse(content={
            "timestamp": time.time(),
            "duration_minutes": duration_minutes,
            "anomaly_count": len(formatted_anomalies),
            "anomalies": formatted_anomalies
        })
    
    except Exception as e:
        logger.error(f"Error getting active anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active anomalies")


@performance_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="User acknowledging the alert"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Acknowledge an active alert.
    
    Args:
        alert_id: ID of alert to acknowledge
        acknowledged_by: User or system acknowledging the alert
    
    Returns:
        Acknowledgment status
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Find and acknowledge the alert
        success = monitor.anomaly_detector.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": f"Alert {alert_id} acknowledged by {acknowledged_by}",
                "timestamp": time.time()
            })
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@performance_router.post("/anomalies/{anomaly_id}/resolve")
async def resolve_anomaly(
    anomaly_id: str,
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Mark an anomaly as resolved.
    
    Args:
        anomaly_id: ID of anomaly to resolve
    
    Returns:
        Resolution status
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Resolve the anomaly
        success = monitor.anomaly_detector.resolve_anomaly(anomaly_id)
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": f"Anomaly {anomaly_id} marked as resolved",
                "timestamp": time.time()
            })
        else:
            raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")
    
    except Exception as e:
        logger.error(f"Error resolving anomaly {anomaly_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve anomaly")


@performance_router.get("/analytics/summary")
async def get_analytics_summary(
    duration_hours: int = Query(24, description="Duration in hours (default: 24)"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get performance analytics summary.
    
    Args:
        duration_hours: Time period for analysis
    
    Returns:
        Analytics summary with trends and insights
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        duration_seconds = duration_hours * 3600
        current_time = time.time()
        
        # Get historical metrics for analysis
        all_metrics = await monitor.get_current_metrics()
        
        # Analyze key metrics
        analytics_summary = {
            "timestamp": current_time,
            "analysis_duration_hours": duration_hours,
            "system_performance": {},
            "anomaly_statistics": {},
            "resource_utilization": {},
            "trends": {}
        }
        
        # System performance analysis
        cpu_metrics = monitor.metrics_collector.get_historical_metrics(
            metric_name="system_cpu_usage_percent",
            duration_seconds=duration_seconds
        )
        
        if cpu_metrics:
            cpu_values = [m.value for m in cpu_metrics]
            analytics_summary["system_performance"]["cpu"] = {
                "average": sum(cpu_values) / len(cpu_values),
                "peak": max(cpu_values),
                "minimum": min(cpu_values),
                "data_points": len(cpu_values)
            }
        
        memory_metrics = monitor.metrics_collector.get_historical_metrics(
            metric_name="system_memory_usage_percent",
            duration_seconds=duration_seconds
        )
        
        if memory_metrics:
            memory_values = [m.value for m in memory_metrics]
            analytics_summary["system_performance"]["memory"] = {
                "average": sum(memory_values) / len(memory_values),
                "peak": max(memory_values),
                "minimum": min(memory_values),
                "data_points": len(memory_values)
            }
        
        # Anomaly statistics
        analytics_summary["anomaly_statistics"] = monitor.anomaly_detector.get_anomaly_statistics()
        
        # Resource utilization trends
        # Add trend analysis here if needed
        
        return JSONResponse(content=analytics_summary)
    
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")


@performance_router.get("/export/metrics")
async def export_metrics(
    format_type: str = Query("json", description="Export format (json, csv)"),
    duration_hours: int = Query(24, description="Duration in hours"),
    metric_filter: Optional[str] = Query(None, description="Filter metrics by pattern"),
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Export performance metrics data.
    
    Args:
        format_type: Export format (json, csv)  
        duration_hours: Time period to export
        metric_filter: Optional pattern to filter metrics
    
    Returns:
        Exported metrics data
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        if format_type.lower() not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        duration_seconds = duration_hours * 3600
        
        # Get metrics data for export
        export_data = {
            "export_timestamp": time.time(),
            "duration_hours": duration_hours,
            "format": format_type.lower(),
            "metrics": {}
        }
        
        # Get all available metrics
        all_metrics = await monitor.get_current_metrics()
        
        for metric_name in all_metrics.keys():
            # Apply filter if specified
            if metric_filter and metric_filter.lower() not in metric_name.lower():
                continue
            
            # Get historical data for this metric
            historical_data = monitor.metrics_collector.get_historical_metrics(
                metric_name=metric_name,
                duration_seconds=duration_seconds
            )
            
            if historical_data:
                export_data["metrics"][metric_name] = [
                    {
                        "timestamp": m.timestamp,
                        "value": m.value,
                        "metric_type": m.metric_type.name,
                        "labels": m.labels
                    }
                    for m in historical_data
                ]
        
        if format_type.lower() == "json":
            return JSONResponse(content=export_data)
        
        elif format_type.lower() == "csv":
            # Convert to CSV format
            csv_data = "timestamp,metric_name,value,metric_type,labels\n"
            
            for metric_name, data_points in export_data["metrics"].items():
                for point in data_points:
                    labels_str = json.dumps(point["labels"]).replace('"', '""')
                    csv_data += f"{point['timestamp']},{metric_name},{point['value']},{point['metric_type']},\"{labels_str}\"\n"
            
            return StreamingResponse(
                io.StringIO(csv_data),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=metrics_export_{int(time.time())}.csv"}
            )
    
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


@performance_router.get("/config")
async def get_monitoring_config(
    monitor: Optional[PerformanceMonitor] = Depends(get_monitor_dependency)
):
    """
    Get current performance monitoring configuration.
    
    Returns:
        Current monitoring configuration settings
    """
    if not monitor:
        return JSONResponse(content={
            "monitoring_enabled": False,
            "message": "Performance monitoring not available"
        })
    
    try:
        config = monitor.config
        
        return JSONResponse(content={
            "monitoring_enabled": config.enabled,
            "collection_interval_seconds": config.collection_interval_seconds,
            "metrics_retention_seconds": config.metrics_retention_seconds,
            "anomaly_detection_enabled": config.anomaly_detection_enabled,
            "alerting_enabled": config.alerting_enabled,
            "system_metrics_enabled": config.system_metrics_enabled,
            "circuit_breaker_monitoring": config.circuit_breaker_monitoring,
            "connection_pool_monitoring": config.connection_pool_monitoring,
            "thresholds": {
                "cpu_usage_warning": config.cpu_usage_warning_threshold,
                "cpu_usage_critical": config.cpu_usage_critical_threshold,
                "memory_usage_warning": config.memory_usage_warning_threshold,
                "memory_usage_critical": config.memory_usage_critical_threshold,
                "response_time_warning_ms": config.response_time_warning_threshold_ms,
                "response_time_critical_ms": config.response_time_critical_threshold_ms,
                "error_rate_warning": config.error_rate_warning_threshold,
                "error_rate_critical": config.error_rate_critical_threshold
            },
            "historical_data": {
                "enabled": config.enable_historical_data,
                "data_points": config.historical_data_points,
                "aggregation_interval_seconds": config.data_aggregation_interval_seconds
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting monitoring config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring configuration")


# Add the router to be imported by main API
def get_performance_router():
    """Get the performance monitoring API router."""
    return performance_router