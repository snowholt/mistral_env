"""
System status service for comprehensive system monitoring and reporting.

This service aggregates information from various components to provide
comprehensive system status information.
"""
import logging
import platform
import psutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .memory_service import MemoryService, MemoryStatus
from .cache_service import CacheService
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_info

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Comprehensive system status information."""
    timestamp: datetime
    memory_status: MemoryStatus
    loaded_models: Dict[str, Any]
    cache_stats: Dict[str, Any]
    system_info: Dict[str, Any]
    uptime_seconds: float


class StatusService:
    """Service for system status monitoring and reporting."""
    
    def __init__(self):
        self.memory_service = MemoryService()
        self.cache_service = CacheService()
        self.model_manager = ModelManager()
        self.start_time = time.time()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get basic system information.
        
        Returns:
            Dict[str, Any]: System information
        """
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "gpu_info": get_gpu_info()
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    def get_loaded_models_info(self) -> Dict[str, Any]:
        """
        Get information about currently loaded models.
        
        Returns:
            Dict[str, Any]: Loaded models information
        """
        try:
            loaded_model_names = self.model_manager.list_loaded_models()
            
            model_info = {}
            for model_name in loaded_model_names:
                model_instance = self.model_manager.get_loaded_model(model_name)
                if model_instance is None:
                    continue
                try:
                    # Get model details
                    model_info[model_name] = {
                        "engine_type": getattr(model_instance, 'engine_type', 'unknown'),
                        "model_id": getattr(model_instance, 'model_id', 'unknown'),
                        "quantization": getattr(model_instance, 'quantization', None),
                        "device": str(getattr(model_instance, 'device', 'unknown')),
                        "loaded_at": getattr(model_instance, 'loaded_at', None),
                        "memory_usage": self._estimate_model_memory_usage(model_instance)
                    }
                except Exception as e:
                    logger.warning(f"Error getting info for model {model_name}: {e}")
                    model_info[model_name] = {"error": str(e)}
            
            return {
                "count": len(loaded_model_names),
                "models": model_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get loaded models info: {e}")
            return {"count": 0, "models": {}, "error": str(e)}
    
    def _estimate_model_memory_usage(self, model_instance) -> Dict[str, Any]:
        """
        Estimate memory usage for a loaded model.
        
        Args:
            model_instance: The loaded model instance
            
        Returns:
            Dict[str, Any]: Memory usage estimation
        """
        try:
            import torch
            
            if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'parameters'):
                # Count parameters
                total_params = sum(p.numel() for p in model_instance.model.parameters())
                trainable_params = sum(p.numel() for p in model_instance.model.parameters() if p.requires_grad)
                
                # Estimate memory (roughly 4 bytes per parameter for fp32, 2 for fp16, etc.)
                bytes_per_param = 4  # Assume fp32 by default
                if hasattr(model_instance.model, 'dtype'):
                    if model_instance.model.dtype == torch.float16:
                        bytes_per_param = 2
                    elif model_instance.model.dtype == torch.int8:
                        bytes_per_param = 1
                
                estimated_memory_bytes = total_params * bytes_per_param
                estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
                
                return {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "estimated_memory_mb": estimated_memory_mb,
                    "estimated_memory_human": self.cache_service._format_size(estimated_memory_bytes)
                }
                
        except Exception as e:
            logger.debug(f"Could not estimate memory usage: {e}")
        
        return {"estimated": False, "error": "Could not estimate"}
    
    def get_comprehensive_status(self) -> SystemStatus:
        """
        Get comprehensive system status.
        
        Returns:
            SystemStatus: Complete system status information
        """
        try:
            # Get all status components
            memory_status = self.memory_service.get_memory_status()
            loaded_models = self.get_system_wide_model_info()  # Use system-wide detection
            cache_stats = self.cache_service.get_cache_statistics()
            system_info = self.get_system_info()
            uptime = time.time() - self.start_time
            
            return SystemStatus(
                timestamp=datetime.now(),
                memory_status=memory_status,
                loaded_models=loaded_models,
                cache_stats=cache_stats,
                system_info=system_info,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            # Return minimal status
            return SystemStatus(
                timestamp=datetime.now(),
                memory_status=MemoryStatus([], {}, {"is_available": False}, False),
                loaded_models={"count": 0, "local_models": {}, "external_processes": [], "local_count": 0, "external_count": 0},
                cache_stats={"error": str(e)},
                system_info={"error": str(e)},
                uptime_seconds=time.time() - self.start_time
            )
    
    def format_status_display(self, status: SystemStatus) -> str:
        """
        Format system status for console display.
        
        Args:
            status: System status information
            
        Returns:
            str: Formatted status display
        """
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("BEAUTYAI INFERENCE FRAMEWORK - SYSTEM STATUS")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Uptime: {self._format_uptime(status.uptime_seconds)}")
        lines.append("")
        
        # System Information
        lines.append("SYSTEM INFORMATION")
        lines.append("-" * 30)
        if "error" not in status.system_info:
            lines.append(f"Platform: {status.system_info.get('platform', 'Unknown')}")
            lines.append(f"Python: {status.system_info.get('python_version', 'Unknown')}")
            gpu_info = status.system_info.get('gpu_info', {})
            if gpu_info.get('is_available'):
                lines.append(f"GPU: {gpu_info.get('device_name', 'Unknown')} ({gpu_info.get('device_count', 0)} devices)")
            else:
                lines.append("GPU: Not available")
        else:
            lines.append(f"Error: {status.system_info['error']}")
        lines.append("")
        
        # Memory Status
        lines.append("MEMORY STATUS")
        lines.append("-" * 30)
        memory_display = self.memory_service.format_memory_display(status.memory_status)
        lines.append(memory_display)
        lines.append("")
        
        # Loaded Models
        lines.append("LOADED MODELS")
        lines.append("-" * 30)
        
        # Check for the new system-wide format
        if "local_models" in status.loaded_models:
            # New system-wide format
            total_count = status.loaded_models.get("count", 0)
            local_count = status.loaded_models.get("local_count", 0)
            external_count = status.loaded_models.get("external_count", 0)
            
            if total_count > 0:
                lines.append(f"Total Models Loaded: {total_count} ({local_count} local + {external_count} external)")
                lines.append("")
                
                # Show local models
                local_models = status.loaded_models.get("local_models", {})
                if local_models:
                    lines.append("LOCAL MODELS (in this process):")
                    for model_name, model_info in local_models.items():
                        if "error" not in model_info:
                            lines.append(f"• {model_name}")
                            lines.append(f"  Engine: {model_info.get('engine_type', 'Unknown')}")
                            lines.append(f"  Model ID: {model_info.get('model_id', 'Unknown')}")
                            if model_info.get('quantization'):
                                lines.append(f"  Quantization: {model_info['quantization']}")
                            
                            # Memory usage if available
                            mem_usage = model_info.get('memory_usage', {})
                            if mem_usage.get('estimated_memory_human'):
                                lines.append(f"  Memory: ~{mem_usage['estimated_memory_human']}")
                        else:
                            lines.append(f"• {model_name} (Error: {model_info['error']})")
                        lines.append("")
                
                # Show external processes
                external_processes = status.loaded_models.get("external_processes", [])
                if external_processes:
                    lines.append("EXTERNAL PROCESSES (models loaded in other processes):")
                    for proc_info in external_processes:
                        lines.append(f"• PID {proc_info['pid']} ({proc_info['process_type']})")
                        lines.append(f"  Model: {proc_info['model_hint']}")
                        lines.append(f"  GPU Memory: {proc_info['memory_usage_human']}")
                        lines.append(f"  Process: {proc_info['process_name']}")
                        lines.append("")
            else:
                lines.append("No models currently loaded")
                lines.append("")
        else:
            # Legacy format fallback
            if status.loaded_models["count"] > 0:
                for model_name, model_info in status.loaded_models["models"].items():
                    if "error" not in model_info:
                        lines.append(f"• {model_name}")
                        lines.append(f"  Engine: {model_info.get('engine_type', 'Unknown')}")
                        lines.append(f"  Model ID: {model_info.get('model_id', 'Unknown')}")
                        if model_info.get('quantization'):
                            lines.append(f"  Quantization: {model_info['quantization']}")
                        
                        # Memory usage if available
                        mem_usage = model_info.get('memory_usage', {})
                        if mem_usage.get('estimated_memory_human'):
                            lines.append(f"  Memory: ~{mem_usage['estimated_memory_human']}")
                    else:
                        lines.append(f"• {model_name} (Error: {model_info['error']})")
                    lines.append("")
            else:
                lines.append("No models currently loaded")
                lines.append("")
        
        # Cache Statistics
        lines.append("CACHE STATISTICS")
        lines.append("-" * 30)
        if "error" not in status.cache_stats:
            lines.append(f"Cache Directory: {status.cache_stats.get('cache_directory', 'Unknown')}")
            lines.append(f"Cached Models: {status.cache_stats.get('total_models', 0)}")
            lines.append(f"Total Cache Size: {status.cache_stats.get('total_size_human', '0 B')}")
            lines.append(f"Average Model Size: {status.cache_stats.get('average_model_size_human', '0 B')}")
            
            largest = status.cache_stats.get('largest_model', {})
            if largest.get('model_id'):
                lines.append(f"Largest Model: {largest['model_id']} ({largest['size_human']})")
        else:
            lines.append(f"Error: {status.cache_stats['error']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human readable format."""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get basic health status for monitoring systems.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            status = self.get_comprehensive_status()
            
            # Determine health status
            health_checks = {
                "memory_available": True,
                "gpu_accessible": status.memory_status.has_gpu,
                "cache_accessible": "error" not in status.cache_stats,
                "models_loaded": status.loaded_models["count"] > 0,
                "system_responsive": True  # If we got here, system is responsive
            }
            
            # Check memory usage
            if status.memory_status.gpu_stats:
                for gpu_stat in status.memory_status.gpu_stats:
                    if gpu_stat.get('memory_used_percent', 0) > 95:  # 95% threshold
                        health_checks["memory_available"] = False
                        health_checks["memory_available"] = False
                        break
            
            overall_health = "healthy" if all(health_checks.values()) else "warning"
            
            return {
                "status": overall_health,
                "timestamp": status.timestamp.isoformat(),
                "uptime_seconds": status.uptime_seconds,
                "checks": health_checks,
                "loaded_models_count": status.loaded_models["count"],
                "cache_models_count": status.cache_stats.get("total_models", 0),
                "memory_status": {
                    "gpu_available": status.memory_status.has_gpu,
                    "gpu_count": len(status.memory_status.gpu_stats),
                    "system_memory_used_percent": status.memory_status.system_stats.get("percent", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_system_wide_model_info(self) -> Dict[str, Any]:
        """
        Get information about models loaded system-wide (across all processes).
        This includes models loaded in other BeautyAI processes.
        
        Returns:
            Dict[str, Any]: System-wide loaded models information
        """
        try:
            import subprocess
            import psutil
            
            # First get local loaded models
            local_models = self.get_loaded_models_info()
            
            # Check for BeautyAI processes using GPU memory
            external_processes = []
            try:
                # Get processes using GPU memory
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                pid = int(parts[0])
                                process_name = parts[1]
                                used_memory_mb = float(parts[2])
                            if len(parts) >= 3:
                                pid = int(parts[0])
                                process_name = parts[1]
                                used_memory_mb = float(parts[2])
                                
                                # Check if this is a BeautyAI-related process
                                try:
                                    proc = psutil.Process(pid)
                                    cmdline = ' '.join(proc.cmdline())
                                    
                                    # Look for BeautyAI-related processes
                                    if any(keyword in cmdline.lower() for keyword in 
                                          ['beautyai', 'uvicorn', 'inference', 'model']):
                                        
                                        # Try to determine what model might be loaded
                                        model_hint = "Unknown Model"
                                        if 'qwen' in cmdline.lower():
                                            model_hint = "Qwen-based Model"
                                        elif 'mistral' in cmdline.lower():
                                            model_hint = "Mistral-based Model"
                                        elif 'llama' in cmdline.lower():
                                            model_hint = "Llama-based Model"
                                        
                                        external_processes.append({
                                            "pid": pid,
                                            "process_name": process_name,
                                            "memory_usage_mb": used_memory_mb,
                                            "memory_usage_human": self.cache_service._format_size(used_memory_mb * 1024 * 1024),
                                            "cmdline_hint": cmdline[:100] + "..." if len(cmdline) > 100 else cmdline,
                                            "model_hint": model_hint,
                                            "process_type": "API Server" if "uvicorn" in cmdline else "External Process"
                                        })
                                        
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                                    
            except Exception as e:
                logger.debug(f"Could not check external processes: {e}")
            
            # Combine local and external information
            total_models = local_models["count"] + len(external_processes)
            
            return {
                "count": total_models,
                "local_models": local_models["models"],
                "external_processes": external_processes,
                "local_count": local_models["count"],
                "external_count": len(external_processes)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system-wide model info: {e}")
            return {
                "count": 0,
                "local_models": {},
                "external_processes": [],
                "local_count": 0,
                "external_count": 0,
                "error": str(e)
            }
