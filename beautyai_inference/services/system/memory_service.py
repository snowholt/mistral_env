"""
Memory management service for GPU and system memory monitoring and operations.

This service extracts memory-related functionality from the model lifecycle service
and utils, providing a centralized interface for memory management operations.
"""
import logging
import torch
import gc
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...utils.memory_utils import (
    get_gpu_memory_stats, 
    get_system_memory_stats, 
    get_gpu_info,
    clear_gpu_memory,
    format_size
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Memory status information."""
    gpu_stats: List[Dict[str, Any]]
    system_stats: Dict[str, float]
    gpu_info: Dict[str, Any]
    has_gpu: bool


class MemoryService:
    """Service for memory management and monitoring operations."""
    
    def get_memory_status(self) -> MemoryStatus:
        """
        Get comprehensive memory status for both GPU and system.
        
        Returns:
            MemoryStatus: Current memory status information
        """
        try:
            gpu_info = get_gpu_info()
            has_gpu = gpu_info.get("is_available", False)
            
            gpu_stats = get_gpu_memory_stats() if has_gpu else []
            system_stats = get_system_memory_stats()
            
            return MemoryStatus(
                gpu_stats=gpu_stats,
                system_stats=system_stats,
                gpu_info=gpu_info,
                has_gpu=has_gpu
            )
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            return MemoryStatus(
                gpu_stats=[],
                system_stats={},
                gpu_info={"is_available": False},
                has_gpu=False
            )
    
    def clear_gpu_memory(self) -> bool:
        """
        Clear GPU memory cache.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if torch.cuda.is_available():
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                clear_gpu_memory()
                
                logger.info("GPU memory cache cleared successfully")
                return True
            else:
                logger.warning("GPU not available for memory clearing")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear GPU memory: {e}")
            return False
    
    def check_available_memory(self, device_index: int = 0) -> int:
        """
        Check available GPU memory on specific device.
        
        Args:
            device_index: GPU device index to check
            
        Returns:
            int: Available memory in bytes
        """
        try:
            if not torch.cuda.is_available():
                # Return a conservative estimate if no GPU available
                return 8 * 1024 * 1024 * 1024  # 8GB fallback
                
            device = torch.device(f"cuda:{device_index}")
            stats = torch.cuda.get_device_properties(device)
            total_memory = stats.total_memory
            
            # Get current usage
            allocated_memory = torch.cuda.memory_allocated(device)
            
            # Calculate available memory
            available = total_memory - allocated_memory
            
            logger.debug(f"GPU {device_index} available memory: {format_size(available)}")
            return available
            
        except (ImportError, RuntimeError, IndexError) as e:
            logger.warning(f"Could not check GPU memory for device {device_index}: {e}")
            # Return conservative estimate
            return 8 * 1024 * 1024 * 1024
    
    def get_memory_usage_percentage(self, device_index: int = 0) -> float:
        """
        Get memory usage percentage for specific GPU device.
        
        Args:
            device_index: GPU device index to check
            
        Returns:
            float: Memory usage percentage (0-100)
        """
        try:
            if not torch.cuda.is_available():
                return 0.0
                
            device = torch.device(f"cuda:{device_index}")
            stats = torch.cuda.get_device_properties(device)
            total_memory = stats.total_memory
            
            allocated_memory = torch.cuda.memory_allocated(device)
            
            percentage = (allocated_memory / total_memory) * 100 if total_memory > 0 else 0.0
            
            return percentage
            
        except (ImportError, RuntimeError, IndexError) as e:
            logger.warning(f"Could not get memory usage for device {device_index}: {e}")
            return 0.0
    
    def format_memory_display(self, memory_status: MemoryStatus) -> str:
        """
        Format memory status for display.
        
        Args:
            memory_status: Memory status information
            
        Returns:
            str: Formatted memory display
        """
        lines = []
        
        # System memory
        if memory_status.system_stats:
            lines.append("System Memory:")
            stats = memory_status.system_stats
            lines.append(f"  Total: {stats.get('total_gb', 0):.2f} GB")
            lines.append(f"  Used:  {stats.get('used_gb', 0):.2f} GB ({stats.get('percent', 0):.1f}%)")
            lines.append(f"  Free:  {stats.get('available_gb', 0):.2f} GB")
        
        # GPU memory
        if memory_status.has_gpu and memory_status.gpu_stats:
            lines.append("\nGPU Memory:")
            for gpu_stat in memory_status.gpu_stats:
                lines.append(f"  GPU {gpu_stat.get('index', 0)}: {gpu_stat.get('name', 'Unknown')}")
                lines.append(f"    Used:  {gpu_stat.get('memory_used_mb', 0):.2f} MB / "
                           f"{gpu_stat.get('memory_total_mb', 0):.2f} MB "
                           f"({gpu_stat.get('memory_used_percent', 0):.1f}%)")
                lines.append(f"    Free:  {gpu_stat.get('memory_free_mb', 0):.2f} MB")
                
                # Add visual progress bar
                used_percent = gpu_stat.get('memory_used_percent', 0)
                bar_length = 20
                filled_length = int(bar_length * used_percent / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                lines.append(f"    [{bar}] {used_percent:.1f}%")
        elif memory_status.has_gpu:
            lines.append("\nGPU Memory: Available but no stats")
        else:
            lines.append("\nGPU Memory: Not available")
        
        return "\n".join(lines)
    
    def estimate_model_memory_requirements(self, model_size_gb: float, quantization: Optional[str] = None) -> float:
        """
        Estimate memory requirements for a model.
        
        Args:
            model_size_gb: Model size in GB
            quantization: Quantization type (4bit, 8bit, None)
            
        Returns:
            float: Estimated memory requirement in GB
        """
        # Base model size
        estimated_gb = model_size_gb
        
        # Quantization adjustments
        if quantization == "4bit":
            estimated_gb *= 0.25  # 4-bit quantization
        elif quantization == "8bit":
            estimated_gb *= 0.5   # 8-bit quantization
        
        # Add overhead for inference (typically 20-30%)
        overhead_factor = 1.25
        estimated_gb *= overhead_factor
        
        logger.debug(f"Estimated memory requirement: {estimated_gb:.2f} GB "
                    f"(model: {model_size_gb} GB, quant: {quantization})")
        
        return estimated_gb
    
    def check_memory_sufficient(self, required_gb: float, device_index: int = 0) -> bool:
        """
        Check if sufficient memory is available for a model.
        
        Args:
            required_gb: Required memory in GB
            device_index: GPU device index to check
            
        Returns:
            bool: True if sufficient memory available
        """
        try:
            available_bytes = self.check_available_memory(device_index)
            available_gb = available_bytes / (1024**3)
            
            # Add safety margin (500MB)
            safety_margin_gb = 0.5
            
            sufficient = available_gb >= (required_gb + safety_margin_gb)
            
            logger.debug(f"Memory check - Required: {required_gb:.2f} GB, "
                        f"Available: {available_gb:.2f} GB, "
                        f"Sufficient: {sufficient}")
            
            return sufficient
            
        except Exception as e:
            logger.error(f"Failed to check memory sufficiency: {e}")
            return False
