"""
Utility functions for memory tracking and other common operations.
"""
import torch
import psutil
import platform
import time
from typing import Dict, Any, Optional, List


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if torch.cuda.is_available():
        return {
            "is_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "total_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "arch": platform.machine(),
        }
    else:
        return {"is_available": False}


def get_gpu_memory_stats() -> List[Dict[str, Any]]:
    """
    Get detailed GPU memory usage statistics for all available GPUs.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries with memory stats for each GPU
    """
    stats = []
    
    if not torch.cuda.is_available():
        return stats
        
    # Get stats for each GPU
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        total_memory = device_props.total_memory
        
        # Memory in bytes
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        memory_free = total_memory - memory_allocated
        
        # Calculate percentages
        memory_used_percent = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0
        
        # Get GPU utilization (only on NVIDIA with nvidia-smi)
        gpu_utilization = 0
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', str(i)],
                capture_output=True,
                text=True
            )
            gpu_utilization = float(result.stdout.strip())
        except:
            pass
            
        # Add to stats list
        stats.append({
            "index": i,
            "name": device_props.name,
            "total_memory": total_memory,
            "memory_used": memory_allocated,
            "memory_reserved": memory_reserved,
            "memory_free": memory_free,
            "memory_total_mb": total_memory / (1024**2),
            "memory_used_mb": memory_allocated / (1024**2),
            "memory_free_mb": memory_free / (1024**2),
            "memory_used_percent": memory_used_percent,
            "gpu_utilization": gpu_utilization
        })
    
    return stats


def get_system_memory_stats() -> Dict[str, float]:
    """Get system memory usage statistics."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / 1024**3,
        "available_gb": memory.available / 1024**3,
        "used_gb": memory.used / 1024**3,
        "percent": memory.percent,
    }


def time_function(func, *args, **kwargs) -> Dict[str, Any]:
    """Time the execution of a function."""
    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get memory stats after function execution
    memory_stats = get_gpu_memory_stats() if torch.cuda.is_available() else {}
    
    return {
        "result": result,
        "execution_time": execution_time,
        "memory_stats": memory_stats,
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_size(size_bytes: float) -> str:
    """Format a size in bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"


def clear_terminal_screen() -> None:
    """Clear the terminal screen."""
    print("\033c", end="")
