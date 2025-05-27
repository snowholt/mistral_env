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
    Uses nvidia-smi for system-wide accurate memory reporting.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries with memory stats for each GPU
    """
    stats = []
    
    if not torch.cuda.is_available():
        return stats
    
    try:
        import subprocess
        
        # Get memory info using nvidia-smi for system-wide accuracy
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    index = int(parts[0])
                    name = parts[1]
                    total_memory_mb = float(parts[2])
                    used_memory_mb = float(parts[3])
                    free_memory_mb = float(parts[4])
                    gpu_utilization = float(parts[5])
                    
                    # Convert to bytes
                    total_memory = total_memory_mb * (1024**2)
                    memory_used = used_memory_mb * (1024**2)
                    memory_free = free_memory_mb * (1024**2)
                    
                    # Calculate percentage
                    memory_used_percent = (used_memory_mb / total_memory_mb) * 100 if total_memory_mb > 0 else 0
                    
                    stats.append({
                        "index": index,
                        "name": name,
                        "total_memory": total_memory,
                        "memory_used": memory_used,
                        "memory_reserved": 0,  # nvidia-smi doesn't distinguish reserved vs used
                        "memory_free": memory_free,
                        "memory_total_mb": total_memory_mb,
                        "memory_used_mb": used_memory_mb,
                        "memory_free_mb": free_memory_mb,
                        "memory_used_percent": memory_used_percent,
                        "gpu_utilization": gpu_utilization
                    })
        else:
            # Fallback to PyTorch CUDA functions (process-local only)
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory
                
                # Memory in bytes (process-local)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_free = total_memory - memory_allocated
                
                # Calculate percentages
                memory_used_percent = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0
                
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
                    "gpu_utilization": 0  # Not available without nvidia-smi
                })
                
    except Exception as e:
        # Emergency fallback - return empty stats with error info
        print(f"Warning: Could not get GPU memory stats: {e}")
        return []
    
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
