"""
Cache management service for model caches and temporary files.

This service extracts cache-related functionality from the model lifecycle service,
providing centralized cache management operations.
"""
import logging
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Cache information for a model."""
    model_id: str
    cache_path: Path
    size_bytes: int
    size_human: str
    exists: bool


class CacheService:
    """Service for cache management operations."""
    
    def __init__(self):
        self.huggingface_cache_dir = self._get_huggingface_cache_dir()
    
    def _get_huggingface_cache_dir(self) -> Path:
        """Get the Hugging Face cache directory."""
        # Check environment variable first
        cache_dir = os.getenv("HF_HOME")
        if cache_dir:
            return Path(cache_dir)
        
        # Check XDG cache directory
        cache_dir = os.getenv("XDG_CACHE_HOME")
        if cache_dir:
            return Path(cache_dir) / "huggingface"
        
        # Default to user home
        return Path.home() / ".cache" / "huggingface"
    
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        # Skip files that can't be accessed
                        continue
        except (OSError, FileNotFoundError):
            # Directory doesn't exist or can't be accessed
            pass
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"
    
    def get_model_cache_info(self, model_id: str) -> CacheInfo:
        """
        Get cache information for a specific model.
        
        Args:
            model_id: Model identifier (e.g., 'microsoft/DialoGPT-medium')
            
        Returns:
            CacheInfo: Cache information for the model
        """
        try:
            # Convert model ID to cache directory name
            # Hugging Face uses format: models--<org>--<model>
            cache_name = model_id.replace("/", "--")
            cache_name = f"models--{cache_name}"
            
            cache_path = self.huggingface_cache_dir / "hub" / cache_name
            
            exists = cache_path.exists()
            size_bytes = self._calculate_directory_size(cache_path) if exists else 0
            size_human = self._format_size(size_bytes)
            
            return CacheInfo(
                model_id=model_id,
                cache_path=cache_path,
                size_bytes=size_bytes,
                size_human=size_human,
                exists=exists
            )
            
        except Exception as e:
            logger.error(f"Failed to get cache info for {model_id}: {e}")
            return CacheInfo(
                model_id=model_id,
                cache_path=Path(),
                size_bytes=0,
                size_human="0 B",
                exists=False
            )
    
    def clear_model_cache(self, model_id: str) -> bool:
        """
        Clear cache for a specific model.
        
        Args:
            model_id: Model identifier to clear cache for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cache_info = self.get_model_cache_info(model_id)
            
            if not cache_info.exists:
                logger.info(f"No cache found for model {model_id}")
                return True
            
            # Remove the cache directory
            shutil.rmtree(cache_info.cache_path)
            
            logger.info(f"Successfully cleared cache for {model_id} "
                       f"(freed {cache_info.size_human})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache for {model_id}: {e}")
            return False
    
    def list_cached_models(self) -> List[CacheInfo]:
        """
        List all cached models.
        
        Returns:
            List[CacheInfo]: List of cache information for all cached models
        """
        cached_models = []
        
        try:
            hub_cache_dir = self.huggingface_cache_dir / "hub"
            
            if not hub_cache_dir.exists():
                logger.info("No Hugging Face cache directory found")
                return cached_models
            
            # Iterate through cache directories
            for cache_dir in hub_cache_dir.iterdir():
                if cache_dir.is_dir() and cache_dir.name.startswith("models--"):
                    # Extract model ID from cache directory name
                    cache_name = cache_dir.name
                    if cache_name.startswith("models--"):
                        model_id = cache_name[8:]  # Remove "models--" prefix
                        model_id = model_id.replace("--", "/")
                        
                        size_bytes = self._calculate_directory_size(cache_dir)
                        size_human = self._format_size(size_bytes)
                        
                        cached_models.append(CacheInfo(
                            model_id=model_id,
                            cache_path=cache_dir,
                            size_bytes=size_bytes,
                            size_human=size_human,
                            exists=True
                        ))
            
            # Sort by size (largest first)
            cached_models.sort(key=lambda x: x.size_bytes, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list cached models: {e}")
        
        return cached_models
    
    def get_total_cache_size(self) -> Dict[str, Any]:
        """
        Get total cache size statistics.
        
        Returns:
            Dict[str, Any]: Total cache size information
        """
        try:
            hub_cache_dir = self.huggingface_cache_dir / "hub"
            
            if not hub_cache_dir.exists():
                return {
                    "total_size_bytes": 0,
                    "total_size_human": "0 B",
                    "model_count": 0,
                    "cache_dir": str(self.huggingface_cache_dir)
                }
            
            total_size = self._calculate_directory_size(hub_cache_dir)
            cached_models = self.list_cached_models()
            
            return {
                "total_size_bytes": total_size,
                "total_size_human": self._format_size(total_size),
                "model_count": len(cached_models),
                "cache_dir": str(self.huggingface_cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get total cache size: {e}")
            return {
                "total_size_bytes": 0,
                "total_size_human": "0 B",
                "model_count": 0,
                "cache_dir": str(self.huggingface_cache_dir),
                "error": str(e)
            }
    
    def clear_all_cache(self) -> bool:
        """
        Clear all model caches.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            hub_cache_dir = self.huggingface_cache_dir / "hub"
            
            if not hub_cache_dir.exists():
                logger.info("No cache directory to clear")
                return True
            
            # Get size before clearing
            cache_stats = self.get_total_cache_size()
            
            # Remove entire hub cache directory
            shutil.rmtree(hub_cache_dir)
            
            logger.info(f"Successfully cleared all model caches "
                       f"(freed {cache_stats['total_size_human']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear all caches: {e}")
            return False
    
    def cleanup_broken_cache_entries(self) -> int:
        """
        Clean up broken or incomplete cache entries.
        
        Returns:
            int: Number of entries cleaned up
        """
        cleaned_count = 0
        
        try:
            hub_cache_dir = self.huggingface_cache_dir / "hub"
            
            if not hub_cache_dir.exists():
                return 0
            
            # Look for incomplete downloads or broken symlinks
            for cache_dir in hub_cache_dir.iterdir():
                if cache_dir.is_dir():
                    try:
                        # Check for incomplete downloads (tmp files)
                        tmp_files = list(cache_dir.glob("*.tmp"))
                        if tmp_files:
                            for tmp_file in tmp_files:
                                tmp_file.unlink()
                                logger.debug(f"Removed temporary file: {tmp_file}")
                        
                        # Check for broken symlinks
                        for item in cache_dir.rglob("*"):
                            if item.is_symlink() and not item.exists():
                                item.unlink()
                                logger.debug(f"Removed broken symlink: {item}")
                                cleaned_count += 1
                        
                        # Check if directory is empty after cleanup
                        if cache_dir.is_dir() and not any(cache_dir.iterdir()):
                            cache_dir.rmdir()
                            logger.debug(f"Removed empty cache directory: {cache_dir}")
                            cleaned_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error cleaning cache entry {cache_dir}: {e}")
                        continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} broken cache entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup broken cache entries: {e}")
        
        return cleaned_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dict[str, Any]: Detailed cache statistics
        """
        try:
            cached_models = self.list_cached_models()
            total_stats = self.get_total_cache_size()
            
            # Calculate statistics
            largest_model = max(cached_models, key=lambda x: x.size_bytes) if cached_models else None
            average_size = total_stats["total_size_bytes"] / len(cached_models) if cached_models else 0
            
            return {
                "cache_directory": str(self.huggingface_cache_dir),
                "total_models": len(cached_models),
                "total_size_bytes": total_stats["total_size_bytes"],
                "total_size_human": total_stats["total_size_human"],
                "average_model_size_bytes": average_size,
                "average_model_size_human": self._format_size(average_size),
                "largest_model": {
                    "model_id": largest_model.model_id if largest_model else None,
                    "size_human": largest_model.size_human if largest_model else None
                },
                "cached_models": [
                    {
                        "model_id": model.model_id,
                        "size_human": model.size_human,
                        "size_bytes": model.size_bytes
                    }
                    for model in cached_models[:10]  # Top 10 largest
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {
                "error": str(e),
                "cache_directory": str(self.huggingface_cache_dir),
                "total_models": 0,
                "total_size_human": "0 B"
            }
