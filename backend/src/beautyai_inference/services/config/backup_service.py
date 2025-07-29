"""
Configuration backup service for BeautyAI.

This service provides comprehensive configuration backup and restore capabilities:
- Creating timestamped backups of configuration files
- Maintaining backup history and metadata
- Creating compressed backup archives
- Restoring configuration from backup files
- Managing backup cleanup and retention
"""
import logging
import os
import shutil
import datetime
import json
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig

logger = logging.getLogger(__name__)


class BackupService(BaseService):
    """Service for backing up and restoring configuration files.
    
    Handles creation, management, and restoration of configuration backups,
    including both individual files and compressed archives.
    """
    
    def __init__(self):
        super().__init__()
        self.backup_history: List[Dict[str, str]] = []
    
    def backup_config(self, app_config: AppConfig, backup_dir: str = "backups", 
                     label: str = "", compress: bool = False, 
                     keep_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Back up configuration files to a specified directory.
        
        This method creates timestamped backups of configuration files,
        including the main configuration and model registry files.
        
        Args:
            app_config: The application configuration to backup
            backup_dir: Directory to store backups
            label: Optional label for the backup
            compress: Whether to create compressed archive
            keep_count: Number of backups to keep (cleanup old ones)
            
        Returns:
            Dict with backup results containing success flag and backup paths
        """
        try:
            # Create backup directory if it doesn't exist
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True, parents=True)
            
            # Generate backup timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get backup label suffix
            label_suffix = f"_{label}" if label else ""
            
            # Backup main configuration
            config_file = getattr(app_config, 'config_file', None)
            config_backup = None
            
            if config_file and os.path.exists(config_file):
                config_backup = backup_path / f"config{label_suffix}_{timestamp}.json"
                shutil.copy2(config_file, config_backup)
            
            # Backup models configuration
            models_file = app_config.models_file
            models_backup = None
            
            if models_file and os.path.exists(models_file):
                models_backup = backup_path / f"models{label_suffix}_{timestamp}.json"
                shutil.copy2(models_file, models_backup)
            
            # Add to backup history
            backup_record = {
                "timestamp": timestamp,
                "label": label or "auto-backup",
                "config_file": str(config_backup) if config_backup else None,
                "models_file": str(models_backup) if models_backup else None
            }
            self.backup_history.append(backup_record)
            
            # Create compressed archive if requested
            archive_path = None
            if compress:
                archive_path = backup_path / f"beautyai_config_backup{label_suffix}_{timestamp}.tar.gz"
                
                with tarfile.open(archive_path, "w:gz") as tar:
                    if config_backup and config_backup.exists():
                        tar.add(config_backup, arcname=os.path.basename(config_backup))
                    if models_backup and models_backup.exists():
                        tar.add(models_backup, arcname=os.path.basename(models_backup))
            
            # Clean up old backups if requested
            if keep_count is not None and keep_count > 0:
                self._cleanup_old_backups(backup_path, keep_count)
            
            return {
                "success": True,
                "timestamp": timestamp,
                "config_backup": str(config_backup) if config_backup else None,
                "models_backup": str(models_backup) if models_backup else None,
                "archive_path": str(archive_path) if archive_path else None,
                "message": "Backup completed successfully"
            }
            
        except Exception as e:
            logger.exception("Failed to back up configuration")
            return {
                "success": False,
                "timestamp": None,
                "config_backup": None,
                "models_backup": None,
                "archive_path": None,
                "message": f"Backup failed: {str(e)}"
            }
    
    def restore_config(self, config_backup_file: str, models_backup_file: Optional[str] = None,
                      target_config: Optional[str] = None, target_models: Optional[str] = None,
                      validate: bool = True) -> Dict[str, Any]:
        """
        Restore configuration from backup files.
        
        This method restores configuration from previously created backups.
        It automatically creates a safety backup of current configuration
        before proceeding.
        
        Args:
            config_backup_file: Path to config backup file to restore
            models_backup_file: Optional path to models backup file to restore
            target_config: Target path for config file (defaults to current config location)
            target_models: Target path for models file (defaults to current models location)
            validate: Whether to validate restored configuration
            
        Returns:
            Dict with restore results containing success flag and any errors
        """
        try:
            if not os.path.exists(config_backup_file):
                return {
                    "success": False,
                    "message": f"Config backup file {config_backup_file} not found"
                }
            
            # Verify backup file is valid JSON
            try:
                with open(config_backup_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "message": f"Backup file {config_backup_file} is not valid JSON"
                }
            
            # Determine target paths
            if not target_config:
                target_config = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            
            # Backup current configuration before restoring
            safety_backups = []
            backup_suffix = f".pre_restore.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if os.path.exists(target_config):
                safety_backup = f"{target_config}{backup_suffix}"
                shutil.copy2(target_config, safety_backup)
                safety_backups.append(safety_backup)
            
            # Restore main configuration
            shutil.copy2(config_backup_file, target_config)
            
            # Restore models if provided
            if models_backup_file:
                if not os.path.exists(models_backup_file):
                    return {
                        "success": False,
                        "message": f"Models backup file {models_backup_file} not found"
                    }
                
                # Verify models file is valid JSON
                try:
                    with open(models_backup_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    # Rollback main config restore
                    if safety_backups:
                        shutil.copy2(safety_backups[0], target_config)
                    return {
                        "success": False,
                        "message": f"Models backup file {models_backup_file} is not valid JSON"
                    }
                
                # Load config to determine models file path
                if not target_models:
                    temp_config = AppConfig(config_file=target_config)
                    target_models = temp_config.models_file
                
                # Create directory structure if needed
                os.makedirs(os.path.dirname(target_models), exist_ok=True)
                
                # Backup current models file
                if os.path.exists(target_models):
                    models_safety_backup = f"{target_models}{backup_suffix}"
                    shutil.copy2(target_models, models_safety_backup)
                    safety_backups.append(models_safety_backup)
                
                # Restore models file
                shutil.copy2(models_backup_file, target_models)
            
            # Validate the restored configuration if requested
            validation_result = None
            if validate:
                try:
                    restored_config = AppConfig.from_file(config_file=target_config)
                    validation_result = {"valid": True, "errors": []}
                except Exception as e:
                    validation_result = {"valid": False, "errors": [str(e)]}
            
            return {
                "success": True,
                "target_config": target_config,
                "target_models": target_models if models_backup_file else None,
                "safety_backups": safety_backups,
                "validation_result": validation_result,
                "message": "Configuration restored successfully"
            }
            
        except Exception as e:
            logger.exception("Failed to restore configuration")
            return {
                "success": False,
                "message": f"Restore failed: {str(e)}"
            }
    
    def list_backups(self, backup_dir: str = "backups") -> List[Dict[str, Any]]:
        """
        List available backup files.
        
        Args:
            backup_dir: Directory containing backups
            
        Returns:
            List of backup information dictionaries
        """
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return []
        
        backups = []
        
        # Find all backup files
        config_backups = list(backup_path.glob("config_*.json"))
        models_backups = list(backup_path.glob("models_*.json"))
        archives = list(backup_path.glob("beautyai_config_backup_*.tar.gz"))
        
        # Group by timestamp
        backup_groups = {}
        
        for config_file in config_backups:
            # Extract timestamp from filename
            parts = config_file.stem.split('_')
            if len(parts) >= 2:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                if timestamp not in backup_groups:
                    backup_groups[timestamp] = {
                        "timestamp": timestamp,
                        "config_file": None,
                        "models_file": None,
                        "archive_file": None,
                        "created": datetime.datetime.fromtimestamp(os.path.getmtime(config_file))
                    }
                backup_groups[timestamp]["config_file"] = str(config_file)
        
        for models_file in models_backups:
            parts = models_file.stem.split('_')
            if len(parts) >= 2:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                if timestamp in backup_groups:
                    backup_groups[timestamp]["models_file"] = str(models_file)
        
        for archive_file in archives:
            parts = archive_file.stem.split('_')
            if len(parts) >= 2:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                if timestamp in backup_groups:
                    backup_groups[timestamp]["archive_file"] = str(archive_file)
        
        # Convert to list and sort by timestamp
        backups = list(backup_groups.values())
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return backups
    
    def _cleanup_old_backups(self, backup_path: Path, keep_count: int) -> None:
        """
        Clean up old backups, keeping only the most recent ones.
        
        Args:
            backup_path: Path to backup directory
            keep_count: Number of most recent backups to keep
        """
        # Find all config backup files
        config_backups = list(backup_path.glob("config_*.json"))
        models_backups = list(backup_path.glob("models_*.json"))
        archives = list(backup_path.glob("beautyai_config_backup_*.tar.gz"))
        
        # Sort by modification time (oldest first)
        config_backups.sort(key=lambda x: os.path.getmtime(x))
        models_backups.sort(key=lambda x: os.path.getmtime(x))
        archives.sort(key=lambda x: os.path.getmtime(x))
        
        # Delete oldest backups beyond keep_count
        if len(config_backups) > keep_count:
            for old_backup in config_backups[:-keep_count]:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
                
        if len(models_backups) > keep_count:
            for old_backup in models_backups[:-keep_count]:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
                
        if len(archives) > keep_count:
            for old_archive in archives[:-keep_count]:
                os.remove(old_archive)
                logger.info(f"Removed old archive: {old_archive}")
    
    def get_backup_history(self) -> List[Dict[str, str]]:
        """Get the history of backups created during this session."""
        return self.backup_history.copy()
