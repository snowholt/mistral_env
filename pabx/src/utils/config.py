"""
Configuration management for PABX system
Loads YAML and JSON configuration files
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager with singleton pattern"""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.reload()
    
    def reload(self):
        """Reload configuration from files"""
        base_dir = Path(__file__).parent.parent.parent
        config_dir = base_dir / "config"
        
        # Load main settings
        settings_file = config_dir / "settings.yaml"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {settings_file}")
        
        # Load device configuration
        devices_file = config_dir / "devices.json"
        if devices_file.exists():
            with open(devices_file, 'r') as f:
                self._config['devices'] = json.load(f)
        
        # Expand paths
        self._expand_paths()
    
    def _expand_paths(self):
        """Expand relative paths to absolute paths"""
        if 'audio' in self._config:
            if 'test_files_dir' in self._config['audio']:
                self._config['audio']['test_files_dir'] = os.path.expanduser(
                    self._config['audio']['test_files_dir']
                )
            if 'recordings_dir' in self._config['audio']:
                self._config['audio']['recordings_dir'] = os.path.expanduser(
                    self._config['audio']['recordings_dir']
                )
        
        if 'capture' in self._config:
            if 'pcap_dir' in self._config['capture']:
                self._config['capture']['pcap_dir'] = os.path.expanduser(
                    self._config['capture']['pcap_dir']
                )
        
        if 'logging' in self._config:
            if 'json' in self._config['logging']:
                self._config['logging']['json']['file'] = os.path.expanduser(
                    self._config['logging']['json']['file']
                )
            if 'session' in self._config['logging']:
                self._config['logging']['session']['dir'] = os.path.expanduser(
                    self._config['logging']['session']['dir']
                )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy()
    
    def set(self, key: str, value: Any):
        """Set configuration value (runtime only, not persisted)"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config
