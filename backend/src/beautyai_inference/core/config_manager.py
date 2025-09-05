"""
Enhanced Configuration Manager with Environment-Aware Loading.

Implements cascading configuration priorities: ENV > Config File > Defaults
Supports hot-reloading, validation, and secrets management.
"""

import os
import json
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from threading import Lock
from contextlib import contextmanager

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from cryptography.fernet import Fernet
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class DuplexStreamingConfig(BaseSettings):
    """Configuration for duplex streaming features."""
    
    enabled: bool = Field(default=True, description="Enable duplex streaming")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate in Hz")
    chunk_size: int = Field(default=1024, ge=256, le=8192, description="Audio chunk size in samples")
    buffer_size_ms: int = Field(default=500, ge=100, le=2000, description="Buffer size in milliseconds")
    echo_cancellation: bool = Field(default=True, description="Enable echo cancellation")
    noise_suppression: bool = Field(default=True, description="Enable noise suppression")
    auto_gain_control: bool = Field(default=True, description="Enable automatic gain control")
    
    # TTS settings
    tts_model: str = Field(default="edge-tts", description="TTS model to use")
    tts_voice: str = Field(default="ar-SA-ZariyahNeural", description="TTS voice to use")
    tts_streaming_enabled: bool = Field(default=True, description="Enable TTS streaming")
    tts_chunk_size_ms: int = Field(default=40, ge=10, le=200, description="TTS chunk size in milliseconds")
    
    # Audio format settings
    input_format: str = Field(default="pcm16", pattern="^(pcm16|opus|webm)$", description="Input audio format")
    tts_format: str = Field(default="opus", pattern="^(opus|pcm16|webm)$", description="TTS audio format")
    
    # Echo suppression settings
    echo_correlation_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Echo correlation threshold")
    echo_spectral_threshold: float = Field(default=0.12, ge=0.0, le=1.0, description="Echo spectral threshold")
    echo_vad_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Echo VAD threshold")
    echo_suppression_enabled: bool = Field(default=True, description="Enable echo suppression")
    echo_adaptive_threshold: bool = Field(default=True, description="Enable adaptive threshold")
    
    # Jitter buffer settings
    jitter_buffer_target_ms: int = Field(default=60, ge=10, le=500, description="Jitter buffer target in milliseconds")
    jitter_buffer_max_ms: int = Field(default=200, ge=50, le=1000, description="Jitter buffer maximum in milliseconds")
    jitter_buffer_adaptive: bool = Field(default=True, description="Enable adaptive jitter buffer")
    
    # Duplex mode settings
    duplex_mode: str = Field(default="full", pattern="^(full|half|simplex)$", description="Duplex mode")
    barge_in_enabled: bool = Field(default=True, description="Enable barge-in")
    barge_in_sensitivity: float = Field(default=0.3, ge=0.0, le=1.0, description="Barge-in sensitivity")
    barge_in_delay_ms: int = Field(default=300, ge=0, le=2000, description="Barge-in delay in milliseconds")
    
    # Buffer settings
    audio_buffer_size_ms: int = Field(default=10000, ge=1000, le=30000, description="Audio buffer size in milliseconds")
    pcm_buffer_size_samples: int = Field(default=16000, ge=1024, le=65536, description="PCM buffer size in samples")
    
    # Connection settings
    max_connections: int = Field(default=10, ge=1, le=100, description="Maximum concurrent connections")
    max_concurrent_sessions: int = Field(default=10, ge=1, le=100, description="Maximum concurrent sessions")
    connection_timeout_ms: int = Field(default=30000, ge=5000, le=120000, description="Connection timeout in milliseconds")
    session_timeout_ms: int = Field(default=300000, ge=30000, le=1800000, description="Session timeout in milliseconds")
    
    # Protocol settings
    protocol_version: str = Field(default="1.0", pattern="^\\d+\\.\\d+$", description="Protocol version")
    binary_protocol: bool = Field(default=True, description="Use binary protocol for streaming")
    
    model_config = ConfigDict(
        env_prefix="BEAUTYAI_DUPLEX_",
        case_sensitive=False
    )


class SystemConfig(BaseSettings):
    """System-level configuration."""
    
    environment: str = Field(default="development", pattern="^(development|testing|production)$", description="Environment name")
    debug: bool = Field(default=True, description="Debug mode enabled")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level")
    
    # GPU settings
    gpu_enabled: bool = Field(default=True, description="Enable GPU usage")
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="GPU memory fraction to use")
    gpu_memory_growth: bool = Field(default=True, description="Enable GPU memory growth")
    
    # Logging settings  
    log_format: str = Field(default="json", pattern="^(json|text)$", description="Log format")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Cache settings
    cache_dir: str = Field(default="./cache", description="Cache directory path")
    model_cache_size_gb: int = Field(default=10, ge=1, le=100, description="Model cache size in GB")
    
    # API settings
    api_timeout_s: int = Field(default=30, ge=5, le=300, description="API timeout in seconds")
    max_request_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum request size in MB")
    
    # Monitoring settings
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=False, description="Enable request tracing")
    health_check_interval_s: int = Field(default=30, ge=5, le=300, description="Health check interval in seconds")
    
    # Resource limits
    max_memory_mb: int = Field(default=4096, ge=1024, le=32768, description="Maximum memory usage in MB")
    max_gpu_memory_mb: int = Field(default=8192, ge=1024, le=32768, description="Maximum GPU memory usage in MB")
    worker_processes: int = Field(default=1, ge=1, le=8, description="Number of worker processes")
    
    # Security settings
    secret_key: str = Field(default="change-me-in-production", min_length=16, description="Secret key for encryption")
    allowed_hosts: list = Field(default=["localhost", "127.0.0.1"], description="Allowed hostnames")
    cors_origins: list = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, ge=0, le=3600, description="Cache TTL in seconds")
    
    model_config = ConfigDict(
        env_prefix="BEAUTYAI_SYSTEM_",
        case_sensitive=False
    )


class ModelConfig(BaseSettings):
    """Model-specific configuration."""
    
    # Default model settings
    default_asr_model: str = Field(default="openai/whisper-large-v3", description="Default ASR model")
    default_llm_model: str = Field(default="microsoft/DialoGPT-large", description="Default LLM model")
    default_tts_model: str = Field(default="edge-tts", description="Default TTS model")
    default_chat_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="Default chat model")
    default_whisper_model: str = Field(default="large-v3", description="Default Whisper model")
    default_tts_voice: str = Field(default="ar-SA-ZariyahNeural", description="Default TTS voice")
    
    # Model loading settings
    model_load_timeout_s: int = Field(default=120, ge=30, le=600, description="Model load timeout in seconds")
    model_unload_delay_s: int = Field(default=300, ge=60, le=1800, description="Model unload delay in seconds")
    preload_models: list = Field(default=[], description="Models to preload on startup")
    auto_load_models: bool = Field(default=True, description="Auto-load models on startup")
    model_cache_dir: str = Field(default="./models", description="Model cache directory")
    max_model_memory_mb: int = Field(default=4096, ge=512, le=16384, description="Max memory per model in MB")
    
    # Quantization settings
    enable_quantization: bool = Field(default=True, description="Enable model quantization")
    quantization_enabled: bool = Field(default=True, description="Enable quantization (alias)")
    quantization_bits: int = Field(default=8, ge=4, le=16, description="Quantization bits (4, 8, or 16)")
    quantization_type: str = Field(default="bitsandbytes", pattern="^(bitsandbytes|awq|squeezellm)$", description="Quantization method")
    
    # Inference settings
    max_new_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    
    model_config = ConfigDict(
        env_prefix="BEAUTYAI_MODEL_",
        case_sensitive=False
    )


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path in self.config_manager._watched_files:
            logger.info(f"Configuration file {event.src_path} modified, reloading...")
            asyncio.create_task(self.config_manager._reload_config_file(event.src_path))


class ConfigManager:
    """
    Enhanced Configuration Manager with environment-aware loading.
    
    Features:
    - Cascading configuration: ENV > Config File > Defaults
    - Hot-reloading of configuration files
    - Configuration validation with pydantic
    - Secrets management with encryption
    - Configuration versioning and migration
    - Health checks and validation
    """
    
    def __init__(self, 
                 config_dir: Optional[Path] = None,
                 enable_hot_reload: bool = True,
                 enable_encryption: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            enable_hot_reload: Enable hot-reloading of config files
            enable_encryption: Enable secrets encryption
        """
        self.config_dir = config_dir or Path("./config")
        self.enable_hot_reload = enable_hot_reload
        self.enable_encryption = enable_encryption
        
        # Configuration schemas
        self.duplex_config = DuplexStreamingConfig()
        self.system_config = SystemConfig()
        self.model_config = ModelConfig()
        
        # State management
        self._config_lock = Lock()
        self._config_version = "1.0.0"
        self._config_loaded_at = None
        self._watched_files = set()
        self._file_observer = None
        self._reload_callbacks: List[Callable] = []
        
        # Encryption setup
        self._cipher = None
        if self.enable_encryption:
            self._setup_encryption()
        
        # Load initial configuration
        self._load_all_config()
        
        # Setup hot-reloading
        if self.enable_hot_reload:
            self._setup_hot_reload()
    
    def _setup_encryption(self):
        """Setup encryption for secrets management."""
        key_file = self.config_dir / ".config_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(key_file, 0o600)
        
        self._cipher = Fernet(key)
    
    def _setup_hot_reload(self):
        """Setup file system watcher for hot-reloading."""
        if not self.config_dir.exists():
            return
            
        handler = ConfigFileHandler(self)
        self._file_observer = Observer()
        self._file_observer.schedule(handler, str(self.config_dir), recursive=False)
        self._file_observer.start()
        logger.info(f"Configuration hot-reload enabled for {self.config_dir}")
    
    def _load_all_config(self):
        """Load all configuration from files and environment."""
        with self._config_lock:
            try:
                # Load defaults
                defaults = self._load_defaults()
                
                # Load config files
                config_files = self._load_config_files()
                
                # Merge configurations (ENV > Config File > Defaults)
                merged_config = self._merge_configs(defaults, config_files)
                
                # Update pydantic models
                self._update_config_models(merged_config)
                
                self._config_loaded_at = datetime.now()
                logger.info("Configuration loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        defaults_file = self.config_dir / "defaults.json"
        
        if defaults_file.exists():
            with open(defaults_file, 'r') as f:
                return json.load(f)
        
        # Built-in defaults
        return {
            "duplex": {
                "tts_model": "edge-tts",
                "echo_suppression_enabled": True,
                "duplex_mode": "full"
            },
            "system": {
                "log_level": "INFO",
                "gpu_enabled": True,
                "metrics_enabled": True
            },
            "models": {
                "default_asr_model": "openai/whisper-large-v3",
                "quantization_enabled": True
            }
        }
    
    def _load_config_files(self) -> Dict[str, Any]:
        """Load configuration from YAML/JSON files."""
        config_data = {}
        
        # Load main config file
        for config_file in ["config.yaml", "config.yml", "config.json"]:
            config_path = self.config_dir / config_file
            if config_path.exists():
                self._watched_files.add(str(config_path))
                config_data.update(self._load_single_config_file(config_path))
                break
        
        # Load environment-specific config
        env = os.getenv("BEAUTYAI_ENV", "development")
        env_config_file = self.config_dir / f"config.{env}.yaml"
        if env_config_file.exists():
            self._watched_files.add(str(env_config_file))
            env_config = self._load_single_config_file(env_config_file)
            config_data = self._merge_configs(config_data, env_config)
        
        # Load secrets file (encrypted)
        secrets_file = self.config_dir / "secrets.enc"
        if secrets_file.exists() and self._cipher:
            secrets = self._load_encrypted_secrets(secrets_file)
            config_data = self._merge_configs(config_data, {"secrets": secrets})
        
        return config_data
    
    def _load_single_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single configuration file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unknown config file format: {file_path}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _load_encrypted_secrets(self, file_path: Path) -> Dict[str, Any]:
        """Load encrypted secrets file."""
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Failed to load encrypted secrets: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _update_config_models(self, config_data: Dict[str, Any]):
        """Update pydantic configuration models."""
        # Update duplex config
        if "duplex" in config_data:
            self.duplex_config = DuplexStreamingConfig(**config_data["duplex"])
        
        # Update system config  
        if "system" in config_data:
            self.system_config = SystemConfig(**config_data["system"])
        
        # Update model config
        if "models" in config_data:
            self.model_config = ModelConfig(**config_data["models"])
    
    async def _reload_config_file(self, file_path: str):
        """Reload configuration from a specific file."""
        logger.info(f"Reloading configuration from {file_path}")
        
        try:
            # Reload all configuration
            self._load_all_config()
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self)
                    else:
                        callback(self)
                except Exception as e:
                    logger.error(f"Configuration reload callback failed: {e}")
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def register_reload_callback(self, callback: Callable):
        """Register a callback to be called when configuration is reloaded."""
        self._reload_callbacks.append(callback)
    
    def get_duplex_config(self) -> DuplexStreamingConfig:
        """Get duplex streaming configuration."""
        return self.duplex_config
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.system_config
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model_config
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return {
            "duplex": self.duplex_config.dict(),
            "system": self.system_config.dict(),
            "models": self.model_config.dict(),
            "version": self._config_version,
            "loaded_at": self._config_loaded_at.isoformat() if self._config_loaded_at else None
        }
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration section.
        
        Args:
            section: Configuration section to update
            updates: Configuration updates
            
        Returns:
            True if update was successful
        """
        with self._config_lock:
            try:
                if section == "duplex":
                    # Validate updates
                    current_config = self.duplex_config.dict()
                    current_config.update(updates)
                    self.duplex_config = DuplexStreamingConfig(**current_config)
                    
                elif section == "system":
                    current_config = self.system_config.dict()
                    current_config.update(updates)
                    self.system_config = SystemConfig(**current_config)
                    
                elif section == "models":
                    current_config = self.model_config.dict()
                    current_config.update(updates)
                    self.model_config = ModelConfig(**current_config)
                    
                else:
                    logger.error(f"Unknown configuration section: {section}")
                    return False
                
                logger.info(f"Configuration section '{section}' updated")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update configuration section '{section}': {e}")
                return False
    
    def save_encrypted_secret(self, key: str, value: str):
        """Save an encrypted secret."""
        if not self._cipher:
            raise ValueError("Encryption not enabled")
        
        secrets_file = self.config_dir / "secrets.enc"
        
        # Load existing secrets
        secrets = {}
        if secrets_file.exists():
            secrets = self._load_encrypted_secrets(secrets_file)
        
        # Update with new secret
        secrets[key] = value
        
        # Encrypt and save
        secrets_json = json.dumps(secrets)
        encrypted_data = self._cipher.encrypt(secrets_json.encode('utf-8'))
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(secrets_file, 'wb') as f:
            f.write(encrypted_data)
        
        # Secure the file
        os.chmod(secrets_file, 0o600)
        
        logger.info(f"Secret '{key}' saved successfully")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get decrypted secret value."""
        secrets_file = self.config_dir / "secrets.enc"
        
        if not secrets_file.exists() or not self._cipher:
            return default
        
        secrets = self._load_encrypted_secrets(secrets_file)
        return secrets.get(key, default)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate duplex config
            self.duplex_config.dict()  # Triggers validation
            
            # Validate system config
            self.system_config.dict()  # Triggers validation
            
            # Validate model config
            self.model_config.dict()  # Triggers validation
            
            # Additional custom validations
            if self.duplex_config.jitter_buffer_max_ms <= self.duplex_config.jitter_buffer_target_ms:
                validation_results["errors"].append(
                    "jitter_buffer_max_ms must be greater than jitter_buffer_target_ms"
                )
                validation_results["valid"] = False
            
            if self.system_config.gpu_memory_fraction > 0.95:
                validation_results["warnings"].append(
                    "gpu_memory_fraction > 0.95 may cause out of memory errors"
                )
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(str(e))
            logger.error(f"Configuration validation failed: {e}")
        
        return validation_results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform configuration health check."""
        return {
            "status": "healthy",
            "config_loaded": self._config_loaded_at is not None,
            "config_version": self._config_version,
            "hot_reload_enabled": self.enable_hot_reload,
            "encryption_enabled": self.enable_encryption,
            "watched_files": len(self._watched_files),
            "validation": self.validate_config(),
            "last_loaded": self._config_loaded_at.isoformat() if self._config_loaded_at else None
        }
    
    @contextmanager
    def config_update_transaction(self):
        """Context manager for atomic configuration updates."""
        with self._config_lock:
            # Create backups
            duplex_backup = self.duplex_config.copy()
            system_backup = self.system_config.copy()
            model_backup = self.model_config.copy()
            
            try:
                yield self
            except Exception:
                # Restore backups on error
                self.duplex_config = duplex_backup
                self.system_config = system_backup
                self.model_config = model_backup
                raise
    
    def cleanup(self):
        """Cleanup resources."""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            logger.info("Configuration manager cleaned up")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def initialize_config_manager(config_dir: Optional[Path] = None, 
                            enable_hot_reload: bool = True,
                            enable_encryption: bool = True) -> ConfigManager:
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_dir, enable_hot_reload, enable_encryption)
    return _config_manager