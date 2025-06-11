"""
API Request/Response Data Models.

Defines the data structures used for API communication, designed to be
compatible with REST/GraphQL endpoints and JSON serialization.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json


@dataclass
class APIRequest:
    """Base class for all API requests."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create request from dictionary (from JSON deserialization)."""
        return cls(**data)


@dataclass
class APIResponse:
    """Base class for all API responses."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ErrorResponse(APIResponse):
    """Standard error response format."""
    success: bool = False
    error_code: str = ""
    error_message: str = ""
    error_details: Optional[Dict[str, Any]] = None


# Model Management API Models
@dataclass
class ModelListRequest(APIRequest):
    """Request to list models in registry."""
    pass


@dataclass
class ModelListResponse(APIResponse):
    """Response containing list of models."""
    success: bool = True
    models: List[Dict[str, Any]] = None
    total_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.models is None:
            self.models = []


@dataclass
class ModelAddRequest(APIRequest):
    """Request to add a model to registry."""
    model_name: str
    model_config: Dict[str, Any]
    set_as_default: bool = False


@dataclass
class ModelAddResponse(APIResponse):
    """Response for model addition."""
    success: bool = True
    model_name: str = ""
    message: str = ""


@dataclass
class ModelLoadRequest(APIRequest):
    """Request to load a model into memory."""
    force_reload: bool = False


@dataclass
class ModelLoadResponse(APIResponse):
    """Response for model loading."""
    success: bool = True
    model_name: str = ""
    model_id: str = ""
    memory_usage_mb: Optional[float] = None
    load_time_seconds: Optional[float] = None


# Chat API Models
@dataclass
class ChatRequest(APIRequest):
    """Enhanced request for chat interaction with direct parameter access."""
    model_name: str
    message: str
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    
    # Core Generation Parameters (Direct Access)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    
    # Advanced Sampling Parameters
    min_p: Optional[float] = None           # Minimum probability threshold
    typical_p: Optional[float] = None       # Typical sampling parameter
    epsilon_cutoff: Optional[float] = None  # Epsilon cutoff for sampling
    eta_cutoff: Optional[float] = None      # Eta cutoff for sampling
    diversity_penalty: Optional[float] = None
    encoder_repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    
    # Beam Search Parameters
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    
    # Thinking Mode Control
    enable_thinking: Optional[bool] = None  # True/False/None (auto-detect)
    thinking_mode: Optional[str] = None     # "auto", "force", "disable"
    
    # Content Filtering Control
    disable_content_filter: bool = False    # Disable content filtering entirely
    content_filter_strictness: Optional[str] = None  # "strict", "balanced", "relaxed", "disabled"
    
    # Preset Configurations (Based on Optimization Results)
    preset: Optional[str] = None  # "conservative", "balanced", "creative", "speed_optimized", "qwen_optimized"
    
    # Legacy support (will be merged with direct parameters)
    generation_config: Optional[Dict[str, Any]] = None
    stream: bool = False
    
    def get_effective_generation_config(self) -> Dict[str, Any]:
        """
        Build the effective generation configuration from all sources.
        Priority: Direct parameters > Preset > generation_config > defaults
        """
        # Start with preset-based defaults
        config = self._get_preset_config()
        
        # Merge legacy generation_config
        if self.generation_config:
            config.update(self.generation_config)
        
        # Override with direct parameters (highest priority)
        direct_params = {
            # Core parameters
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'do_sample': self.do_sample,
            
            # Advanced sampling
            'min_p': self.min_p,
            'typical_p': self.typical_p,
            'epsilon_cutoff': self.epsilon_cutoff,
            'eta_cutoff': self.eta_cutoff,
            'diversity_penalty': self.diversity_penalty,
            'encoder_repetition_penalty': self.encoder_repetition_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size,
            
            # Beam search
            'num_beams': self.num_beams,
            'num_beam_groups': self.num_beam_groups,
            'length_penalty': self.length_penalty,
            'early_stopping': self.early_stopping,
        }
        
        # Add non-None direct parameters
        for key, value in direct_params.items():
            if value is not None:
                config[key] = value
        
        return config
    
    def _get_preset_config(self) -> Dict[str, Any]:
        """Get configuration based on preset optimization results."""
        presets = {
            "conservative": {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 10,
                "repetition_penalty": 1.0,
                "max_new_tokens": 256,
                "do_sample": True
            },
            "balanced": {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "max_new_tokens": 512,
                "do_sample": True
            },
            "creative": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
                "do_sample": True
            },
            "speed_optimized": {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 10,
                "repetition_penalty": 1.0,
                "max_new_tokens": 256,
                "do_sample": True
            },
            # Based on your actual optimization results for qwen3-unsloth-q4ks
            "qwen_optimized": {
                "temperature": 0.3,         # Best result from 20250610_224757
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "max_new_tokens": 256,
                "do_sample": True
            },
            # High quality preset based on demo results
            "high_quality": {
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": 10,
                "repetition_penalty": 1.15,
                "max_new_tokens": 1024,
                "do_sample": True
            },
            # Creative high-performance preset
            "creative_optimized": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 80,
                "repetition_penalty": 1.15,
                "max_new_tokens": 1024,
                "do_sample": True
            }
        }
        
        return presets.get(self.preset, presets["balanced"])
    
    def should_enable_thinking(self) -> bool:
        """Determine if thinking mode should be enabled."""
        if self.thinking_mode == "force":
            return True
        elif self.thinking_mode == "disable":
            return False
        elif self.enable_thinking is not None:
            return self.enable_thinking
        else:
            # Auto-detect: check for /no_think in message
            return not (self.message.strip().startswith("/no_think") or "/no_think" in self.message.lower())
    
    def get_processed_message(self) -> str:
        """Get the message with thinking mode processing applied."""
        message = self.message.strip()
        
        # Handle /no_think command
        if message.startswith("/no_think"):
            # Remove the command and return clean message
            return message[9:].strip()
        
        # Add thinking system message prefix if thinking is enabled
        if self.should_enable_thinking():
            # Check if model supports thinking (Qwen3, etc.)
            return message  # Thinking will be handled by the model
        else:
            # Explicitly disable thinking
            return message
    
    def get_effective_content_filter_config(self) -> Dict[str, Any]:
        """Get the effective content filter configuration."""
        if self.disable_content_filter:
            return {"strictness_level": "disabled"}
        
        if self.content_filter_strictness:
            return {"strictness_level": self.content_filter_strictness}
        
        # Default based on preset
        preset_filter_map = {
            "speed_optimized": "relaxed",
            "conservative": "strict", 
            "balanced": "balanced",
            "creative": "relaxed",
            "qwen_optimized": "balanced",
            "high_quality": "balanced",
            "creative_optimized": "relaxed"
        }
        
        strictness = preset_filter_map.get(self.preset, "balanced")
        return {"strictness_level": strictness}


@dataclass
class ChatResponse(APIResponse):
    """Enhanced response for chat interaction with detailed generation info."""
    success: bool = True
    response: str = ""
    session_id: str = ""
    model_name: str = ""
    
    # Generation Statistics
    generation_stats: Optional[Dict[str, Any]] = None
    
    # Parameter Information
    effective_config: Optional[Dict[str, Any]] = None  # What parameters were actually used
    preset_used: Optional[str] = None
    thinking_enabled: Optional[bool] = None
    
    # Content Filtering Information
    content_filter_applied: Optional[bool] = None     # Whether content filtering was applied
    content_filter_strictness: Optional[str] = None   # Strictness level used
    content_filter_bypassed: Optional[bool] = None    # Whether filtering was bypassed
    
    # Performance Metrics
    tokens_generated: Optional[int] = None
    generation_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Response Metadata
    thinking_content: Optional[str] = None  # If thinking was used, the thinking part
    final_content: Optional[str] = None     # The final answer part


# Configuration API Models
@dataclass
class ConfigGetRequest(APIRequest):
    """Request to get configuration."""
    section: Optional[str] = None


@dataclass
class ConfigGetResponse(APIResponse):
    """Response containing configuration."""
    success: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.config is None:
            self.config = {}


@dataclass
class ConfigSetRequest(APIRequest):
    """Request to set configuration value."""
    key: str
    value: Any
    section: Optional[str] = None


@dataclass
class ConfigSetResponse(APIResponse):
    """Response for configuration update."""
    success: bool = True
    key: str = ""
    old_value: Any = None
    new_value: Any = None


# System Status API Models
@dataclass
class SystemStatusRequest(APIRequest):
    """Request for system status."""
    include_memory: bool = True
    include_models: bool = True
    include_cache: bool = True


@dataclass
class SystemStatusResponse(APIResponse):
    """Response containing system status."""
    success: bool = True
    system_info: Dict[str, Any] = None
    memory_info: Optional[Dict[str, Any]] = None
    loaded_models: Optional[List[Dict[str, Any]]] = None
    cache_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.system_info is None:
            self.system_info = {}


# Test API Models
@dataclass
class TestRequest(APIRequest):
    """Request to test a model."""
    model_name: str
    prompt: str
    generation_config: Optional[Dict[str, Any]] = None
    validation_criteria: Optional[Dict[str, Any]] = None


@dataclass
class TestResponse(APIResponse):
    """Response for model test."""
    success: bool = True
    model_name: str = ""
    prompt: str = ""
    response: str = ""
    generation_stats: Dict[str, Any] = None
    validation_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.generation_stats is None:
            self.generation_stats = {}


# Benchmark API Models
@dataclass
class BenchmarkRequest(APIRequest):
    """Request to benchmark a model."""
    model_name: str
    benchmark_type: str  # "latency", "throughput", "comprehensive"
    config: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResponse(APIResponse):
    """Response for model benchmark."""
    success: bool = True
    model_name: str = ""
    benchmark_type: str = ""
    results: Dict[str, Any] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.results is None:
            self.results = {}
        if self.summary is None:
            self.summary = {}


# Session Management API Models
@dataclass
class SessionSaveRequest(APIRequest):
    """Request to save a chat session."""
    session_id: str
    session_data: Dict[str, Any]
    output_file: Optional[str] = None


@dataclass
class SessionSaveResponse(APIResponse):
    """Response for session save."""
    success: bool = True
    session_id: str = ""
    file_path: str = ""
    file_size_bytes: int = 0


@dataclass
class SessionLoadRequest(APIRequest):
    """Request to load a chat session."""
    input_file: str


@dataclass
class SessionLoadResponse(APIResponse):
    """Response for session load."""
    success: bool = True
    session_data: Dict[str, Any] = None
    session_id: str = ""
    message_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.session_data is None:
            self.session_data = {}


# Type aliases for convenience
ModelRequest = Union[ModelListRequest, ModelAddRequest, ModelLoadRequest]
ModelResponse = Union[ModelListResponse, ModelAddResponse, ModelLoadResponse]
ConfigRequest = Union[ConfigGetRequest, ConfigSetRequest]
ConfigResponse = Union[ConfigGetResponse, ConfigSetResponse]
SystemRequest = SystemStatusRequest
SystemResponse = SystemStatusResponse


# Backup/Restore API Models
@dataclass
class BackupRequest(APIRequest):
    """Request to create a configuration backup."""
    backup_name: str
    description: Optional[str] = None
    include_models: bool = True
    include_cache: bool = False


@dataclass
class BackupResponse(APIResponse):
    """Response for configuration backup."""
    success: bool = True
    backup_name: str = ""
    timestamp: str = ""
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    message: str = ""


@dataclass
class RestoreRequest(APIRequest):
    """Request to restore configuration from backup."""
    backup_name: str
    confirm: bool = False
    restore_models: bool = True


@dataclass
class RestoreResponse(APIResponse):
    """Response for configuration restore."""
    success: bool = True
    backup_name: str = ""
    restored_timestamp: str = ""
    message: str = ""
