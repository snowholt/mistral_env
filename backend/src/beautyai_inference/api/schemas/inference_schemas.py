"""
Inference API Schemas.

Defines request and response schemas for inference operations.
"""
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatMessage:
    """Schema for chat message."""
    role: str  # system, user, assistant
    content: str


@dataclass
class ChatRequest:
    """Request schema for chat completion."""
    model_name: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


@dataclass
class ChatChoice:
    """Schema for chat completion choice."""
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, str]] = None  # For streaming
    index: int = 0
    finish_reason: Optional[str] = None


@dataclass
class ChatUsage:
    """Schema for token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """Response schema for chat completion."""
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None
    model: str = ""
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class TestRequest:
    """Request schema for model testing."""
    model_name: str
    test_type: str = "basic"  # basic, advanced, custom
    custom_prompts: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7


@dataclass
class TestResult:
    """Schema for individual test result."""
    prompt: str
    response: str
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    token_count: Optional[int] = None


@dataclass
class TestSummary:
    """Schema for test summary statistics."""
    total_tests: int
    passed: int
    failed: int
    success_rate: float
    avg_latency_ms: Optional[float] = None
    avg_tokens: Optional[float] = None


@dataclass
class TestResponse:
    """Response schema for model testing."""
    model_name: str
    test_type: str
    results: List[TestResult]
    summary: TestSummary
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkRequest:
    """Request schema for model benchmarking."""
    model_name: str
    benchmark_type: str = "latency"  # latency, throughput, memory
    num_requests: int = 10
    concurrent: bool = False
    prompt_length: Optional[int] = None
    max_tokens: Optional[int] = None


@dataclass
class BenchmarkMetrics:
    """Schema for benchmark metrics."""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class BenchmarkStatistics:
    """Schema for benchmark statistics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_time_ms: float
    total_tokens: Optional[int] = None


@dataclass
class BenchmarkResponse:
    """Response schema for model benchmarking."""
    model_name: str
    benchmark_type: str
    num_requests: int
    concurrent: bool
    metrics: BenchmarkMetrics
    statistics: BenchmarkStatistics
    timestamp: datetime = field(default_factory=datetime.now)
