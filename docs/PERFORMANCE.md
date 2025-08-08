# Performance Guide

Performance optimization and monitoring guide for the BeautyAI Inference Framework.

## 🎯 Performance Overview

The BeautyAI framework is designed for high-performance AI inference with multiple optimization layers:

- **Model-level**: Quantization, caching, batch processing
- **System-level**: GPU memory management, CPU optimization
- **Network-level**: WebSocket optimization, request batching
- **Infrastructure-level**: Load balancing, horizontal scaling

## 📊 Benchmarking

### Quick Performance Test
```bash
# Run comprehensive benchmark
beautyai benchmark --model qwen3-14b-instruct --duration 60 --concurrent 5

# Voice-specific benchmark
beautyai voice-benchmark --duration 30 --voice-samples 10

# Memory usage analysis
beautyai model info --model qwen3-14b-instruct --include-memory
```

### Detailed Benchmarking Script
```python
#!/usr/bin/env python3
"""
BeautyAI Performance Benchmark Suite
"""
import asyncio
import time
import statistics
import json
from typing import List, Dict
import aiohttp
import psutil
import GPUtil

class PerformanceBenchmark:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {
            "chat_latency": [],
            "voice_latency": [],
            "throughput": [],
            "memory_usage": [],
            "gpu_usage": []
        }
    
    async def benchmark_chat_latency(self, num_requests: int = 100):
        """Measure chat response latency"""
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start_time = time.time()
                
                async with session.post(
                    f"{self.api_url}/api/chat",
                    json={
                        "message": f"Hello, this is test message {i}",
                        "temperature": 0.7,
                        "max_tokens": 50
                    }
                ) as response:
                    await response.json()
                
                latency = time.time() - start_time
                self.results["chat_latency"].append(latency)
                
                if i % 10 == 0:
                    print(f"Chat latency test: {i}/{num_requests}")
    
    async def benchmark_voice_latency(self, num_requests: int = 20):
        """Measure voice response latency"""
        # Implementation for voice latency testing
        pass
    
    def monitor_system_resources(self, duration: int = 60):
        """Monitor CPU, RAM, and GPU usage"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU
            try:
                gpus = GPUtil.getGPUs()
                gpu_usage = gpus[0].load * 100 if gpus else 0
                gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
            except:
                gpu_usage = 0
                gpu_memory = 0
            
            self.results["memory_usage"].append({
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "ram_percent": memory.percent,
                "gpu_percent": gpu_usage,
                "gpu_memory_percent": gpu_memory
            })
            
            time.sleep(1)
    
    def generate_report(self) -> Dict:
        """Generate performance report"""
        report = {
            "summary": {
                "chat_latency_mean": statistics.mean(self.results["chat_latency"]) if self.results["chat_latency"] else 0,
                "chat_latency_p95": statistics.quantiles(self.results["chat_latency"], n=20)[18] if len(self.results["chat_latency"]) > 1 else 0,
                "chat_latency_p99": statistics.quantiles(self.results["chat_latency"], n=100)[98] if len(self.results["chat_latency"]) > 1 else 0,
            },
            "raw_data": self.results
        }
        return report

# Usage example
async def main():
    benchmark = PerformanceBenchmark()
    
    print("Starting performance benchmark...")
    await benchmark.benchmark_chat_latency(50)
    
    report = benchmark.generate_report()
    print(f"Average chat latency: {report['summary']['chat_latency_mean']:.3f}s")
    print(f"95th percentile: {report['summary']['chat_latency_p95']:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚡ Model Optimization

### Quantization Settings

#### 4-bit Quantization (Recommended)
```json
{
    "model_name": "qwen3-14b-instruct",
    "quantization": {
        "enabled": true,
        "type": "4bit",
        "compute_dtype": "float16",
        "quant_type": "nf4",
        "use_double_quant": true
    },
    "expected_memory_gb": 8.5,
    "expected_speedup": "2.5x"
}
```

#### 8-bit Quantization (Balanced)
```json
{
    "model_name": "qwen3-14b-instruct",
    "quantization": {
        "enabled": true,
        "type": "8bit",
        "compute_dtype": "float16"
    },
    "expected_memory_gb": 14.2,
    "expected_speedup": "1.8x"
}
```

#### Full Precision (Maximum Quality)
```json
{
    "model_name": "qwen3-14b-instruct",
    "quantization": {
        "enabled": false
    },
    "expected_memory_gb": 28.0,
    "expected_speedup": "1.0x"
}
```

### Model Caching Strategy
```python
class ModelCache:
    def __init__(self, max_models: int = 2):
        self.max_models = max_models
        self.cache = OrderedDict()
        self.access_times = {}
    
    def get_model(self, model_name: str):
        if model_name in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(model_name)
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        return None
    
    def add_model(self, model_name: str, model):
        if len(self.cache) >= self.max_models:
            # Remove least recently used
            oldest_model = self.cache.popitem(last=False)
            self._unload_model(oldest_model[1])
        
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()
```

## 🚀 System Optimization

### GPU Memory Management
```bash
# Monitor GPU memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

# Clear CUDA cache (if needed)
python -c "import torch; torch.cuda.empty_cache()"

# Set GPU memory growth (for TensorFlow models)
export TF_MEMORY_GROWTH=True
```

### CPU Optimization
```bash
# Set CPU affinity for better performance
taskset -c 0-7 python run_server.py  # Use specific CPU cores

# Optimize Python threading
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use jemalloc for better memory allocation
sudo apt install libjemalloc2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

### Memory Optimization
```python
# Model configuration for memory efficiency
MODEL_CONFIG = {
    "torch_dtype": "float16",  # Use half precision
    "device_map": "auto",      # Automatic device mapping
    "low_cpu_mem_usage": True, # Reduce CPU memory during loading
    "use_cache": True,         # Enable KV cache for faster inference
    "pad_token_id": 0,         # Set pad token to avoid warnings
    "max_memory": {0: "20GB"}  # Limit GPU memory usage
}
```

## 🌐 Network Optimization

### WebSocket Optimization
```python
class OptimizedWebSocketHandler:
    def __init__(self):
        self.compression = "deflate"  # Enable compression
        self.max_message_size = 16 * 1024 * 1024  # 16MB max message
        self.ping_interval = 20  # Ping every 20 seconds
        self.ping_timeout = 10   # Timeout after 10 seconds
    
    async def handle_voice_stream(self, websocket):
        # Buffer audio chunks for efficiency
        audio_buffer = []
        buffer_size = 4096  # 4KB chunks
        
        async for message in websocket:
            audio_buffer.append(message)
            
            # Process when buffer is full or timeout
            if len(audio_buffer) >= buffer_size:
                await self.process_audio_batch(audio_buffer)
                audio_buffer.clear()
```

### Request Batching
```python
class RequestBatcher:
    def __init__(self, batch_size: int = 4, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()
    
    async def add_request(self, request):
        async with self.batch_lock:
            self.pending_requests.append(request)
            
            if len(self.pending_requests) >= self.batch_size:
                batch = self.pending_requests.copy()
                self.pending_requests.clear()
                return await self.process_batch(batch)
        
        # Wait for timeout if batch not full
        await asyncio.sleep(self.timeout)
        return await self.flush_batch()
```

## 📈 Performance Monitoring

### Real-time Metrics Dashboard
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "requests_per_second": 0,
            "average_latency": 0,
            "memory_usage": 0,
            "gpu_utilization": 0,
            "error_rate": 0
        }
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0
    
    def record_request(self, latency: float, success: bool = True):
        self.request_count += 1
        self.total_latency += latency
        
        # Update metrics
        elapsed_time = time.time() - self.start_time
        self.metrics["requests_per_second"] = self.request_count / elapsed_time
        self.metrics["average_latency"] = self.total_latency / self.request_count
    
    def get_system_metrics(self):
        # CPU and Memory
        self.metrics["memory_usage"] = psutil.virtual_memory().percent
        
        # GPU
        try:
            gpu = GPUtil.getGPUs()[0]
            self.metrics["gpu_utilization"] = gpu.load * 100
            self.metrics["gpu_memory"] = gpu.memoryUtil * 100
        except:
            pass
        
        return self.metrics
```

### Prometheus Metrics (Optional)
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('beautyai_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('beautyai_request_duration_seconds', 'Request latency')
MEMORY_USAGE = Gauge('beautyai_memory_usage_bytes', 'Memory usage')
GPU_UTILIZATION = Gauge('beautyai_gpu_utilization_percent', 'GPU utilization')

# Usage in API endpoints
@REQUEST_LATENCY.time()
async def chat_endpoint():
    REQUEST_COUNT.inc()
    # ... endpoint logic
```

## 🔧 Configuration Tuning

### FastAPI Optimization
```python
# run_server.py optimization
app = FastAPI(
    title="BeautyAI API",
    docs_url="/docs" if DEBUG else None,  # Disable docs in production
    redoc_url=None,  # Disable redoc
)

# Add middleware for performance
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Optimize CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Run with optimized settings
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=2,  # Multiple workers for CPU-bound tasks
        loop="uvloop",  # Faster event loop
        access_log=False,  # Disable access logs for performance
        server_header=False,  # Disable server header
        date_header=False,  # Disable date header
    )
```

### Model Loading Optimization
```python
# Optimized model loading
def load_model_optimized(model_name: str, quantization_config: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set memory allocation strategy
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        use_cache=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # If available
    )
    
    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="max-autotune")
    
    return model
```

## 📊 Performance Benchmarks

### Expected Performance (Single GPU)

| Model | Precision | Memory (GB) | Tokens/sec | Latency (s) |
|-------|-----------|-------------|------------|-------------|
| Qwen3-14B | 4-bit | 8.5 | 35-45 | 0.8-1.2 |
| Qwen3-14B | 8-bit | 14.2 | 25-35 | 1.0-1.5 |
| Qwen3-14B | FP16 | 28.0 | 15-25 | 1.5-2.0 |
| Mistral-7B | 4-bit | 4.2 | 55-70 | 0.5-0.8 |
| Llama2-13B | 4-bit | 7.8 | 30-40 | 0.9-1.3 |

### Voice Processing Benchmarks

| Component | Processing Time | Notes |
|-----------|----------------|--------|
| Speech-to-Text | 0.3-0.8s | Depends on audio length |
| Language Model | 0.8-1.5s | Main bottleneck |
| Text-to-Speech | 0.2-0.5s | Edge TTS is very fast |
| **Total End-to-End** | **1.3-2.8s** | Target: <2s for good UX |

## 🎯 Performance Tuning Checklist

### ✅ Model Level
- [ ] Enable appropriate quantization (4-bit recommended)
- [ ] Use half precision (float16) when possible
- [ ] Enable model compilation (PyTorch 2.0+)
- [ ] Configure optimal max_new_tokens
- [ ] Use KV caching for faster inference

### ✅ System Level
- [ ] Monitor GPU memory usage
- [ ] Set appropriate CPU affinity
- [ ] Configure memory allocators (jemalloc)
- [ ] Enable GPU optimizations (TF32, Flash Attention)
- [ ] Monitor system resources continuously

### ✅ Network Level
- [ ] Enable WebSocket compression
- [ ] Implement request batching
- [ ] Use HTTP/2 where possible
- [ ] Configure appropriate timeouts
- [ ] Enable gzip compression

### ✅ Infrastructure Level
- [ ] Use SSD storage for models
- [ ] Configure proper cooling for sustained loads
- [ ] Set up load balancing for multiple instances
- [ ] Monitor and alert on performance metrics
- [ ] Regular performance testing

---

**Next**: [Troubleshooting Guide](TROUBLESHOOTING.md) | [Architecture Guide](ARCHITECTURE.md)
