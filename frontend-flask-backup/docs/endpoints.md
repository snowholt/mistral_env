# BeautyAI Framework API Endpoints Documentation

## Overview
This document provides comprehensive documentation for the BeautyAI Framework API endpoints. The API is organized into three main modules: Models, Inference, and System management.

## Base Response Format
All endpoints return responses in a consistent format:

```json
{
  "success": boolean,
  "data": object | null,
  "timestamp": "ISO8601_string",
  "execution_time_ms": float | null
}
```

Error responses include additional fields:
```json
{
  "success": false,
  "error_code": "string",
  "error_message": "string", 
  "error_details": object | null
}
```

## Authentication
All endpoints require authentication via the `AuthContext` dependency with specific permissions.

---

## Model Management Endpoints (`/models`)

### 1. List Models
- **Route**: `GET /models/`
- **Description**: List all models in the registry with pagination support
- **Permissions**: `model_read`
- **Query Parameters**:
  - `limit` (optional): Maximum number of models to return (1-100)
  - `offset` (optional): Number of models to skip (default: 0)
- **Response Schema**: `ModelListResponse`
```json
{
  "success": true,
  "models": [
    {
      "model_name": "string",
      "model_id": "string", 
      "architecture": "string",
      "engine": "string",
      "status": "loaded|unloaded",
      "memory_usage_mb": float
    }
  ],
  "total_count": integer
}
```

### 2. Add Model
- **Route**: `POST /models/`
- **Description**: Add a new model configuration to the registry
- **Permissions**: `model_write`
- **Input Schema**: `ModelAddRequest`
```json
{
  "model_name": "string",
  "model_config": {
    "model_id": "string",
    "architecture": "string",
    "engine": "transformers|vllm",
    "quantization": "4bit|8bit|awq|squeezelm|none",
    "parameters": {}
  },
  "set_as_default": boolean
}
```
- **Response Schema**: `ModelAddResponse`
```json
{
  "success": true,
  "model_name": "string",
  "message": "string"
}
```

### 3. Get Model Details
- **Route**: `GET /models/{model_name}`
- **Description**: Get detailed information about a specific model
- **Permissions**: `model_read`
- **Path Parameters**: `model_name` (string)
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "model_name": "string",
    "model_config": {},
    "status": "loaded|unloaded",
    "memory_usage_mb": float,
    "load_time_seconds": float
  }
}
```

### 4. Update Model
- **Route**: `PUT /models/{model_name}`
- **Description**: Update an existing model configuration
- **Permissions**: `model_write`
- **Path Parameters**: `model_name` (string)
- **Input Schema**: 
```json
{
  "updates": {},
  "set_as_default": boolean
}
```
- **Response Schema**: `APIResponse`

### 5. Remove Model
- **Route**: `DELETE /models/{model_name}`
- **Description**: Remove a model from the registry
- **Permissions**: `model_delete`
- **Path Parameters**: `model_name` (string)
- **Query Parameters**: `clear_cache` (boolean)
- **Response Schema**: `APIResponse`

### 6. Load Model
- **Route**: `POST /models/{model_name}/load`
- **Description**: Load a model into memory for inference
- **Permissions**: `model_load`
- **Path Parameters**: `model_name` (string)
- **Input Schema**: `ModelLoadRequest`
```json
{
  "force_reload": boolean
}
```
- **Response Schema**: `ModelLoadResponse`
```json
{
  "success": true,
  "model_name": "string",
  "model_id": "string",
  "memory_usage_mb": float,
  "load_time_seconds": float
}
```

### 7. Unload Model
- **Route**: `POST /models/{model_name}/unload`
- **Description**: Unload a model from memory
- **Permissions**: `model_load`
- **Path Parameters**: `model_name` (string)
- **Response Schema**: `APIResponse`

### 8. Get Model Status
- **Route**: `GET /models/{model_name}/status`
- **Description**: Get current status of a model
- **Permissions**: `model_read`
- **Path Parameters**: `model_name` (string)
- **Response Schema**: `APIResponse`

### 9. Validate Model
- **Route**: `POST /models/{model_name}/validate`
- **Description**: Validate a model configuration
- **Permissions**: `model_read`
- **Path Parameters**: `model_name` (string)
- **Response Schema**: `APIResponse`

### 10. Set Default Model
- **Route**: `POST /models/default/{model_name}`
- **Description**: Set a model as the default
- **Permissions**: `model_write`
- **Path Parameters**: `model_name` (string)
- **Response Schema**: `APIResponse`

---

## Inference Endpoints (`/inference`)

### 1. Chat Completion
- **Route**: `POST /inference/chat`
- **Description**: Generate a chat completion response from a model
- **Permissions**: `chat`
- **Input Schema**: `ChatRequest`
```json
{
  "model_name": "string",
  "message": "string",
  "session_id": "string | null",
  "chat_history": [
    {
      "role": "user|assistant|system",
      "content": "string"
    }
  ],
  "generation_config": {
    "max_tokens": integer,
    "temperature": float,
    "top_p": float,
    "repetition_penalty": float
  },
  "stream": boolean
}
```
- **Response Schema**: `ChatResponse`
```json
{
  "success": true,
  "response": "string",
  "session_id": "string",
  "model_name": "string",
  "generation_stats": {
    "tokens_generated": integer,
    "inference_time_ms": float,
    "tokens_per_second": float
  }
}
```

### 2. Model Test
- **Route**: `POST /inference/test`
- **Description**: Run a single test inference with a model
- **Permissions**: `test`
- **Input Schema**: `TestRequest`
```json
{
  "model_name": "string",
  "prompt": "string",
  "generation_config": {},
  "validation_criteria": {}
}
```
- **Response Schema**: `TestResponse`
```json
{
  "success": true,
  "model_name": "string",
  "prompt": "string",
  "response": "string",
  "generation_stats": {},
  "validation_result": {}
}
```

### 3. Run Benchmark
- **Route**: `POST /inference/benchmark`
- **Description**: Run performance benchmark on a model
- **Permissions**: `benchmark`
- **Input Schema**: `BenchmarkRequest`
```json
{
  "model_name": "string",
  "benchmark_type": "latency|throughput|comprehensive",
  "config": {
    "num_runs": integer,
    "prompt": "string",
    "max_tokens": integer,
    "temperature": float
  }
}
```
- **Response Schema**: `BenchmarkResponse`
```json
{
  "success": true,
  "model_name": "string",
  "benchmark_type": "string",
  "results": {},
  "summary": {}
}
```

### 4. Save Session
- **Route**: `POST /inference/sessions/save`
- **Description**: Save a chat session to storage
- **Permissions**: `session_save`
- **Input Schema**: `SessionSaveRequest`
```json
{
  "session_id": "string",
  "session_data": {},
  "output_file": "string | null"
}
```
- **Response Schema**: `SessionSaveResponse`
```json
{
  "success": true,
  "session_id": "string",
  "file_path": "string",
  "file_size_bytes": integer
}
```

### 5. Load Session
- **Route**: `POST /inference/sessions/load`
- **Description**: Load a previously saved chat session
- **Permissions**: `session_load`
- **Input Schema**: `SessionLoadRequest`
```json
{
  "input_file": "string"
}
```
- **Response Schema**: `SessionLoadResponse`
```json
{
  "success": true,
  "session_data": {},
  "session_id": "string",
  "message_count": integer
}
```

### 6. List Sessions
- **Route**: `GET /inference/sessions`
- **Description**: List available chat sessions
- **Permissions**: `session_load`
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "sessions": [],
    "total_count": integer
  }
}
```

### 7. Delete Session
- **Route**: `DELETE /inference/sessions/{session_name}`
- **Description**: Delete a saved chat session
- **Permissions**: `session_delete`
- **Path Parameters**: `session_name` (string)
- **Response Schema**: `APIResponse`

---

## System Management Endpoints (`/system`)

### 1. System Status
- **Route**: `GET /system/status`
- **Description**: Get comprehensive system status
- **Permissions**: `system_status`
- **Query Parameters**: `detailed` (boolean)
- **Response Schema**: `SystemStatusResponse`
```json
{
  "success": true,
  "system_info": {
    "platform": "string",
    "python_version": "string",
    "framework_version": "string"
  },
  "memory_info": {
    "total_memory_gb": float,
    "available_memory_gb": float,
    "gpu_memory_gb": float,
    "memory_usage_percent": float
  },
  "gpu_info": {
    "gpu_available": boolean,
    "gpu_name": "string",
    "gpu_memory_used_gb": float,
    "gpu_utilization_percent": float
  },
  "model_info": {
    "loaded_models": [],
    "total_loaded": integer,
    "default_model": "string"
  }
}
```

### 2. Memory Status
- **Route**: `GET /system/memory`
- **Description**: Get detailed memory usage information
- **Permissions**: `system_status`
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "system_memory": {
      "total_gb": float,
      "available_gb": float,
      "used_gb": float,
      "usage_percent": float
    },
    "gpu_memory": {
      "total_gb": float,
      "available_gb": float,
      "used_gb": float,
      "usage_percent": float
    },
    "process_memory": {
      "rss_mb": float,
      "vms_mb": float
    }
  }
}
```

### 3. Clear Memory
- **Route**: `POST /system/memory/clear`
- **Description**: Clear unused memory and caches
- **Permissions**: `cache_clear`
- **Query Parameters**: `force` (boolean)
- **Response Schema**: `APIResponse`

### 4. Cache Status
- **Route**: `GET /system/cache`
- **Description**: Get cache status and statistics
- **Permissions**: `system_status`
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "total_cache_size_gb": float,
    "cache_entries": integer,
    "cache_location": "string",
    "oldest_entry": "ISO8601_string",
    "newest_entry": "ISO8601_string"
  }
}
```

### 5. Clear Cache
- **Route**: `POST /system/cache/clear`
- **Description**: Clear model caches
- **Permissions**: `cache_clear`
- **Query Parameters**: 
  - `model_name` (optional): Specific model to clear
  - `force` (boolean): Force clear operation
- **Response Schema**: `APIResponse`

### 6. Resource Usage
- **Route**: `GET /system/resources`
- **Description**: Get current resource usage statistics
- **Permissions**: `system_status`
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "cpu": {
      "usage_percent": float,
      "cores": integer,
      "frequency_mhz": integer
    },
    "memory": {
      "total_gb": float,
      "available_gb": float,
      "usage_percent": float
    },
    "gpu": {
      "usage_percent": float,
      "memory_usage_percent": float,
      "temperature_c": float
    },
    "disk": {
      "total_gb": float,
      "available_gb": float,
      "usage_percent": float
    },
    "network": {
      "bytes_sent": integer,
      "bytes_received": integer
    }
  }
}
```

### 7. Performance Metrics
- **Route**: `GET /system/performance`
- **Description**: Get performance metrics over a time window
- **Permissions**: `system_status`
- **Query Parameters**: `window_minutes` (integer, default: 60)
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "time_window_minutes": integer,
    "metrics": {
      "avg_response_time_ms": float,
      "total_requests": integer,
      "successful_requests": integer,
      "error_rate_percent": float,
      "throughput_requests_per_minute": float,
      "avg_memory_usage_percent": float,
      "avg_gpu_usage_percent": float
    },
    "timestamp": "ISO8601_string"
  }
}
```

### 8. Restart System
- **Route**: `POST /system/restart`
- **Description**: Restart system services
- **Permissions**: `admin`
- **Query Parameters**: `force` (boolean)
- **Response Schema**: `APIResponse`

### 9. System Logs
- **Route**: `GET /system/logs`
- **Description**: Get recent system logs
- **Permissions**: `admin`
- **Query Parameters**: 
  - `level` (string, default: "INFO"): Log level filter
  - `lines` (integer, default: 100): Number of log lines
- **Response Schema**: `APIResponse`
```json
{
  "success": true,
  "data": {
    "log_level": "string",
    "lines_requested": integer,
    "logs": [
      {
        "timestamp": "ISO8601_string",
        "level": "string",
        "message": "string",
        "source": "string"
      }
    ],
    "total_lines": integer
  }
}
```

---

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found (model/resource not found)
- `500`: Internal Server Error

### Common Error Responses
- `ModelNotFoundError`: Model does not exist in registry
- `ModelLoadError`: Failed to load model into memory
- `ValidationError`: Invalid request parameters or configuration
- `SystemError`: System-level operation failure

### Permission Requirements
- `model_read`: View model information
- `model_write`: Add/update models
- `model_delete`: Remove models
- `model_load`: Load/unload models
- `chat`: Chat completions
- `test`: Model testing
- `benchmark`: Performance benchmarking
- `session_save`: Save chat sessions
- `session_load`: Load chat sessions
- `session_delete`: Delete chat sessions
- `system_status`: View system status
- `cache_clear`: Clear caches and memory
- `admin`: Administrative operations

---

## Usage Examples

### Load and Chat with a Model
```bash
# 1. Load model
curl -X POST "/models/qwen-7b/load" \
  -H "Content-Type: application/json" \
  -d '{"force_reload": false}'

# 2. Chat with model
curl -X POST "/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen-7b",
    "message": "Hello, how are you?",
    "generation_config": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

### System Monitoring
```bash
# Get system status
curl -X GET "/system/status?detailed=true"

# Check memory usage
curl -X GET "/system/memory"

# Clear cache
curl -X POST "/system/cache/clear?force=true"
```





### Models:
```
{
  "default_model": "qwen3-model",
  "models": {
    "qwen3-model": {
      "model_id": "Qwen/Qwen3-14B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 1024,
      "name": "qwen3-model",
      "description": "Qwen3 14B model (4-bit quantized)",
      "model_architecture": "causal_lm",
      "custom_generation_params": {
        "temperature": 0.1,
        "top_p": 0.95,
        "do_sample": true,
        "repetition_penalty": 1.1
      }
    },
    "bee1reason-arabic-qwen-14b": {
      "model_id": "beetlware/Bee1reason-arabic-Qwen-14B",
      "engine_type": "transformers",
      "quantization": "none",
      "dtype": "float16",
      "max_new_tokens": 1024,
      "name": "bee1reason-arabic-qwen-14b",
      "description": "Arabic-optimized Qwen 14B model (float16)",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": true,
        "repetition_penalty": 1.1
      }
    },
    "bee1reason-arabic-qwen-14b-gguf": {
      "model_id": "beetlware/Bee1reason-arabic-Qwen-14B-Q4_K_M-GGUF",
      "engine_type": "llama.cpp",
      "quantization": "Q4_K_M",
      "dtype": "float16",
      "max_new_tokens": 1024,
      "name": "bee1reason-arabic-qwen-14b-gguf",
      "description": "Arabic-optimized Qwen 14B model (GGUF Q4_K_M format)",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.1
      }
    },
    "llama-4-maverick-17b": {
      "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 1024,
      "name": "llama-4-maverick-17b",
      "description": "Meta Llama-4 Maverick 17B Instruct model",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.6,
        "top_p": 0.9,
        "do_sample": true,
        "repetition_penalty": 1.1
      }
    },
    "deepseek-r1-qwen-14b-multilingual": {
      "model_id": "lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 1024,
      "name": "deepseek-r1-qwen-14b-multilingual",
      "description": "DeepSeek R1 Distill Qwen 14B Multilingual model",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": true,
        "repetition_penalty": 1.1
      }
    },
    "arabic-deepseek-r1-distill-8b": {
      "model_id": "Omartificial-Intelligence-Space/Arabic-DeepSeek-R1-Distill-8B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 1024,
      "name": "arabic-deepseek-r1-distill-8b",
      "description": "Arabic-optimized DeepSeek R1 Distill 8B model for reasoning",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.5,
        "top_p": 0.9,
        "do_sample": true,
        "repetition_penalty": 1.15
      }
    },
    "deepseek-r1-qwen-14b-multilingual-gguf": {
      "model_id": "pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF",
      "engine_type": "llama.cpp",
      "quantization": "Q4_K_M",
      "dtype": "float16",
      "max_new_tokens": 1024,
      "name": "deepseek-r1-qwen-14b-multilingual-gguf",
      "description": "DeepSeek R1 Distill Qwen 14B Multilingual model (GGUF Q4_K_M format)",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.1
      }
    },

    "arabic-deepseek-r1-distill-llama3-8b": {
      "model_id": "Paula139/DeepSeek-R1-destill-llama3-8b-arabic-fine-tuned",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 1024,
      "name": "arabic-deepseek-r1-distill-llama3-8b",
      "description": "Arabic fine-tuned DeepSeek R1 Distill Llama3 8B model for reasoning",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.5,
        "top_p": 0.9,
        "do_sample": true,
        "repetition_penalty": 1.15
      }
    },
    "arabic-morph-deepseek-r1-distill-llama-8b": {
      "model_id": "omarxadel/Arabic-Morph-DeepSeek-R1-Distill-Llama-8B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "bfloat16",
      "max_new_tokens": 512,
      "name": "arabic-morph-deepseek-r1-distill-llama-8b",
      "description": "Arabic Morphological DeepSeek R1 Distill Llama 8B model",
      "model_architecture": "causal_lm",
      
      "custom_generation_params": {
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": true,
        "repetition_penalty": 1.1
      }
    }
  }
}
```