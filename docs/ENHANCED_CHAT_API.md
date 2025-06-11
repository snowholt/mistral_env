# Enhanced Chat API Documentation

## Overview

The improved chat endpoint makes parameter tuning much more user-friendly by allowing direct parameter specification instead of complex JSON formatting.

## Key Improvements

### 1. Direct Parameter Access
Instead of complex JSON in `generation_config`, you can now specify parameters directly:

```json
{
  "model_name": "qwen3-model",
  "message": "What is artificial intelligence?",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 20,
  "repetition_penalty": 1.1,
  "max_new_tokens": 512
}
```

### 2. Thinking Mode Control
Easy thinking mode control with `/no_think` command or explicit parameters:

```json
{
  "model_name": "qwen3-model", 
  "message": "/no_think Explain AI briefly",
  "preset": "speed_optimized"
}
```

Or explicitly:
```json
{
  "model_name": "qwen3-model",
  "message": "Explain artificial intelligence",
  "thinking_mode": "disable",
  "temperature": 0.1
}
```

### 3. Optimization-Based Presets

Based on your parameter optimization results, we now have smart presets:

#### Conservative (High Speed, Good Quality)
```json
{
  "model_name": "qwen3-model",
  "message": "Hello",
  "preset": "conservative"
}
```
- **Performance**: Speed score 100, Quality 6.6, Efficiency 660
- **Settings**: temp=0.1, top_p=0.8, top_k=10, rep_penalty=1.0, tokens=256

#### Balanced (Best Overall) 
```json
{
  "model_name": "qwen3-model", 
  "message": "Hello",
  "preset": "balanced"
}
```
- **Performance**: Speed score 100, Quality 8.95, Efficiency 895
- **Settings**: temp=0.5, top_p=0.9, top_k=20, rep_penalty=1.05, tokens=512

#### Creative (Maximum Quality)
```json
{
  "model_name": "qwen3-model",
  "message": "Hello", 
  "preset": "creative"
}
```
- **Performance**: Speed score 100, Quality 8.95, Efficiency 895
- **Settings**: temp=0.7, top_p=0.95, top_k=40, rep_penalty=1.1, tokens=1024

#### Speed Optimized (RTX 4090 Tuned)
```json
{
  "model_name": "qwen3-model",
  "message": "Hello",
  "preset": "speed_optimized"
}
```
- **Target**: 50-100+ tokens/second
- **Settings**: Optimized for RTX 4090 hardware

## Parameter Priority System

The system uses a smart priority system:

1. **Direct Parameters** (highest priority)
2. **Preset Configuration** 
3. **Legacy generation_config**
4. **Model defaults** (lowest priority)

Example combining approaches:
```json
{
  "model_name": "qwen3-model",
  "message": "Explain quantum computing",
  "preset": "balanced",
  "temperature": 0.8,  // Overrides preset temperature
  "thinking_mode": "force"
}
```

## Enhanced Response Format

The response now includes detailed information:

```json
{
  "success": true,
  "response": "Artificial intelligence is...",
  "model_name": "qwen3-model",
  "effective_config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    // ... actual parameters used
  },
  "preset_used": "balanced",
  "thinking_enabled": true,
  "tokens_generated": 156,
  "generation_time_ms": 2340.5,
  "tokens_per_second": 66.7,
  "thinking_content": "Let me think about this...",
  "final_content": "Artificial intelligence is...",
  "generation_stats": {
    "performance": {
      "total_time_ms": 2450.0,
      "generation_time_ms": 2340.5,
      "tokens_generated": 156,
      "tokens_per_second": 66.7,
      "thinking_tokens": 23
    }
  }
}
```

## Qwen3 Model Optimizations

Based on your test results, special handling for Qwen3 models:

### Thinking Mode Parameters
- **With thinking**: temp=0.6, top_p=0.95, top_k=20, rep_penalty=1.5
- **Without thinking**: temp=0.7, top_p=0.8, top_k=20, rep_penalty=1.5

### Automatic Detection
The system automatically detects Qwen3 models and applies optimized settings.

## Usage Examples

### Quick Speed Test
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "What is AI?",
    "preset": "speed_optimized",
    "max_new_tokens": 32
  }'
```

### High Quality Response
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model", 
    "message": "Explain the implications of quantum computing",
    "preset": "creative",
    "thinking_mode": "force"
  }'
```

### Custom Fine-Tuning
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "Hello",
    "temperature": 0.3,
    "top_p": 0.85,
    "top_k": 15,
    "repetition_penalty": 1.08,
    "max_new_tokens": 400
  }'
```

## Performance Monitoring

The response includes detailed performance metrics for optimization:

- **Generation Time**: Pure model inference time
- **Tokens per Second**: Real performance measurement
- **Total Time**: Including API overhead
- **Thinking Analysis**: Separate thinking vs final content metrics

This allows you to:
1. Monitor actual performance vs expectations
2. Compare different parameter combinations
3. Optimize for your specific use cases
4. Track model efficiency over time

## Migration Guide

### Old Format (Complex)
```json
{
  "model_name": "qwen3-model",
  "message": "Hello",
  "generation_config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
    "do_sample": true
  }
}
```

### New Format (Simple)
```json
{
  "model_name": "qwen3-model", 
  "message": "Hello",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 20,
  "repetition_penalty": 1.1,
  "max_new_tokens": 512
}
```

### Even Simpler (Preset)
```json
{
  "model_name": "qwen3-model",
  "message": "Hello", 
  "preset": "balanced"
}
```

The old format is still supported for backward compatibility!
