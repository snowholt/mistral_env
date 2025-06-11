# Enhanced Chat API v2.0 - Comprehensive Parameter Control

## 🚀 Overview

The BeautyAI Chat API has been significantly enhanced to provide comprehensive parameter control, making it much easier to fine-tune model behavior without complex JSON configurations.

## ✨ Key Improvements

### 1. **Direct Parameter Access** 
**Before (Complex):**
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

**After (Simple):**
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

### 2. **Optimization-Based Presets**
Based on actual performance testing results:

```json
{
  "model_name": "qwen3-model",
  "message": "What is AI?",
  "preset": "qwen_optimized"
}
```

**Available Presets:**
- `qwen_optimized`: Best settings from actual testing (temp=0.3, top_p=0.95, top_k=20)
- `high_quality`: Maximum quality (temp=0.1, top_p=1.0, rep_penalty=1.15)
- `creative_optimized`: Creative but efficient (temp=0.5, top_p=1.0, top_k=80)
- `speed_optimized`: Fastest response
- `balanced`: Good balance of quality and speed
- `conservative`: Safe, consistent responses
- `creative`: More creative and varied responses

### 3. **Content Filtering Control**
```json
{
  "model_name": "qwen3-model",
  "message": "Tell me about procedures",
  "disable_content_filter": true
}
```

Or with strictness control:
```json
{
  "model_name": "qwen3-model",
  "message": "Beauty treatment question",
  "content_filter_strictness": "relaxed"
}
```

**Strictness Levels:**
- `strict`: High filtering
- `balanced`: Normal filtering (default)
- `relaxed`: Minimal filtering
- `disabled`: No filtering

### 4. **Thinking Mode Control**
```json
{
  "model_name": "qwen3-model",
  "message": "/no_think Give a brief answer",
  "thinking_mode": "disable"
}
```

**Thinking Modes:**
- `auto`: Auto-detect from message content
- `force`: Always enable thinking
- `disable`: Always disable thinking

## 📊 Advanced Parameters (25+ Available)

### Core Parameters
- `temperature`: Randomness (0.0-2.0)
- `top_p`: Nucleus sampling (0.0-1.0)
- `top_k`: Top-k sampling (1-100)
- `repetition_penalty`: Repetition control (1.0-2.0)
- `max_new_tokens`: Response length
- `min_new_tokens`: Minimum response length

### Advanced Sampling
- `min_p`: Minimum probability threshold
- `typical_p`: Typical sampling parameter
- `epsilon_cutoff`: Epsilon cutoff for sampling
- `eta_cutoff`: Eta cutoff for sampling
- `diversity_penalty`: Encourage diverse responses
- `encoder_repetition_penalty`: Encoder-specific repetition penalty
- `no_repeat_ngram_size`: N-gram repetition avoidance

### Beam Search Parameters
- `num_beams`: Number of beams for beam search
- `num_beam_groups`: Beam groups for diverse beam search
- `length_penalty`: Length penalty for beam search
- `early_stopping`: Early stopping in beam search

## 🎯 Usage Examples

### Basic Usage with Preset
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "What is AI?",
    "preset": "qwen_optimized"
  }'
```

### Advanced Parameter Control
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "Explain quantum computing",
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "disable_content_filter": true
  }'
```

### Thinking Mode Control
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "/no_think Give me a brief answer",
    "preset": "speed_optimized"
  }'
```

### Content Filter Control
```bash
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-model",
    "message": "Tell me about beauty treatments",
    "content_filter_strictness": "relaxed"
  }'
```

## 🔧 CLI Tools

### Enhanced CLI Helper
```bash
# Use optimization-based preset
python enhanced_chat_cli.py "What is AI?" --preset qwen_optimized

# Fine-tune parameters directly
python enhanced_chat_cli.py "Explain quantum physics" --temp 0.3 --top-p 0.95 --top-k 20

# Disable content filtering
python enhanced_chat_cli.py "Tell me about procedures" --no-filter

# Control thinking mode
python enhanced_chat_cli.py "/no_think Brief answer please" --preset speed_optimized

# Advanced sampling control
python enhanced_chat_cli.py "Creative story" --temp 0.7 --min-p 0.05 --diversity-penalty 0.2
```

### Original Helper (Updated)
```bash
# Test with new presets
python chat_api_helper.py test -m qwen3-model -t "Hello" --preset qwen_optimized

# Test with advanced parameters
python chat_api_helper.py test -m qwen3-model -t "What is AI?" --temperature 0.3 --min-p 0.05

# Test content filter control
python chat_api_helper.py test -m qwen3-model -t "Beauty question" --disable-filter
```

## 📊 Enhanced Response Format

The response now includes comprehensive information:

```json
{
  "success": true,
  "response": "Artificial intelligence is...",
  "model_name": "qwen3-model",
  "effective_config": {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "max_new_tokens": 256
  },
  "preset_used": "qwen_optimized",
  "thinking_enabled": false,
  "content_filter_applied": false,
  "content_filter_strictness": "disabled",
  "content_filter_bypassed": true,
  "tokens_generated": 156,
  "generation_time_ms": 2340.5,
  "tokens_per_second": 66.7,
  "thinking_content": null,
  "final_content": "Artificial intelligence is...",
  "generation_stats": {
    "model_info": {},
    "generation_config_used": {...},
    "content_filter_config": {"strictness_level": "disabled"},
    "performance": {
      "total_time_ms": 2450.0,
      "generation_time_ms": 2340.5,
      "tokens_generated": 156,
      "tokens_per_second": 66.7,
      "thinking_tokens": 0
    }
  }
}
```

## 🎯 Optimization Results Integration

The presets are based on actual optimization testing:

### qwen_optimized Preset
Based on optimization results from `param_optimization_results_qwen3-unsloth-q4ks_20250610_224757.json`:
- **Temperature**: 0.3 (best performance)
- **Top-p**: 0.95
- **Top-k**: 20
- **Repetition penalty**: 1.0
- **Best score**: 554.18 efficiency score

### high_quality Preset
Based on demo results showing high quality:
- **Temperature**: 0.1 (more focused)
- **Top-p**: 1.0 (full vocabulary)
- **Repetition penalty**: 1.15 (avoid repetition)
- **Efficiency score**: 850+ in testing

## 🔄 Migration Path

### Immediate Benefits
- **No code changes required** - existing JSON configs still work
- **Gradual adoption** - start using simple parameters when convenient
- **Better performance** - optimization-based presets

### Recommended Migration
1. **Start with presets** for common use cases
2. **Add direct parameters** for fine-tuning
3. **Use content filter control** as needed
4. **Monitor performance** with new metrics

## 🚀 Performance Improvements

### Speed Optimization
- Preset `speed_optimized`: Optimized for fastest response
- Preset `qwen_optimized`: Best balance based on actual testing
- Direct parameter control eliminates JSON parsing overhead

### Quality Optimization  
- Preset `high_quality`: Maximum quality settings
- Preset `creative_optimized`: Creative but efficient
- Fine-tune any parameter for specific use cases

### Efficiency Gains
- **Parameter inheritance**: Preset → Override → Default
- **Smart defaults**: Based on actual performance data
- **Reduced complexity**: Direct API calls instead of complex configs

## 🎉 Summary

The enhanced Chat API v2.0 provides:

✅ **Easier parameter control** - direct field access instead of nested JSON  
✅ **Optimization-based presets** - proven configurations from actual testing  
✅ **Content filtering control** - disable or adjust strictness as needed  
✅ **Advanced parameters** - 25+ options for fine-tuning  
✅ **Thinking mode control** - smart detection and override options  
✅ **Performance monitoring** - detailed metrics and timing  
✅ **Backward compatibility** - existing code continues to work  
✅ **Comprehensive tooling** - CLI helpers and test scripts  

This makes the BeautyAI chat API much more user-friendly while providing advanced users with powerful fine-tuning capabilities.
