# 🚀 BeautyAI Optimized Model Registry Summary

## 📊 **Performance Benchmark Results** (2025-06-10)

### 🏆 **Speed Champions** (Tokens/Second)
1. **qwen3-unsloth-q4ks**: 65.5 tok/s ⚡ (Fastest)
2. **qwen3-official-q4km**: 64.3 tok/s  
3. **qwen3-official-q8**: 64.1 tok/s  
4. **qwen3-unsloth-q4km**: 62.6 tok/s
5. **bee1reason-arabic-q4ks**: 49.5 tok/s
6. **bee1reason-arabic-q4km**: 48.0 tok/s
7. **bee1reason-arabic-q6k**: 47.4 tok/s (estimated)
8. **qwen3-model** (transformers): 25.2 tok/s
9. **deepseek-r1-multilingual**: 21.3 tok/s

## 🎯 **Recommended Model Selection**

### **For Maximum Speed**
- **Primary**: `qwen3-unsloth-q4ks` (65.5 tok/s)
- **Backup**: `qwen3-official-q4km` (64.3 tok/s)

### **For Best Quality-Speed Balance**
- **Primary**: `qwen3-official-q8` (64.1 tok/s, best quality)
- **Backup**: `qwen3-unsloth-q4km` (62.6 tok/s)

### **For Arabic Content**
- **Primary**: `bee1reason-arabic-q4ks` (49.5 tok/s)
- **Quality**: `bee1reason-arabic-q6k` (best Arabic quality)

### **For Multilingual Tasks**
- **Primary**: `deepseek-r1-multilingual` (21.3 tok/s, specialized)

## 🔧 **Registry Optimizations Made**

### **1. Engine Type Separation**
- **Transformers**: GPU-optimized 4-bit quantization
- **LlamaCpp**: CPU/GPU GGUF quantizations

### **2. Performance Tiers**
- **Fast**: Q4_K_S quantizations (smallest, fastest)
- **Balanced**: Q4_K_M quantizations (good quality-speed ratio)
- **Quality**: Q8_0/Q6_K quantizations (best quality)

### **3. Model Categories**
- **Official Qwen3**: Direct from Qwen team
- **Unsloth Qwen3**: Community-optimized versions
- **Arabic Specialized**: Bee1reason fine-tuned for Arabic
- **Multilingual**: DeepSeek R1 distilled

### **4. Issues Resolved**
- **Disabled corrupted models**: `qwen3-unsloth-q8` (file corruption)
- **Fixed quantization availability**: Arabic model max Q6_K (no Q8_0)
- **Optimized filenames**: Correct GGUF file matching

## 📈 **Key Performance Insights**

### **Speed vs Engine**
- **LlamaCpp GGUF**: 2-3x faster than Transformers
- **Q4_K_S**: Fastest quantization with minimal quality loss
- **Q8_0**: Best quality with good speed (only 1% slower than Q4_K_M)

### **Memory Usage**
- **Q4_K_S**: ~8.7GB (fastest)
- **Q4_K_M**: ~9.1GB (balanced)
- **Q8_0**: ~15.8GB (quality)

### **Load Times**
- **LlamaCpp**: 1-6 seconds
- **Transformers**: 23-38 seconds

## 🎯 **Final Recommendations**

### **Production Setup**
1. **Default fast model**: `qwen3-unsloth-q4ks`
2. **Quality fallback**: `qwen3-official-q8`
3. **Arabic content**: `bee1reason-arabic-q4ks`
4. **Multilingual**: `deepseek-r1-multilingual`

### **Development Setup**
- Use Q4_K_S for quick iteration
- Switch to Q8_0 for final quality checks
- Keep transformers models for specific use cases

### **Resource Optimization**
- **8GB VRAM**: Use Q4_K_S quantizations
- **16GB+ VRAM**: Use Q8_0 for best quality
- **24GB+ VRAM**: Can run transformers models

## 📋 **Registry Status**
- ✅ **Total Models**: 10 configured
- ✅ **Working Models**: 9 (90% success rate)
- ❌ **Disabled Models**: 1 (corrupted file)
- 🎯 **Optimized for**: Speed, Quality, Specialization

---
*Last updated: 2025-06-10*
*Benchmark environment: RTX 4090, 24GB VRAM*
