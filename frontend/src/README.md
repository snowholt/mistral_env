# BeautyAI Assistant Web UI 🌟

A beautiful, user-friendly web interface for chatting with BeautyAI models featuring:

- **🎨 Stunning animated fractal background** - Mathematical beauty in motion
- **⚡ Real-time chat interface** - Smooth, responsive conversations
- **🎛️ Advanced parameter controls** - Fine-tune model behavior
- **🔧 Optimization presets** - Pre-configured settings for different use cases
- **🧠 Thinking mode visualization** - See the model's reasoning process
- **🛡️ Content filtering controls** - Adjust safety settings
- **📊 Performance metrics** - Token generation stats and timing

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install flask aiohttp flask-cors requests
   ```

2. **Start BeautyAI API** (required):
   ```bash
   # In your main BeautyAI project directory
   python -m beautyai_inference.api.app
   ```

3. **Launch the Web UI:**
   ```bash
   cd src/web_ui
   python start_ui.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Features

### 🎨 Beautiful Design
- Animated 3D fractal particle background
- Glassmorphism UI elements
- Smooth animations and transitions
- Responsive design for mobile and desktop

### 🤖 Advanced Chat Controls
- **Model Selection**: Choose from available BeautyAI models
- **Optimization Presets**: Quick settings for different scenarios
  - `qwen_optimized`: Best settings from testing (temp=0.3, top_p=0.95)
  - `high_quality`: Maximum quality output
  - `creative_optimized`: Creative but efficient
  - `speed_optimized`: Fast responses
  - `balanced`: Good all-around performance
  - `conservative`: Safe, predictable outputs

### 🎛️ Parameter Control
**Core Parameters:**
- Temperature (0-2): Controls randomness
- Top P (0-1): Nucleus sampling threshold
- Top K (1-100): Consider top K tokens
- Max Tokens (50-4000): Response length limit

**Advanced Parameters:**
- Repetition Penalty: Avoid repetitive text
- Min P: Minimum probability threshold
- Content Filter Strictness: Adjust safety levels
- Thinking Mode: Enable/disable reasoning display

### 🧠 Thinking Mode
- See the model's internal reasoning process
- Toggle thinking mode on/off
- Use `/no_think` command for quick responses

### 🛡️ Content Filtering
- Adjustable strictness levels: Strict, Balanced, Relaxed, Disabled
- Override content filters when needed
- Real-time filter status display

### 📊 Performance Metrics
- Token generation count
- Generation time in milliseconds
- Tokens per second rate
- Model configuration used

## API Integration

The web UI connects to BeautyAI's `/inference/chat` endpoint with full parameter support:

```json
{
  "model_name": "qwen3-model",
  "message": "What is artificial intelligence?",
  "preset": "qwen_optimized",
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 20,
  "max_new_tokens": 2048,
  "enable_thinking": true,
  "content_filter_strictness": "balanced"
}
```

## Troubleshooting

**🔌 Connection Issues:**
- Ensure BeautyAI API is running on localhost:8000
- Check API health: `curl http://localhost:8000/health`

**🤖 No Models Available:**
- Load models in BeautyAI: `beautyai model load qwen3-model`
- Check models: `curl http://localhost:8000/models`

**🐛 Web UI Errors:**
- Check browser console for JavaScript errors
- Verify Flask dependencies are installed
- Check terminal for Python errors

## Development

**File Structure:**
```
web_ui/
├── app.py              # Flask application
├── start_ui.py         # Launcher script
├── templates/
│   └── index.html      # Main UI template
└── README.md           # This file
```

**Customization:**
- Edit `templates/index.html` to modify the UI
- Adjust `app.py` to add new API endpoints
- Modify fractal background parameters in JavaScript

## Requirements

- Python 3.7+
- Flask 2.0+
- aiohttp 3.8+
- BeautyAI API running on localhost:8000

## License

Part of the BeautyAI independent benchmarking and testing system.
