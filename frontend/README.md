# BeautyAI Frontend - Voice UI

This directory contains the web-based user interface for the BeautyAI Inference Framework, featuring advanced voice-to-voice conversation capabilities.

## Features

- 🎤 **Real-time Voice Chat**: WebSocket-based voice conversation
- 🌍 **Multi-language Support**: Arabic and English voice processing
- 🎨 **Modern Web UI**: Responsive design with modular components
- 🔧 **Multiple Modes**: Simple and advanced voice processing options
- 📱 **Cross-platform**: Works on desktop and mobile browsers

## Getting Started

### Prerequisites
- Python 3.11+ (for Flask backend)
- Modern web browser with WebRTC support

### Installation

```bash
# Create virtual environment for frontend
python3 -m venv frontend_env
source frontend_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the web UI server
python src/app.py
```

## Project Structure

```
frontend/
├── src/
│   ├── app.py              # Flask web server
│   ├── config.json         # Configuration settings
│   ├── templates/          # HTML templates
│   │   ├── index.html      # Main UI
│   │   ├── index_modular.html  # Modular UI
│   │   └── components/     # Reusable components
│   └── static/
│       ├── css/           # Stylesheets
│       ├── js/            # JavaScript files
│       └── test_websocket_wss.html  # WebSocket tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies  
└── package.json          # Project metadata
```

## Backend Integration

The frontend communicates with the backend API:
- **REST API**: `http://localhost:8000`
- **Simple Voice WebSocket**: `ws://localhost:8000/ws/simple-voice-chat`
- **Advanced Voice WebSocket**: `ws://localhost:8000/ws/voice-conversation`
- **API Documentation**: `http://localhost:8000/docs`

## Usage

1. Start the backend API server (from backend directory)
2. Start the frontend web UI (from frontend directory)  
3. Open browser to `http://localhost:5000`
4. Click microphone button to start voice conversation

## Documentation

Detailed documentation is available in the `docs/` directory:
- WebSocket implementation guides
- Voice processing workflows
- SSL/HTTPS configuration
- Troubleshooting guides
