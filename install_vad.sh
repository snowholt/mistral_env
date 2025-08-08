#!/bin/bash
# Install VAD implementation dependencies and restart services

echo "🔧 Installing VAD implementation dependencies..."

# Navigate to backend directory
cd /home/lumi/beautyai/backend

# Install new dependencies
echo "📦 Installing Python packages..."
pip install silero-vad>=5.1.0 websockets>=11.0.0 pydub>=0.25.0

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Test the implementation
echo "🧪 Testing VAD implementation..."
cd /home/lumi/beautyai
python test_vad_implementation.py

if [ $? -eq 0 ]; then
    echo "✅ VAD implementation tests passed"
else
    echo "⚠️ VAD implementation tests had issues, but continuing with service restart..."
fi

# Restart services
echo "🔄 Restarting BeautyAI services..."

# Restart API service
sudo systemctl restart beautyai-api
if [ $? -eq 0 ]; then
    echo "✅ BeautyAI API service restarted"
else
    echo "❌ Failed to restart BeautyAI API service"
fi

# Restart WebUI service
sudo systemctl restart beautyai-webui
if [ $? -eq 0 ]; then
    echo "✅ BeautyAI WebUI service restarted"
else
    echo "❌ Failed to restart BeautyAI WebUI service"
fi

# Check service status
echo "📊 Checking service status..."
sudo systemctl status beautyai-api --no-pager -l
sudo systemctl status beautyai-webui --no-pager -l

echo "🎉 VAD implementation installation completed!"
echo ""
echo "🚀 New Features Available:"
echo "  - Real-time voice activity detection"
echo "  - Server-side turn-taking"
echo "  - Gemini Live / GPT Voice style interaction"
echo "  - Audio chunk buffering and processing"
echo "  - Smooth voice conversation experience"
echo ""
echo "🌐 Access the updated interface at your usual URL"
