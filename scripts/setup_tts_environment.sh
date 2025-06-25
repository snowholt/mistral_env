#!/bin/bash
# Setup TTS Environment for BeautyAI
# This script creates a Python 3.11 environment for TTS compatibility

echo "🎙️ BeautyAI TTS Environment Setup"
echo "=================================="

# Check if pyenv is available
if command -v pyenv &> /dev/null; then
    echo "✅ pyenv found"
    
    # Install Python 3.11 if not available
    if ! pyenv versions | grep -q "3.11"; then
        echo "📥 Installing Python 3.11 via pyenv..."
        pyenv install 3.11.9
    fi
    
    # Create virtual environment with Python 3.11
    echo "🔧 Creating Python 3.11 virtual environment..."
    pyenv virtualenv 3.11.9 beautyai-tts
    echo "✅ Virtual environment created: beautyai-tts"
    
    echo ""
    echo "🚀 To activate the TTS environment:"
    echo "   pyenv activate beautyai-tts"
    echo "   pip install TTS"
    echo ""
    echo "🔄 To switch back to main environment:"
    echo "   pyenv deactivate"
    
elif command -v conda &> /dev/null; then
    echo "✅ conda found"
    
    # Create conda environment with Python 3.11
    echo "🔧 Creating Python 3.11 conda environment..."
    conda create -n beautyai-tts python=3.11 -y
    echo "✅ Conda environment created: beautyai-tts"
    
    echo ""
    echo "🚀 To activate the TTS environment:"
    echo "   conda activate beautyai-tts"
    echo "   pip install TTS"
    echo ""
    echo "🔄 To switch back:"
    echo "   conda deactivate"
    
else
    echo "❌ Neither pyenv nor conda found"
    echo ""
    echo "🛠️ Install options:"
    echo "1. Install pyenv:"
    echo "   curl https://pyenv.run | bash"
    echo ""
    echo "2. Install Miniconda:"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "3. Use system package manager:"
    echo "   sudo apt install python3.11 python3.11-venv  # Ubuntu/Debian"
    echo "   python3.11 -m venv beautyai-tts"
    echo "   source beautyai-tts/bin/activate"
    echo "   pip install TTS"
fi

echo ""
echo "📋 After setting up Python 3.11 environment:"
echo "1. Activate the environment"
echo "2. pip install TTS"
echo "3. pip install torch torchaudio"
echo "4. Test: python -c \"from TTS.api import TTS; print('TTS installed successfully!')\""
