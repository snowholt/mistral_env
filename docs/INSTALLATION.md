# Installation Guide

Complete installation instructions for the BeautyAI Inference Framework.

## 🔧 Quick Installation

### Backend Setup (Required)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup (Optional)
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Automated Setup
```bash
# Backend automated setup
cd backend
chmod +x unitTests_scripts/shell_scripts/setup_beautyai.sh
./unitTests_scripts/shell_scripts/setup_beautyai.sh
```

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 4090 recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, Windows
- **Python**: 3.11+ (3.12 recommended)
- **CUDA**: 11.8+ or 12.x for GPU acceleration
- **Node.js**: Not required (frontend is Python-based)

### Network Requirements
- Internet access for model downloads
- Bandwidth for Edge TTS voice synthesis

## 🔐 Authentication Setup

### Hugging Face Authentication
```bash
# Required for most models
huggingface-cli login
```
Get your token from: https://huggingface.co/settings/tokens

### GPU Setup Verification
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Production Installation

### System Services Setup
```bash
# Backend API service
cd backend/unitTests_scripts/shell_scripts
./manage-api-service.sh install
./manage-api-service.sh start

# Frontend web UI service
sudo cp beautyai-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable beautyai-webui
sudo systemctl start beautyai-webui
```

### Nginx Configuration (Optional)
```bash
# Copy nginx configuration
sudo cp nginx-clean-config.conf /etc/nginx/sites-available/beautyai
sudo ln -s /etc/nginx/sites-available/beautyai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 🐛 Installation Troubleshooting

### Common Issues

**CUDA Not Available**:
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

**Python Version Issues**:
```bash
# Install Python 3.11+ on Ubuntu
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Memory Issues**:
```bash
# Check available memory
free -h

# Reduce model size with quantization
beautyai model update default --quantization 4bit
```

## 🔄 Development Installation

### Backend Development
```bash
cd backend
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Frontend Development
```bash
cd frontend
pip install -r requirements.txt

# For development with auto-reload
export FLASK_ENV=development
```

### Testing Setup
```bash
# Backend tests
cd backend
python -m pytest unitTests_scripts/ -v

# Frontend tests
cd frontend
python -m pytest tests/ -v
```

---

**Next**: [Configuration Guide](CONFIGURATION.md) | [API Documentation](API.md)
