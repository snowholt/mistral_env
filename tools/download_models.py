#!/usr/bin/env python3
"""
Enhanced Model Download Utility for BeautyAI
Provides optimized downloading with speed improvements and progress tracking.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import json

def setup_environment():
    """Setup environment variables for optimized downloads."""
    optimizations = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_PARALLEL_DOWNLOADS": "4", 
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "TRANSFORMERS_TRUST_REMOTE_CODE": "1",
        "HF_ALLOW_CODE_EVAL": "1"
    }
    
    for key, value in optimizations.items():
        os.environ.setdefault(key, value)
        print(f"  {key}: {os.getenv(key)}")

def install_hf_transfer():
    """Install hf_transfer for faster downloads."""
    try:
        import hf_transfer
        print("✅ hf_transfer already installed")
        return True
    except ImportError:
        print("Installing hf_transfer for faster downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hf_transfer"])
            print("✅ hf_transfer installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install hf_transfer: {e}")
            return False

def get_model_cache_path(model_id: str) -> Path:
    """Get the cache path for a model."""
    cache_dir = os.getenv("HF_HOME")
    if cache_dir:
        base_path = Path(cache_dir)
    else:
        base_path = Path.home() / ".cache" / "huggingface"
    
    # Convert model ID to cache directory name
    cache_name = model_id.replace("/", "--")
    cache_name = f"models--{cache_name}"
    
    return base_path / "hub" / cache_name

def check_model_cached(model_id: str) -> Dict[str, Any]:
    """Check if model is already cached and get cache info."""
    cache_path = get_model_cache_path(model_id)
    
    if cache_path.exists():
        # Calculate cache size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except OSError:
                    pass
        
        return {
            "cached": True,
            "cache_path": str(cache_path),
            "size_bytes": total_size,
            "size_gb": total_size / (1024**3),
            "file_count": file_count
        }
    else:
        return {
            "cached": False,
            "cache_path": str(cache_path),
            "size_bytes": 0,
            "size_gb": 0,
            "file_count": 0
        }

def download_model(model_id: str, force_reload: bool = False) -> bool:
    """Download model with optimizations."""
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm
        
        print(f"\n🔄 Downloading model: {model_id}")
        
        # Check if already cached
        cache_info = check_model_cached(model_id)
        if cache_info["cached"] and not force_reload:
            print(f"✅ Model already cached ({cache_info['size_gb']:.2f} GB, {cache_info['file_count']} files)")
            print(f"   Cache location: {cache_info['cache_path']}")
            return True
        
        start_time = time.time()
        
        # Download with progress tracking
        print("⏳ Starting download with optimizations...")
        
        try:
            result = snapshot_download(
                model_id,
                cache_dir=None,  # Use default cache
                resume_download=True,
                local_files_only=False,
                use_auth_token=True if os.getenv("HUGGING_FACE_HUB_TOKEN") else None,
                ignore_patterns=["*.gguf"] if "gguf" not in model_id.lower() else None
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            
            # Get final cache info
            final_cache_info = check_model_cached(model_id)
            download_speed_mbps = (final_cache_info["size_bytes"] / (1024**2)) / download_time if download_time > 0 else 0
            
            print(f"✅ Download completed successfully!")
            print(f"   Time: {download_time:.2f} seconds")
            print(f"   Size: {final_cache_info['size_gb']:.2f} GB ({final_cache_info['file_count']} files)")
            print(f"   Speed: {download_speed_mbps:.2f} MB/s")
            print(f"   Location: {result}")
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Required libraries not available: {e}")
        return False

def load_model_registry(registry_path: str = "beautyai_inference/config/model_registry.json") -> Dict:
    """Load the model registry."""
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Model registry not found at {registry_path}")
        return {}

def list_available_models(registry_path: str = "beautyai_inference/config/model_registry.json"):
    """List all models in the registry."""
    registry = load_model_registry(registry_path)
    models = registry.get("models", {})
    
    print("\n📚 Available models in registry:")
    print("-" * 80)
    for name, config in models.items():
        model_id = config.get("model_id", "Unknown")
        engine = config.get("engine_type", "Unknown")
        description = config.get("description", "No description")
        
        # Check cache status
        cache_info = check_model_cached(model_id)
        cache_status = f"✅ Cached ({cache_info['size_gb']:.1f}GB)" if cache_info["cached"] else "❌ Not cached"
        
        print(f"  {name}")
        print(f"    Model ID: {model_id}")
        print(f"    Engine: {engine}")
        print(f"    Description: {description}")
        print(f"    Status: {cache_status}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Model Download Utility for BeautyAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a specific model
  python download_models.py --model Paula139/DeepSeek-R1-destill-llama3-8b-arabic-fine-tuned
  
  # Download model by registry name
  python download_models.py --name arabic-deepseek-r1-distill-llama3-8b
  
  # List all available models
  python download_models.py --list
  
  # Check cache status
  python download_models.py --check-cache --name qwen3-model
        """
    )
    
    parser.add_argument("--model", "-m", help="Model ID to download (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--name", "-n", help="Model name from registry")
    parser.add_argument("--list", "-l", action="store_true", help="List all available models")
    parser.add_argument("--check-cache", "-c", action="store_true", help="Check cache status only")
    parser.add_argument("--force-reload", "-f", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--registry", default="beautyai_inference/config/model_registry.json", 
                        help="Path to model registry file")
    
    args = parser.parse_args()
    
    print("=== BeautyAI Enhanced Model Download Utility ===")
    
    # Setup optimizations
    print("\n🔧 Setting up download optimizations...")
    setup_environment()
    
    # Install hf_transfer
    print("\n📦 Checking dependencies...")
    install_hf_transfer()
    
    if args.list:
        list_available_models(args.registry)
        return 0
    
    # Determine model ID
    model_id = None
    if args.model:
        model_id = args.model
    elif args.name:
        registry = load_model_registry(args.registry)
        models = registry.get("models", {})
        if args.name in models:
            model_id = models[args.name].get("model_id")
        else:
            print(f"❌ Model '{args.name}' not found in registry")
            return 1
    
    if not model_id:
        print("❌ Please specify --model or --name")
        parser.print_help()
        return 1
    
    # Check cache status
    cache_info = check_model_cached(model_id)
    print(f"\n📊 Cache status for {model_id}:")
    if cache_info["cached"]:
        print(f"  ✅ Cached: {cache_info['size_gb']:.2f} GB ({cache_info['file_count']} files)")
        print(f"  📁 Location: {cache_info['cache_path']}")
    else:
        print(f"  ❌ Not cached")
    
    if args.check_cache:
        return 0
    
    # Download if needed
    if not cache_info["cached"] or args.force_reload:
        success = download_model(model_id, args.force_reload)
        return 0 if success else 1
    else:
        print("✅ Model already cached. Use --force-reload to re-download.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
