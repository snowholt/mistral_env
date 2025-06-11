#!/usr/bin/env python3
"""
Tokenizer debugging script for GGUF models.

This script helps debug the tokenizer loading issue with unsloth GGUF models
and test potential fixes.
"""
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_model_files(model_id: str):
    """Find model files and tokenizer files for debugging."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_id_safe = model_id.replace("/", "--")
    
    logger.info(f"🔍 Looking for model: {model_id}")
    logger.info(f"Cache directory: {cache_dir}")
    
    # Find model directory
    model_dirs = list(cache_dir.glob(f"models--{model_id_safe}"))
    if not model_dirs:
        logger.error(f"❌ No cache directory found for {model_id}")
        return None
    
    model_dir = model_dirs[0]
    logger.info(f"📁 Model directory: {model_dir}")
    
    # Find snapshot directories
    snapshot_dirs = list(model_dir.glob("snapshots/*"))
    if not snapshot_dirs:
        logger.error(f"❌ No snapshots found in {model_dir}")
        return None
    
    snapshot_dir = snapshot_dirs[0]  # Use latest snapshot
    logger.info(f"📸 Snapshot directory: {snapshot_dir}")
    
    # Find GGUF files
    gguf_files = list(snapshot_dir.glob("*.gguf"))
    logger.info(f"🔍 Found {len(gguf_files)} GGUF files:")
    for gguf_file in gguf_files:
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {gguf_file.name} ({size_mb:.1f} MB)")
    
    # Find tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    logger.info(f"🔤 Tokenizer files:")
    found_tokenizer_files = {}
    for tokenizer_file in tokenizer_files:
        file_path = snapshot_dir / tokenizer_file
        if file_path.exists():
            logger.info(f"  ✅ {tokenizer_file}")
            found_tokenizer_files[tokenizer_file] = file_path
        else:
            logger.info(f"  ❌ {tokenizer_file}")
    
    return {
        'model_dir': model_dir,
        'snapshot_dir': snapshot_dir,
        'gguf_files': gguf_files,
        'tokenizer_files': found_tokenizer_files
    }


def test_llamacpp_with_tokenizer(model_id: str):
    """Test llama.cpp loading with explicit tokenizer path."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("❌ llama-cpp-python not installed")
        return False
    
    # Find model files
    model_info = find_model_files(model_id)
    if not model_info:
        return False
    
    # Get the Q4_K_S model file (smallest)
    gguf_files = model_info['gguf_files']
    model_file = None
    
    # Prefer Q4_K_S for speed
    for gguf_file in gguf_files:
        if "Q4_K_S" in gguf_file.name:
            model_file = gguf_file
            break
    
    if not model_file:
        model_file = gguf_files[0]  # Use first available
    
    logger.info(f"🎯 Using model file: {model_file}")
    
    # Test 1: Load without tokenizer path (current failing method)
    logger.info("\n🧪 Test 1: Loading without explicit tokenizer (current method)")
    try:
        model1 = Llama(
            model_path=str(model_file),
            n_gpu_layers=0,  # CPU only for testing
            n_ctx=512,       # Small context for testing
            verbose=False
        )
        logger.info("✅ Test 1 passed: Model loaded without explicit tokenizer")
        model1 = None  # Clean up
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
    
    # Test 2: Try to provide tokenizer directory
    if 'tokenizer.json' in model_info['tokenizer_files']:
        logger.info("\n🧪 Test 2: Trying with tokenizer directory hint")
        tokenizer_dir = model_info['snapshot_dir']
        
        # Unfortunately, llama-cpp-python doesn't have a direct tokenizer_path parameter
        # But we can try setting the working directory or using environment variables
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tokenizer_dir)
            model2 = Llama(
                model_path=str(model_file),
                n_gpu_layers=0,
                n_ctx=512,
                verbose=False
            )
            logger.info("✅ Test 2 passed: Model loaded with tokenizer directory as cwd")
            model2 = None
        except Exception as e:
            logger.error(f"❌ Test 2 failed: {e}")
        finally:
            os.chdir(original_cwd)
    
    return True


def test_transformers_fallback(model_id: str):
    """Test if we can use transformers as a fallback for GGUF models."""
    logger.info(f"\n🔄 Testing transformers fallback for {model_id}")
    
    # For unsloth GGUF models, try the original model
    if "unsloth" in model_id and "GGUF" in model_id:
        original_model_id = model_id.replace("-GGUF", "").replace("unsloth/", "Qwen/")
        logger.info(f"🎯 Fallback to original model: {original_model_id}")
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(original_model_id)
            logger.info(f"✅ Tokenizer loaded successfully from {original_model_id}")
            return original_model_id
        except Exception as e:
            logger.error(f"❌ Fallback tokenizer failed: {e}")
    
    return None


def main():
    """Main debugging function."""
    logger.info("🔧 GGUF Tokenizer Debugging Tool")
    logger.info("=" * 50)
    
    # Test the problematic model
    problematic_model = "unsloth/Qwen3-14B-GGUF"
    
    # Step 1: Inspect model files
    logger.info("📋 Step 1: Inspecting model files")
    model_info = find_model_files(problematic_model)
    
    if not model_info:
        logger.error("❌ Cannot proceed without model files")
        return
    
    # Step 2: Test llamacpp loading strategies
    logger.info("\n📋 Step 2: Testing llamacpp loading strategies")
    test_llamacpp_with_tokenizer(problematic_model)
    
    # Step 3: Test transformers fallback
    logger.info("\n📋 Step 3: Testing transformers fallback")
    fallback_model = test_transformers_fallback(problematic_model)
    
    # Step 4: Recommendations
    logger.info("\n💡 Recommendations:")
    logger.info("1. Use explicit tokenizer loading in llamacpp engine")
    logger.info("2. Implement graceful fallback to transformers for GGUF models with tokenizer issues")
    logger.info("3. Update model configuration to specify tokenizer source")
    
    if fallback_model:
        logger.info(f"4. For {problematic_model}, use {fallback_model} as tokenizer source")


if __name__ == "__main__":
    main()
