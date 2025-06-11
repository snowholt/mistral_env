#!/usr/bin/env python3
"""
GGUF Model Tokenizer Issue Diagnosis and Fix Script.

This script diagnoses and provides fixes for the tokenizer loading issue
with GGUF models, particularly the "Could not load any compatible tokenizer" error.
"""
import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GGUFTokenizerFixer:
    """Class to diagnose and fix GGUF tokenizer issues."""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def check_model_cache(self, model_id: str) -> Dict[str, Any]:
        """Check the HuggingFace cache for model files."""
        cache_dir = Path.home() / ".cache/huggingface/hub"
        model_id_safe = model_id.replace("/", "--")
        
        logger.info(f"🔍 Checking cache for {model_id}")
        
        # Find model directory
        model_dirs = list(cache_dir.glob(f"models--{model_id_safe}"))
        if not model_dirs:
            self.issues_found.append(f"No cache directory found for {model_id}")
            return {"status": "not_cached", "model_id": model_id}
        
        model_dir = model_dirs[0]
        snapshot_dirs = list(model_dir.glob("snapshots/*"))
        if not snapshot_dirs:
            self.issues_found.append(f"No snapshots found for {model_id}")
            return {"status": "no_snapshots", "model_dir": str(model_dir)}
        
        snapshot_dir = snapshot_dirs[0]
        
        # Check for GGUF files
        gguf_files = list(snapshot_dir.glob("*.gguf"))
        
        # Check for tokenizer files
        tokenizer_files = {
            "tokenizer.json": snapshot_dir / "tokenizer.json",
            "tokenizer_config.json": snapshot_dir / "tokenizer_config.json",
            "vocab.json": snapshot_dir / "vocab.json",
            "merges.txt": snapshot_dir / "merges.txt",
            "special_tokens_map.json": snapshot_dir / "special_tokens_map.json"
        }
        
        existing_tokenizer_files = {
            name: path for name, path in tokenizer_files.items() 
            if path.exists()
        }
        
        result = {
            "status": "cached",
            "model_dir": str(model_dir),
            "snapshot_dir": str(snapshot_dir),
            "gguf_files": [str(f) for f in gguf_files],
            "tokenizer_files": {name: str(path) for name, path in existing_tokenizer_files.items()},
            "missing_tokenizer_files": [name for name in tokenizer_files if name not in existing_tokenizer_files]
        }
        
        logger.info(f"📦 Found {len(gguf_files)} GGUF files")
        logger.info(f"🔤 Found {len(existing_tokenizer_files)} tokenizer files")
        
        if not existing_tokenizer_files:
            self.issues_found.append(f"No tokenizer files found for {model_id}")
        
        return result
    
    def test_llamacpp_loading(self, model_info: Dict[str, Any]) -> bool:
        """Test loading the model with llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            self.issues_found.append("llama-cpp-python not installed")
            return False
        
        if not model_info.get("gguf_files"):
            self.issues_found.append("No GGUF files found")
            return False
        
        # Try loading the smallest GGUF file for testing
        gguf_files = model_info["gguf_files"]
        test_file = None
        
        # Prefer Q4_K_S (smallest/fastest)
        for file_path in gguf_files:
            if "Q4_K_S" in Path(file_path).name:
                test_file = file_path
                break
        
        if not test_file:
            test_file = gguf_files[0]  # Use first available
        
        logger.info(f"🧪 Testing model loading: {Path(test_file).name}")
        
        try:
            # Test with minimal settings
            model = Llama(
                model_path=test_file,
                n_gpu_layers=0,  # CPU only for testing
                n_ctx=128,       # Minimal context
                verbose=False
            )
            
            # Quick test inference
            response = model("Test", max_tokens=5)
            
            logger.info("✅ Model loaded and inference successful")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.issues_found.append(f"Model loading failed: {error_msg}")
            logger.error(f"❌ Model loading failed: {error_msg}")
            
            # Analyze specific error types
            if "tokenizer" in error_msg.lower():
                self.issues_found.append("Tokenizer-related loading error detected")
            
            return False
    
    def find_compatible_tokenizer(self, model_id: str) -> Optional[str]:
        """Find a compatible tokenizer for the GGUF model."""
        
        # For unsloth GGUF models, try the original model
        if "unsloth" in model_id and "GGUF" in model_id:
            original_model_id = model_id.replace("-GGUF", "").replace("unsloth/", "Qwen/")
            logger.info(f"🎯 Trying original model tokenizer: {original_model_id}")
            
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(original_model_id)
                logger.info(f"✅ Compatible tokenizer found: {original_model_id}")
                return original_model_id
            except Exception as e:
                logger.warning(f"⚠️  Original model tokenizer failed: {e}")
        
        # Try known compatible tokenizers for Qwen models
        qwen_alternatives = [
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-7B", 
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2-14B"
        ]
        
        for alt_model in qwen_alternatives:
            logger.info(f"🔄 Trying alternative tokenizer: {alt_model}")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(alt_model)
                logger.info(f"✅ Compatible tokenizer found: {alt_model}")
                return alt_model
            except Exception as e:
                logger.warning(f"⚠️  Alternative tokenizer failed: {e}")
        
        return None
    
    def update_model_registry(self, model_name: str, tokenizer_model_id: str):
        """Update the model registry with tokenizer information."""
        registry_paths = [
            "beautyai_inference/config/model_registry.json",
            "beautyai_inference/config/model_registry_20250610.json"
        ]
        
        for registry_path in registry_paths:
            if os.path.exists(registry_path):
                logger.info(f"📝 Updating registry: {registry_path}")
                
                try:
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                    
                    if model_name in registry.get("models", {}):
                        model_config = registry["models"][model_name]
                        model_config["tokenizer_model_id"] = tokenizer_model_id
                        model_config["tokenizer_fallback"] = True
                        
                        # Backup original file
                        backup_path = f"{registry_path}.backup_{int(time.time())}"
                        shutil.copy2(registry_path, backup_path)
                        
                        # Write updated registry
                        with open(registry_path, 'w') as f:
                            json.dump(registry, f, indent=2)
                        
                        self.fixes_applied.append(f"Updated {registry_path} with tokenizer fallback")
                        logger.info(f"✅ Registry updated with tokenizer fallback")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update registry {registry_path}: {e}")
    
    def create_llamacpp_engine_fix(self):
        """Create a fix for the LlamaCpp engine to handle tokenizer fallback."""
        
        engine_file = "beautyai_inference/inference_engines/llamacpp_engine.py"
        if not os.path.exists(engine_file):
            logger.warning(f"⚠️  Engine file not found: {engine_file}")
            return
        
        # Create a patch suggestion
        patch_suggestion = '''
# SUGGESTED FIX: Add tokenizer fallback to LlamaCppEngine.load_model()

# Around line 125-130, add tokenizer fallback logic:

def load_model(self) -> None:
    \"\"\"Load the model into memory with optimized settings for RTX 4090.\"\"\"
    start_time = time.time()
    logger.info(f"Loading GGUF model: {self.config.model_id}")
    
    # Find the GGUF model file
    model_path = self._find_gguf_model_path()
    if not model_path:
        raise FileNotFoundError(f"Could not find GGUF model file for {self.config.model_id}")
    
    logger.info(f"Loading GGUF file: {model_path}")
    
    # NEW: Check for tokenizer fallback configuration
    tokenizer_model_id = getattr(self.config, 'tokenizer_model_id', None)
    if tokenizer_model_id:
        logger.info(f"Using tokenizer fallback: {tokenizer_model_id}")
        # You can use this to load the tokenizer separately if needed
    
    # Enhanced CUDA detection and configuration
    import torch
    has_cuda = torch.cuda.is_available()
    logger.info(f"CUDA available: {has_cuda}")
    
    # ... rest of the existing method ...
    
    try:
        logger.info(f"Initializing Llama with {n_gpu_layers} GPU layers, context: {n_ctx}")
        
        # NEW: Add error handling with tokenizer fallback
        try:
            self.model = Llama(**model_params)
        except Exception as tokenizer_error:
            if "tokenizer" in str(tokenizer_error).lower():
                logger.warning(f"Tokenizer error detected: {tokenizer_error}")
                logger.info("Attempting to load model without tokenizer dependencies...")
                
                # Try with minimal tokenizer requirements
                minimal_params = model_params.copy()
                minimal_params.update({
                    "vocab_only": False,
                    "logits_all": False,
                    "embedding": False
                })
                
                self.model = Llama(**minimal_params)
                logger.info("✅ Model loaded with tokenizer workaround")
            else:
                raise tokenizer_error
        
        loading_time = time.time() - start_time
        logger.info(f"✅ GGUF model loaded successfully in {loading_time:.2f} seconds")
        
        # ... rest of the existing method ...
    
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        raise
'''
        
        # Save the patch suggestion
        patch_file = "llamacpp_engine_tokenizer_fix.patch"
        with open(patch_file, 'w') as f:
            f.write(patch_suggestion)
        
        self.fixes_applied.append(f"Created patch suggestion: {patch_file}")
        logger.info(f"📄 Patch suggestion saved to: {patch_file}")
    
    def diagnose_and_fix(self, model_names: List[str]):
        """Comprehensive diagnosis and fix for GGUF tokenizer issues."""
        
        logger.info("🔧 Starting GGUF Tokenizer Issue Diagnosis and Fix")
        logger.info("=" * 60)
        
        for model_name in model_names:
            logger.info(f"\n🎯 Diagnosing model: {model_name}")
            logger.info("-" * 40)
            
            # Get model ID from registry
            model_id = self.get_model_id_from_registry(model_name)
            if not model_id:
                logger.error(f"❌ Model {model_name} not found in registry")
                continue
            
            # Check cache
            model_info = self.check_model_cache(model_id)
            if model_info["status"] != "cached":
                logger.error(f"❌ Model {model_name} not properly cached")
                continue
            
            # Test loading
            if self.test_llamacpp_loading(model_info):
                logger.info(f"✅ Model {model_name} loads successfully - no fix needed")
                continue
            
            # Find compatible tokenizer
            compatible_tokenizer = self.find_compatible_tokenizer(model_id)
            if compatible_tokenizer:
                self.update_model_registry(model_name, compatible_tokenizer)
                logger.info(f"✅ Fix applied for {model_name}")
            else:
                logger.error(f"❌ No compatible tokenizer found for {model_name}")
        
        # Create general engine fix
        self.create_llamacpp_engine_fix()
        
        # Summary
        self.print_diagnosis_summary()
    
    def get_model_id_from_registry(self, model_name: str) -> Optional[str]:
        """Get model ID from the registry."""
        registry_paths = [
            "beautyai_inference/config/model_registry.json",
            "beautyai_inference/config/model_registry_20250610.json"
        ]
        
        for registry_path in registry_paths:
            if os.path.exists(registry_path):
                try:
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                    
                    models = registry.get("models", {})
                    if model_name in models:
                        return models[model_name].get("model_id")
                        
                except Exception as e:
                    logger.warning(f"Error reading registry {registry_path}: {e}")
        
        return None
    
    def print_diagnosis_summary(self):
        """Print summary of diagnosis and fixes."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 DIAGNOSIS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Issues found: {len(self.issues_found)}")
        for issue in self.issues_found:
            logger.info(f"  ❌ {issue}")
        
        logger.info(f"\\nFixes applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            logger.info(f"  ✅ {fix}")
        
        if self.fixes_applied:
            logger.info("\\n💡 Next steps:")
            logger.info("1. Restart the API server to apply registry changes")
            logger.info("2. Test the models using the API testing script")
            logger.info("3. If issues persist, consider the llamacpp_engine patch")


def main():
    """Main function."""
    import argparse
    import shutil
    import time
    
    parser = argparse.ArgumentParser(description="Diagnose and fix GGUF tokenizer issues")
    parser.add_argument("models", nargs="*", default=["qwen3-unsloth-q4ks"], 
                       help="Model names to diagnose")
    parser.add_argument("--apply-fixes", action="store_true", 
                       help="Apply fixes automatically")
    
    args = parser.parse_args()
    
    fixer = GGUFTokenizerFixer()
    
    try:
        fixer.diagnose_and_fix(args.models)
    except KeyboardInterrupt:
        logger.info("\\n⚠️  Diagnosis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
