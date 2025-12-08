
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to path
sys.path.append("/home/lumi/beautyai/backend/src")
sys.path.append("/home/lumi/beautyai/backend")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_fix():
    logger.info("Starting verification...")
    
    # Check if directory exists
    cache_dir = Path("/home/lumi/beautyai/backend/logs/triton_cache")
    if cache_dir.exists():
        logger.info(f"✅ Cache directory exists: {cache_dir}")
        logger.info(f"Permissions: {oct(os.stat(cache_dir).st_mode)[-3:]}")
    else:
        logger.error(f"❌ Cache directory does not exist: {cache_dir}")
        return

    # Import engine
    try:
        from beautyai_inference.inference_engines.voice.stt.whisper_large_v3_turbo_engine import WhisperLargeV3TurboEngine
        logger.info("✅ Successfully imported WhisperLargeV3TurboEngine")
    except ImportError as e:
        logger.error(f"❌ Failed to import engine: {e}")
        return

    # Instantiate engine
    logger.info("Instantiating engine...")
    engine = WhisperLargeV3TurboEngine()
    
    # Check env var
    triton_env = os.environ.get("TRITON_CACHE_DIR")
    if triton_env == str(cache_dir):
        logger.info(f"✅ TRITON_CACHE_DIR is correctly set to: {triton_env}")
    else:
        logger.error(f"❌ TRITON_CACHE_DIR is NOT set correctly. Value: {triton_env}")

    # Try to load model (mocking if possible, or just checking init)
    # We won't fully load the model to save time/memory, but we can check if _warmup_model would run
    
    logger.info("Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify_fix())
