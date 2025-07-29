"""
Integration Engine for BeautyAI Framework.
Clean integration without deprecated TTS models.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class IntegrationEngine:
    """Clean integration engine for BeautyAI Framework."""
    
    def __init__(self):
        """Initialize the integration engine."""
        self.supported_engines = {
            "coqui_tts": "CoquiTTSEngine",
            "edge_tts": "EdgeTTSEngine",
            "transformers": "TransformersEngine",
            "llama.cpp": "LlamaCppEngine"
        }
        
        logger.info("✅ Integration engine initialized (clean architecture)")
    
    def get_supported_engines(self) -> List[str]:
        """Get list of supported engines."""
        return list(self.supported_engines.keys())
    
    def is_engine_supported(self, engine_type: str) -> bool:
        """Check if an engine type is supported."""
        return engine_type in self.supported_engines
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about supported engines."""
        return {
            "supported_engines": self.supported_engines,
            "total_engines": len(self.supported_engines),
            "tts_engines": ["coqui_tts", "edge_tts"],
            "llm_engines": ["transformers", "llama.cpp"],
            "deprecated_engines": []  # Clean - no deprecated engines
        }
