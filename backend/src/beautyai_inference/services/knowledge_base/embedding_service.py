"""
Embedding service for generating text embeddings.

Uses sentence-transformers for multilingual embeddings that work well
with Arabic and English text.
"""

import os
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Uses a multilingual model that supports Arabic and English.
    The embeddings are 384-dimensional by default (with MiniLM).
    For production, consider larger models like multilingual-e5-large (1024-dim).
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        self._model = None
        self._model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self._device = os.getenv("EMBEDDING_DEVICE", "cuda")
        self._batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    async def _ensure_model_loaded(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            
            # Verify embedding dimension
            test_embedding = self._model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            if actual_dim != self._embedding_dim:
                logger.warning(
                    f"Model embedding dim ({actual_dim}) != configured dim ({self._embedding_dim}). "
                    f"Using actual dim: {actual_dim}"
                )
                self._embedding_dim = actual_dim
            
            logger.info(f"Embedding model loaded successfully. Dim: {self._embedding_dim}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Returns: List of floats (embedding vector)
        """
        await self._ensure_model_loaded()
        
        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embedding[0].tolist()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Uses batching for efficiency.
        
        Returns: List of embedding vectors
        """
        await self._ensure_model_loaded()
        
        if not texts:
            return []
        
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
