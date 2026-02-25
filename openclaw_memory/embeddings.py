import sys
import numpy as np
from typing import List, Protocol

class BaseEmbedding(Protocol):
    """Protocol defining the interface for any embedding model plugged into OpenClaw."""
    def encode(self, text: str) -> np.ndarray:
        ...

class SentenceTransformerEmbedding:
    """Default local embedding wrapper utilizing all-MiniLM-L6-v2."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("WARNING: 'sentence-transformers' not installed. Please install it to use the default embedding model.")
            sys.exit(1)
            
    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text)
        vec = vec / np.linalg.norm(vec)  # L2 normalize for inner-product
        return vec.astype(np.float32)

class MockEmbedding:
    """Mock zero-vector embedding for testing environments without downloading models."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        
    def encode(self, text: str) -> np.ndarray:
        return np.zeros(self.dimension, dtype=np.float32)
