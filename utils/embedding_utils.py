import logging
from typing import List
import numpy as np
from models.offline_manager import OfflineModelManager

# Global instance of the offline model manager
_offline_manager = None

def get_offline_manager() -> OfflineModelManager:
    """Get the global offline model manager instance"""
    global _offline_manager
    if _offline_manager is None:
        _offline_manager = OfflineModelManager()
    return _offline_manager

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using TF-IDF"""
    if not texts:
        return []
    
    try:
        manager = get_offline_manager()
        
        # Generate TF-IDF embeddings
        embeddings_matrix = manager.generate_embeddings(texts)
        
        # Convert numpy array to list of lists
        return embeddings_matrix.tolist()
    
    except Exception as e:
        logging.error(f"Error generating offline embeddings: {str(e)}")
        raise e

def get_single_embedding(text: str) -> List[float]:
    """Generate embedding for a single text"""
    embeddings = get_embeddings([text])
    return embeddings[0] if embeddings else []

def get_embedding_dimension() -> int:
    """Get the dimension of the embedding vectors"""
    manager = get_offline_manager()
    return manager._embedding_dimension
