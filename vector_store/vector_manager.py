import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
import logging
from pathlib import Path
from utils.embedding_utils import get_embeddings

class VectorManager:
    """Manages vector storage and retrieval using FAISS"""
    
    def __init__(self, index_path: str = "vector_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize FAISS index - dimension will be set dynamically based on TF-IDF
        self.dimension = None  # Will be set when first embeddings are added
        self.index = None
        self.chunks = []  # Store chunk metadata
        
        # Load existing index if it exists
        self.load_index()
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Add chunks to the vector store"""
        if not chunks:
            return
        
        try:
            # Generate embeddings for all chunks
            texts = [chunk['content'] for chunk in chunks]
            embeddings = get_embeddings(texts)
            
            # Initialize index if it doesn't exist
            if self.index is None:
                embeddings_array = np.array(embeddings).astype('float32')
                self.dimension = embeddings_array.shape[1]  # Set dimension from first batch
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store chunk metadata
            start_idx = len(self.chunks)
            for i, chunk in enumerate(chunks):
                chunk['vector_id'] = start_idx + i
                self.chunks.append(chunk)
            
            # Save index
            self.save_index()
            
            self.logger.info(f"Added {len(chunks)} chunks to vector store")
        
        except Exception as e:
            self.logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise e
    
    def search(self, query: str, k: int = 5, search_type: str = "all_modalities") -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = get_embeddings([query])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
            
            # Filter results based on search type
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['score'] = float(score)
                    
                    # Apply search type filter
                    if self._should_include_chunk(chunk, search_type):
                        results.append(chunk)
            
            # Sort by score (higher is better for cosine similarity)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _should_include_chunk(self, chunk: Dict, search_type: str) -> bool:
        """Check if chunk should be included based on search type"""
        if search_type == "all_modalities":
            return True
        elif search_type == "text_only":
            return chunk['type'] == 'text'
        elif search_type == "images_only":
            return chunk['type'] == 'image'
        elif search_type == "audio_only":
            return chunk['type'] == 'audio'
        else:
            return True
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks in the vector store"""
        return len(self.chunks)
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_path / "faiss.index"))
                
                # Save chunks metadata
                with open(self.index_path / "chunks.pkl", 'wb') as f:
                    pickle.dump(self.chunks, f)
                
                self.logger.info("Vector index saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving vector index: {str(e)}")
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            index_file = self.index_path / "faiss.index"
            chunks_file = self.index_path / "chunks.pkl"
            
            if index_file.exists() and chunks_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load chunks metadata
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                self.logger.info(f"Loaded vector index with {len(self.chunks)} chunks")
            else:
                self.logger.info("No existing vector index found, starting fresh")
        
        except Exception as e:
            self.logger.error(f"Error loading vector index: {str(e)}")
            # Start fresh if loading fails
            self.index = None
            self.chunks = []
    
    def clear_index(self) -> None:
        """Clear the vector index and all chunks"""
        try:
            self.index = None
            self.chunks = []
            
            # Remove saved files
            index_file = self.index_path / "faiss.index"
            chunks_file = self.index_path / "chunks.pkl"
            
            if index_file.exists():
                index_file.unlink()
            if chunks_file.exists():
                chunks_file.unlink()
            
            self.logger.info("Vector index cleared")
        
        except Exception as e:
            self.logger.error(f"Error clearing vector index: {str(e)}")
