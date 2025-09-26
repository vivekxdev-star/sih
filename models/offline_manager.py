import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image

class OfflineModelManager:
    """Manages offline models for multimodal RAG system using scikit-learn and basic tools"""
    
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model instances
        self._tfidf_vectorizer = None
        self._corpus_fitted = False
        self._embedding_dimension = 5000  # Max features for TF-IDF
        
        self.logger.info("Offline model manager initialized with TF-IDF embeddings")
    
    def get_embedding_model(self) -> TfidfVectorizer:
        """Get or load the TF-IDF vectorizer"""
        if self._tfidf_vectorizer is None:
            try:
                self.logger.info("Loading TF-IDF vectorizer for text embeddings")
                
                # Try to load existing model
                model_path = self.cache_dir / "tfidf_vectorizer.pkl"
                
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self._tfidf_vectorizer = pickle.load(f)
                    self._corpus_fitted = True
                    self.logger.info("Loaded existing TF-IDF vectorizer")
                else:
                    # Create new vectorizer
                    self._tfidf_vectorizer = TfidfVectorizer(
                        max_features=self._embedding_dimension,
                        ngram_range=(1, 2),  # Unigrams and bigrams
                        stop_words='english',
                        lowercase=True,
                        strip_accents='unicode',
                        min_df=1,  # Minimum document frequency
                        max_df=0.95  # Maximum document frequency
                    )
                    self._corpus_fitted = False
                    self.logger.info("Created new TF-IDF vectorizer")
                
            except Exception as e:
                self.logger.error(f"Error loading TF-IDF model: {e}")
                raise
        
        return self._tfidf_vectorizer
    
    def fit_corpus(self, texts: List[str]) -> None:
        """Fit the TF-IDF vectorizer on a corpus of texts"""
        try:
            if not texts:
                self.logger.warning("Empty corpus provided for fitting")
                return
            
            vectorizer = self.get_embedding_model()
            vectorizer.fit(texts)
            self._corpus_fitted = True
            
            # Save the fitted vectorizer
            model_path = self.cache_dir / "tfidf_vectorizer.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            self.logger.info(f"TF-IDF vectorizer fitted on {len(texts)} documents")
        except Exception as e:
            self.logger.error(f"Error fitting TF-IDF vectorizer: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings for a list of texts"""
        try:
            vectorizer = self.get_embedding_model()
            
            if not self._corpus_fitted:
                # If not fitted yet, fit on the provided texts
                self.fit_corpus(texts)
            
            # Transform texts to TF-IDF vectors
            tfidf_matrix = vectorizer.transform(texts)
            
            # Convert sparse matrix to dense array and return
            return tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else np.array(tfidf_matrix)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def analyze_image(self, image_path: str) -> str:
        """Analyze image using basic computer vision techniques"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get basic image properties
            width, height = image.size
            
            # Generate a basic description based on image properties
            description_parts = []
            
            # Aspect ratio analysis
            aspect_ratio = width / height
            if aspect_ratio > 1.5:
                description_parts.append("wide landscape image")
            elif aspect_ratio < 0.7:
                description_parts.append("tall portrait image")
            else:
                description_parts.append("square or standard rectangular image")
            
            # Size analysis
            total_pixels = width * height
            if total_pixels > 2000000:  # 2MP+
                description_parts.append("high resolution")
            elif total_pixels > 500000:  # 0.5MP+
                description_parts.append("medium resolution")
            else:
                description_parts.append("low resolution")
            
            # Color analysis
            image_array = np.array(image)
            mean_brightness = np.mean(image_array)
            
            if mean_brightness > 200:
                description_parts.append("bright image")
            elif mean_brightness > 100:
                description_parts.append("moderately bright image")
            else:
                description_parts.append("dark image")
            
            # Try to extract any text
            extracted_text = self.extract_text_from_image(image_path)
            if extracted_text:
                description_parts.append(f"contains text: {extracted_text[:100]}")
            
            description = f"Image ({width}x{height}): {', '.join(description_parts)}"
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return f"Image file (analysis failed): {str(e)}"
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Use Tesseract OCR
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            # Clean up the text
            text = text.strip()
            if len(text) > 10:  # Only return if substantial text found
                return text
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Basic audio file handling - returns placeholder for now"""
        try:
            # For now, return a basic description
            # In a production system, you'd use a local speech-to-text model
            file_size = os.path.getsize(audio_path)
            
            return f"Audio file detected ({file_size} bytes). Speech-to-text transcription would be performed here in a complete offline implementation."
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return f"Error processing audio file: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "cache_dir": str(self.cache_dir),
            "embedding_type": "TF-IDF",
            "embedding_dimension": self._embedding_dimension,
            "corpus_fitted": self._corpus_fitted,
            "models_loaded": {
                "tfidf_vectorizer": self._tfidf_vectorizer is not None,
                "tesseract_ocr": True,  # Always available if tesseract is installed
            }
        }
        
        return info
    
    def preload_models(self):
        """Preload all models to cache them"""
        self.logger.info("Preloading offline models...")
        try:
            self.get_embedding_model()
            self.logger.info("All offline models preloaded successfully")
        except Exception as e:
            self.logger.error(f"Error preloading models: {e}")
            raise