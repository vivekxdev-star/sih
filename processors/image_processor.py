import os
import base64
from pathlib import Path
from typing import List, Dict
from PIL import Image
import logging
from models.offline_manager import OfflineModelManager

class ImageProcessor:
    """Processes images for RAG system using vision models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.offline_manager = OfflineModelManager()
    
    def process_image(self, file_path: str, filename: str) -> List[Dict]:
        """Process an image and return chunks with metadata"""
        chunks = []
        
        try:
            # Generate image description using offline analysis
            description = self.offline_manager.analyze_image(file_path)
            
            # Create chunk with image description
            chunk = {
                'type': 'image',
                'content': description,
                'source_file': filename,
                'image_path': file_path,
                'chunk_id': f"{filename}_image_desc"
            }
            chunks.append(chunk)
            
            # Extract any text from image if it appears to contain text
            extracted_text = self.offline_manager.extract_text_from_image(file_path)
            if extracted_text:
                text_chunk = {
                    'type': 'text',
                    'content': f"Text extracted from image: {extracted_text}",
                    'source_file': filename,
                    'image_path': file_path,
                    'chunk_id': f"{filename}_extracted_text"
                }
                chunks.append(text_chunk)
        
        except Exception as e:
            self.logger.error(f"Error processing image {filename}: {str(e)}")
            # Create a basic chunk even if processing fails
            chunk = {
                'type': 'image',
                'content': f"Image file: {filename} (processing failed)",
                'source_file': filename,
                'image_path': file_path,
                'chunk_id': f"{filename}_basic"
            }
            chunks.append(chunk)
        
        return chunks
    
    
    
