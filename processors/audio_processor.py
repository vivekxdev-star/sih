import os
import tempfile
from pathlib import Path
from typing import List, Dict
import logging
from models.offline_manager import OfflineModelManager

class AudioProcessor:
    """Processes audio files for RAG system using Whisper"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.offline_manager = OfflineModelManager()
    
    def process_audio(self, file_path: str, filename: str) -> List[Dict]:
        """Process an audio file and return chunks with metadata"""
        chunks = []
        
        try:
            # Process audio using offline methods
            transcript = self.offline_manager.transcribe_audio(file_path)
            
            if transcript and transcript.strip():
                # Split transcript into chunks (by sentences or time segments)
                text_chunks = self._split_transcript(transcript)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = {
                        'type': 'audio',
                        'content': chunk_text,
                        'source_file': filename,
                        'audio_path': file_path,
                        'segment': i + 1,
                        'chunk_id': f"{filename}_segment_{i + 1}"
                    }
                    chunks.append(chunk)
            else:
                # Create a basic chunk even if transcription is empty
                chunk = {
                    'type': 'audio',
                    'content': f"Audio file: {filename} (no transcription available)",
                    'source_file': filename,
                    'audio_path': file_path,
                    'chunk_id': f"{filename}_basic"
                }
                chunks.append(chunk)
        
        except Exception as e:
            self.logger.error(f"Error processing audio {filename}: {str(e)}")
            # Create a basic chunk even if processing fails
            chunk = {
                'type': 'audio',
                'content': f"Audio file: {filename} (processing failed: {str(e)})",
                'source_file': filename,
                'audio_path': file_path,
                'chunk_id': f"{filename}_error"
            }
            chunks.append(chunk)
        
        return chunks
    
    
    def _compress_audio(self, file_path: str) -> str:
        """Compress audio file if it's too large"""
        try:
            # This is a simplified approach - in production you might use ffmpeg
            # For now, we'll just return the original path and let the API handle it
            self.logger.warning("Audio compression not implemented - using original file")
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error compressing audio: {str(e)}")
            return file_path
    
    def _split_transcript(self, transcript: str) -> List[str]:
        """Split transcript into meaningful chunks"""
        # Split by sentences first
        sentences = transcript.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would make chunk too long, start new chunk
            if len(current_chunk + sentence) > 500:  # Smaller chunks for audio
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            current_chunk += sentence + ". "
        
        # Add final chunk if exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no proper chunks were created, return the whole transcript
        if not chunks and transcript.strip():
            chunks = [transcript.strip()]
        
        return chunks
