import logging
from typing import List, Dict, Any
from models.offline_manager import OfflineModelManager
from vector_store.vector_manager import VectorManager

class QueryEngine:
    """Handles natural language queries and generates responses using RAG"""
    
    def __init__(self, vector_manager: VectorManager):
        self.vector_manager = vector_manager
        self.offline_manager = OfflineModelManager()
        self.logger = logging.getLogger(__name__)
    
    def query(self, query: str, search_type: str = "all_modalities", max_results: int = 5) -> Dict[str, Any]:
        """Process a natural language query and return results with AI-generated summary"""
        try:
            # Search for relevant chunks
            chunks = self.vector_manager.search(query, k=max_results * 2, search_type=search_type)
            
            # Limit to max_results
            chunks = chunks[:max_results]
            
            # Generate AI summary if chunks found
            summary = ""
            if chunks:
                summary = self._generate_summary(query, chunks)
            
            return {
                "query": query,
                "summary": summary,
                "chunks": chunks,
                "total_results": len(chunks)
            }
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "summary": f"Error processing query: {str(e)}",
                "chunks": [],
                "total_results": 0
            }
    
    def _generate_summary(self, query: str, chunks: List[Dict]) -> str:
        """Generate an AI summary based on retrieved chunks"""
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                chunk_info = f"[Source {i}: {chunk['source_file']}]"
                if chunk['type'] == 'text':
                    context_parts.append(f"{chunk_info}\n{chunk['content']}")
                elif chunk['type'] == 'image':
                    context_parts.append(f"{chunk_info}\nImage description: {chunk['content']}")
                elif chunk['type'] == 'audio':
                    context_parts.append(f"{chunk_info}\nAudio transcript: {chunk['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for AI summary
            prompt = f"""Based on the following retrieved content, provide a comprehensive answer to the user's question. 
            
User Question: {query}

Retrieved Content:
{context}

Instructions:
1. Provide a clear, direct answer to the user's question
2. Use information from the retrieved content to support your answer
3. Include specific references to sources using the format [Source X] where X is the source number
4. If the retrieved content doesn't fully answer the question, state what information is available
5. Be concise but thorough
6. If there are multiple types of content (text, images, audio), mention the relevant insights from each

Answer:"""

            # Generate offline summary using extractive summarization
            # Create a simple extractive summary by selecting the most relevant sentences
            sentences = []
            for chunk in chunks:
                content = chunk['content']
                # Split into sentences and take the first few
                chunk_sentences = content.split('.') if content else []
                for sentence in chunk_sentences[:2]:  # Take first 2 sentences per chunk
                    if sentence.strip() and len(sentence.strip()) > 20:
                        sentences.append(f"[Source {chunks.index(chunk) + 1}] {sentence.strip()}")
            
            if sentences:
                summary = f"Based on the retrieved content, here are the key findings:\n\n" + "\n".join(sentences[:5])  # Limit to 5 sentences
                summary += f"\n\nFound {len(chunks)} relevant sources that match your query."
                return summary
            else:
                return f"Found {len(chunks)} relevant results for your query, but couldn't generate a detailed summary."
        
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return f"Found {len(chunks)} relevant results for your query, but couldn't generate a summary due to an error: {str(e)}"
    
    def _extract_key_info(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Extract key information from chunks for better organization"""
        info = {
            "text_sources": [],
            "image_sources": [],
            "audio_sources": [],
            "file_types": set(),
            "source_files": set()
        }
        
        for chunk in chunks:
            info["file_types"].add(chunk["type"])
            info["source_files"].add(chunk["source_file"])
            
            if chunk["type"] == "text":
                info["text_sources"].append(chunk)
            elif chunk["type"] == "image":
                info["image_sources"].append(chunk)
            elif chunk["type"] == "audio":
                info["audio_sources"].append(chunk)
        
        return info
