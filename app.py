import streamlit as st
import os
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime

from processors.document_processor import DocumentProcessor
from processors.image_processor import ImageProcessor
from processors.audio_processor import AudioProcessor
from vector_store.vector_manager import VectorManager
from rag.query_engine import QueryEngine
from utils.file_utils import get_file_type, save_uploaded_file

# Initialize session state
if 'vector_manager' not in st.session_state:
    st.session_state.vector_manager = VectorManager()
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = QueryEngine(st.session_state.vector_manager)
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

def main():
    st.title("🔍 Multimodal RAG System")
    st.markdown("Upload and search across documents, images, and audio files using natural language")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 File Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload files",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp3', 'wav', 'ogg', 'm4a'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, Images (PNG, JPG, etc.), Audio (MP3, WAV, etc.)"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("Processed Files")
            for file_info in st.session_state.processed_files:
                with st.expander(f"📄 {file_info['name']}"):
                    st.write(f"Type: {file_info['type']}")
                    st.write(f"Size: {file_info['size']} bytes")
                    st.write(f"Processed: {file_info['timestamp']}")
                    if st.button(f"Remove {file_info['name']}", key=f"remove_{file_info['id']}"):
                        remove_file(file_info['id'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🗣️ Natural Language Query")
        
        # Query input
        query = st.text_area(
            "Ask a question about your documents:",
            placeholder="e.g., 'Show me the report about international development in 2024' or 'Find images related to email screenshots'",
            height=100
        )
        
        # Query options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            search_type = st.selectbox(
                "Search Type",
                ["All Modalities", "Text Only", "Images Only", "Audio Only"]
            )
        
        with col_opt2:
            max_results = st.slider("Max Results", 1, 20, 5)
        
        if st.button("🔍 Search", type="primary", disabled=not query.strip()):
            if not st.session_state.processed_files:
                st.warning("Please upload and process some files first.")
            else:
                with st.spinner("Searching..."):
                    results = st.session_state.query_engine.query(
                        query, 
                        search_type=search_type.lower().replace(" ", "_"), 
                        max_results=max_results
                    )
                    display_results(results, query)
    
    with col2:
        st.header("📊 System Status")
        
        # Statistics
        total_files = len(st.session_state.processed_files)
        total_chunks = st.session_state.vector_manager.get_total_chunks()
        
        st.metric("Total Files", total_files)
        st.metric("Total Chunks", total_chunks)
        
        # File type breakdown
        if st.session_state.processed_files:
            file_types = {}
            for file_info in st.session_state.processed_files:
                file_type = file_info['type']
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            st.subheader("File Types")
            for file_type, count in file_types.items():
                st.write(f"{file_type.upper()}: {count}")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    doc_processor = DocumentProcessor()
    img_processor = ImageProcessor()
    audio_processor = AudioProcessor()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save file temporarily
            temp_path = save_uploaded_file(uploaded_file)
            file_type = get_file_type(uploaded_file.name)
            
            # Process based on file type
            chunks = []
            if file_type in ['pdf', 'docx']:
                chunks = doc_processor.process_document(temp_path, uploaded_file.name)
            elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                chunks = img_processor.process_image(temp_path, uploaded_file.name)
            elif file_type in ['mp3', 'wav', 'ogg', 'm4a']:
                chunks = audio_processor.process_audio(temp_path, uploaded_file.name)
            
            # Add to vector store
            if chunks:
                st.session_state.vector_manager.add_chunks(chunks)
                
                # Add to processed files list
                file_info = {
                    'id': len(st.session_state.processed_files),
                    'name': uploaded_file.name,
                    'type': file_type,
                    'size': uploaded_file.size,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'chunks': len(chunks)
                }
                st.session_state.processed_files.append(file_info)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    status_text.text("Processing complete!")
    st.success(f"Successfully processed {len(uploaded_files)} files")
    st.rerun()

def remove_file(file_id):
    """Remove file from processed files and vector store"""
    st.session_state.processed_files = [
        f for f in st.session_state.processed_files if f['id'] != file_id
    ]
    # Note: In a production system, you'd also remove from vector store
    st.rerun()

def display_results(results, query):
    """Display search results with citations"""
    if not results:
        st.info("No results found for your query.")
        return
    
    st.header("🎯 Search Results")
    
    # Display AI-generated summary
    if 'summary' in results:
        st.subheader("📝 AI Summary")
        st.write(results['summary'])
    
    # Display retrieved chunks
    if 'chunks' in results and results['chunks']:
        st.subheader("📚 Retrieved Content")
        
        for i, chunk in enumerate(results['chunks']):
            with st.expander(f"Result {i+1}: {chunk['source_file']} (Score: {chunk['score']:.3f})"):
                
                # Display content based on type
                if chunk['type'] == 'text':
                    st.write(chunk['content'])
                elif chunk['type'] == 'image':
                    if 'image_path' in chunk:
                        try:
                            st.image(chunk['image_path'], caption=chunk['content'])
                        except:
                            st.write(f"Image description: {chunk['content']}")
                    else:
                        st.write(f"Image description: {chunk['content']}")
                elif chunk['type'] == 'audio':
                    st.write(f"Audio transcript: {chunk['content']}")
                    if 'audio_path' in chunk:
                        try:
                            st.audio(chunk['audio_path'])
                        except:
                            pass
                
                # Display metadata
                st.caption(f"Source: {chunk['source_file']} | Type: {chunk['type'].upper()}")
                if 'page' in chunk:
                    st.caption(f"Page: {chunk['page']}")
                if 'timestamp' in chunk:
                    st.caption(f"Timestamp: {chunk['timestamp']}")

if __name__ == "__main__":
    main()
