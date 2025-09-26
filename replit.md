# Overview

A multimodal Retrieval-Augmented Generation (RAG) system built with Streamlit that processes and searches across diverse file formats including documents (PDF, DOCX), images (PNG, JPG, etc.), and audio files (MP3, WAV, etc.). The system operates in offline mode using lightweight models and provides natural language querying capabilities with citation transparency. Users can upload files through a web interface, have them processed and indexed, then query the system using plain English to retrieve relevant content across all modalities with AI-generated summaries.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with session-based state management
- **Interface Design**: Single-page application with sidebar for file management and main area for querying
- **File Upload**: Multi-file upload support with real-time processing feedback
- **Search Interface**: Natural language query input with expandable results and source citations

## Backend Processing Pipeline
- **Modular Processors**: Separate processors for documents (`DocumentProcessor`), images (`ImageProcessor`), and audio (`AudioProcessor`)
- **Text Extraction**: 
  - PDFs processed using pdfplumber for robust text extraction
  - DOCX files handled via python-docx library
  - Images use OCR (Tesseract) for text extraction
- **Audio Processing**: Whisper-based speech-to-text conversion with segment chunking
- **Content Chunking**: Documents split into paragraphs/pages, audio into time segments, images into descriptive chunks

## Offline Model Management
- **Embedding Strategy**: TF-IDF vectorization using scikit-learn instead of heavy transformer models
- **Image Analysis**: Basic computer vision using PIL and Tesseract for offline operation
- **Model Caching**: Pickle-based serialization for TF-IDF models and processed data
- **Performance Optimization**: 5000-dimensional TF-IDF vectors with n-gram support (unigrams + bigrams)

## Vector Storage and Retrieval
- **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Index Type**: Inner product index with L2 normalization for cosine similarity
- **Storage Architecture**: File-based persistence with automatic loading/saving
- **Search Capabilities**: Cross-modal semantic search with configurable result limits

## Query Processing
- **RAG Implementation**: QueryEngine combines retrieval with generation using offline models
- **Context Assembly**: Retrieved chunks combined with source metadata for comprehensive responses
- **Response Generation**: AI-generated summaries based on retrieved context with citation links
- **Search Filtering**: Support for modality-specific searches (text, image, audio, or all)

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **FAISS**: Vector similarity search and clustering library by Facebook Research
- **NumPy**: Numerical computing for vector operations and array manipulation
- **scikit-learn**: TF-IDF vectorization and cosine similarity calculations

## Document Processing
- **PyPDF2**: PDF parsing and text extraction
- **pdfplumber**: Enhanced PDF text extraction with better formatting preservation  
- **python-docx**: Microsoft Word document processing

## Image Processing
- **Pillow (PIL)**: Image manipulation and format conversion
- **pytesseract**: OCR (Optical Character Recognition) for text extraction from images

## Audio Processing
- **Whisper**: OpenAI's speech-to-text model for audio transcription (referenced but implementation may use offline alternatives)

## Data Management
- **pandas**: Data manipulation and analysis for structured data handling
- **pickle**: Python object serialization for model and data persistence
- **pathlib**: Modern path handling and file system operations

## System Utilities
- **logging**: Application logging and error tracking
- **tempfile**: Temporary file management for uploaded content
- **datetime**: Timestamp management for processed files