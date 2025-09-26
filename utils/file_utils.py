import os
import tempfile
from pathlib import Path
from typing import Optional
import streamlit as st

def get_file_type(filename: str) -> str:
    """Get file type from filename extension"""
    extension = Path(filename).suffix.lower()
    
    if extension in ['.pdf']:
        return 'pdf'
    elif extension in ['.docx']:
        return 'docx'
    elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
        return extension[1:]  # Remove the dot
    elif extension in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
        return extension[1:]  # Remove the dot
    else:
        return 'unknown'

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location and return path"""
    try:
        # Create a temporary file with the same extension
        file_extension = Path(uploaded_file.name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        raise e

def get_file_size_string(size_bytes: int) -> str:
    """Convert file size in bytes to human readable string"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_file_type(filename: str, allowed_types: list) -> bool:
    """Validate if file type is allowed"""
    file_type = get_file_type(filename)
    return file_type in allowed_types

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove special characters and replace with underscores
    import re
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return clean_name

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
