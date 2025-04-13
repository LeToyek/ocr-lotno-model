import os
import streamlit as st
from pathlib import Path

def get_json_files(directory_path):
    """Get all JSON files in the given directory"""
    return list(Path(directory_path).glob('*.json'))

def validate_directory(directory_path):
    """Validate if directory exists"""
    if not os.path.exists(directory_path):
        st.error("Directory does not exist")
        return False
    return True

def get_subdirectories(directory_path):
    """Get all subdirectories in the given directory"""
    if os.path.exists(directory_path):
        return [d for d in os.listdir(directory_path) 
                if os.path.isdir(os.path.join(directory_path, d))]
    return []