import base64
import streamlit as st

def get_base64_from_image(image_path):
    """Convert an image file to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error reading image file {image_path}: {e}")
        return None