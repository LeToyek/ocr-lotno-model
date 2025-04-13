import os
import json
import streamlit as st
from utils.image_utils import get_base64_from_image
from utils.file_utils import get_json_files

def process_json_files(directory_path):
    """Process all JSON files in the given directory"""
    json_files = get_json_files(directory_path)
    
    if not json_files:
        st.warning(f"No JSON files found in {directory_path}")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "imagePath" in data:
                image_path = data["imagePath"]
                # Check if image_path is relative or absolute
                if not os.path.isabs(image_path):
                    # Assume it's relative to the JSON file location
                    image_path = os.path.join(os.path.dirname(json_file), image_path)
                
                if os.path.exists(image_path):
                    base64_data = get_base64_from_image(image_path)
                    if base64_data:
                        data["imageData"] = base64_data
                        
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        
                        status_text.text(f"Processed {i+1}/{len(json_files)}: {json_file.name} - Success")
                    else:
                        status_text.text(f"Processed {i+1}/{len(json_files)}: {json_file.name} - Failed to convert image")
                else:
                    status_text.text(f"Processed {i+1}/{len(json_files)}: {json_file.name} - Image not found: {image_path}")
            else:
                status_text.text(f"Processed {i+1}/{len(json_files)}: {json_file.name} - No 'imagePath' attribute found")
        
        except Exception as e:
            status_text.text(f"Error processing {json_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(json_files))
    
    st.success(f"Processed {len(json_files)} JSON files")