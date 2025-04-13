import os
import streamlit as st
import subprocess
from ultralytics import YOLO

def train_model(data_yaml, model='yolov8n.pt', epochs=100, batch_size=16, project='D:\\SKRIPSI\\CODE\\ocr-lotno-model\\runs\\train', name='yolov8n_cropped'):
    """
    Train a YOLOv8 model with the specified parameters
    """
    try:
        # Initialize the model
        yolo_model = YOLO(model)
        
        # Train the model
        results = yolo_model.train(
            data=data_yaml, 
            epochs=epochs, 
            batch=batch_size, 
            project=project, 
            name=name
        )
        
        return True, results
    except Exception as e:
        st.error(f"Error training model: {e}")
        return False, None

def convert_labelme_to_yolo(json_dir):
    """
    Convert LabelMe JSON files to YOLO format
    """
    try:
        # Run labelme2yolo command
        cmd = f"labelme2yolo --json_dir {json_dir}"
        st.info(f"Running: {cmd}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Create a placeholder for the output
        output_placeholder = st.empty()
        
        # Stream the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_placeholder.text(output.strip())
        
        # Check for errors
        stderr = process.stderr.read()
        if stderr:
            st.warning(f"Conversion warnings/errors: {stderr}")
        
        # Return the path to the dataset.yaml file
        return os.path.join(json_dir, "YOLODataset", "dataset.yaml")
    except Exception as e:
        st.error(f"Error converting dataset: {e}")
        return None