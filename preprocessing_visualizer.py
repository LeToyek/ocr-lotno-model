# ... existing code ...

import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
import easyocr
import time
import pandas as pd
from matplotlib.figure import Figure
from streamlit_image_comparison import image_comparison
from streamlit_image_select import image_select

# ... existing code ...
st.set_page_config(layout="wide", page_title="OCR Preprocessing Visualizer")

# Define preprocessing functions based on your research
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def otsu_threshold(image):
    if len(image.shape) > 2:
        image = grayscale(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def adaptive_threshold(image, block_size=11, C=2):
    if len(image.shape) > 2:
        image = grayscale(image)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, C)

def bilateral_filtered_image(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def fast_nl_means_denoising(image, h=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=templateWindowSize,
                                    searchWindowSize=searchWindowSize)

def convert_scale_abs(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def canny_edge(image, threshold1=100, threshold2=200):
    if len(image.shape) > 2:
        image = grayscale(image)
    return cv2.Canny(image, threshold1, threshold2)

def dilate(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

def erode(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def morphology_open(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def morphology_close(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def fill_black_corners(image, threshold=5):
    
    mask = np.all(image <= threshold, axis=-1)
    
    # Use inpainting to fill in the black areas
    inpainted_image = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    
    return inpainted_image
# Dictionary of preprocessing functions
preprocessing_functions = {
    "Fill Black Corners": fill_black_corners,
    "Grayscale": grayscale,
    "Otsu Threshold": otsu_threshold,
    "Adaptive Threshold": adaptive_threshold,
    "Canny Edge Detection": canny_edge,
    "Bilateral Filtered Image": bilateral_filtered_image,
    "Fast Non-Local Means Denoising": fast_nl_means_denoising,
    "Convert Scale Abs": convert_scale_abs,
    "Dilate": dilate,
    "Erode": erode,
    "Gaussian Blur": gaussian_blur,
    "Morphological Opening": morphology_open,
    "Morphological Closing": morphology_close,
}

# Function to apply a sequence of preprocessing steps
def apply_preprocessing_pipeline(image, pipeline):
    result = image.copy()
    for step in pipeline:
        func_name = step["function"]
        params = step.get("params", {})
        
        if func_name in preprocessing_functions:
            result = preprocessing_functions[func_name](result, **params)
    
    return result


# Initialize EasyOCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

@st.cache_resource
def load_yolo_model(model_path=None):
    # Load YOLO model from the specified path
    try:
        import ultralytics
        from ultralytics import YOLO
        
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            return model
        else:
            st.warning("No valid YOLO model selected. Please select a model file.")
            return None
    except ImportError:
        st.error("Ultralytics package not found. Please install with: pip install ultralytics")
        return None

ocr_models = {
    "EasyOCR": "easyocr",
    "YOLO": "yolo",
    # Add more models as needed
}

def perform_ocr_with_model(image, model_name, expected_top_text=None, expected_bottom_text=None, display_results=False, show_threshold_line=True, model_path=None):
    if model_name == "easyocr":
        return perform_easyocr_on_image(image, expected_top_text, expected_bottom_text, display_results, show_threshold_line)
    elif model_name == "yolo":
        return perform_yolo_on_image(image, expected_top_text, expected_bottom_text, display_results, show_threshold_line, model_path)
    else:
        st.error(f"Unknown model: {model_name}")
        return {}


# Add this function to perform OCR and return results
def perform_easyocr_on_image(image, expected_top_text=None, expected_bottom_text=None, display_results=False, show_threshold_line=True):
    reader = load_ocr_reader()
    
    # Ensure image is in the right format for EasyOCR
    if len(image.shape) == 3 and image.shape[2] == 3:
        ocr_image = image
    else:
        # Convert grayscale to RGB if needed
        ocr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    
    # Perform OCR
    with st.spinner("Performing OCR..."):
        start_time = time.time()
        results = reader.readtext(ocr_image)
        end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Extract text and confidence
    extracted_texts = []
    confidences = []
    bboxes = []
    
    for detection in results:
        bbox, text, confidence = detection
        extracted_texts.append(text)
        confidences.append(confidence)
        bboxes.append(bbox)
    
    # Calculate metrics
    metrics = {
        "num_detections": len(results),
        "avg_confidence": np.mean(confidences) if confidences else 0,
        "processing_time": processing_time,
        "confidences": confidences,
        "bboxes": bboxes
    }
    
    # Create visualization image with separated top and bottom text
    ocr_visualization = None
    if metrics['bboxes']:
        texts = extracted_texts
        confidences = metrics['confidences']
        bboxes = metrics['bboxes']
        ocr_visualization, clean_top_text, clean_bottom_text, raw_top_text, raw_bottom_text = visualize_ocr_results(
            ocr_image, bboxes, texts, confidences, show_threshold_line
        )
        
        # Store both raw and cleaned versions
        metrics["top_text"] = clean_top_text
        metrics["bottom_text"] = clean_bottom_text
        metrics["raw_top_text"] = raw_top_text
        metrics["raw_bottom_text"] = raw_bottom_text
        
        # Create cleaned full text
        metrics["detected_text"] = clean_top_text + clean_bottom_text
        metrics["raw_detected_text"] = " ".join(extracted_texts)
    else:
        # If no bounding boxes, just return the original image with a line
        ocr_visualization = ocr_image.copy()
        height = ocr_visualization.shape[0]
        threshold_y = height // 3
        if show_threshold_line:
            cv2.line(ocr_visualization, (0, threshold_y), (ocr_visualization.shape[1], threshold_y), (255, 0, 0), 2)
        metrics["top_text"] = ""
        metrics["bottom_text"] = ""
        metrics["raw_top_text"] = ""
        metrics["raw_bottom_text"] = ""
        metrics["detected_text"] = ""
        metrics["raw_detected_text"] = ""
    
    # Add visualization to metrics
    metrics["visualization"] = ocr_visualization
    
    # Clean expected texts for comparison
    clean_expected_top = clean_text(expected_top_text) if expected_top_text else ""
    clean_expected_bottom = clean_text(expected_bottom_text) if expected_bottom_text else ""
    
    # Calculate text similarity for top text if expected text is provided
    if expected_top_text:
        metrics["top_text_similarity"] = calculate_text_similarity(
            metrics["top_text"], clean_expected_top
        )
    
    # Calculate text similarity for bottom text if expected text is provided
    if expected_bottom_text:
        metrics["bottom_text_similarity"] = calculate_text_similarity(
            metrics["bottom_text"], clean_expected_bottom
        )
    
    # Optionally display OCR results
    if display_results:
        col1, col2 = st.columns(2)
        
        # Display detected text and confidence
        col1.markdown(f"**Raw Top Text:** {metrics['raw_top_text']}")
        col1.markdown(f"**Cleaned Top Text:** {metrics['top_text']}")
        col1.markdown(f"**Raw Bottom Text:** {metrics['raw_bottom_text']}")
        col1.markdown(f"**Cleaned Bottom Text:** {metrics['bottom_text']}")
        col1.markdown(f"**Average Confidence:** {metrics['avg_confidence']:.4f}")
        col1.markdown(f"**Number of Detections:** {metrics['num_detections']}")
        col1.markdown(f"**Processing Time:** {metrics['processing_time']:.4f} seconds")
        
        if expected_top_text:
            col1.markdown(f"**Top Text Similarity:** {metrics.get('top_text_similarity', 0):.4f}")
        if expected_bottom_text:
            col1.markdown(f"**Bottom Text Similarity:** {metrics.get('bottom_text_similarity', 0):.4f}")
        
        # Visualize OCR results on image
        if ocr_visualization is not None:
            col2.image(ocr_visualization, use_container_width=True, caption="OCR Visualization")
    
    return metrics

def perform_yolo_on_image(image, expected_top_text=None, expected_bottom_text=None, display_results=False, show_threshold_line=True, model_path=None):
    model = load_yolo_model(model_path)
    
    if model is None:
        st.error("Failed to load YOLO model")
        return {}
    
    # Ensure image is in the right format for YOLO
    if len(image.shape) == 2:
        # Convert grayscale to RGB if needed
        ocr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        ocr_image = image
    
    # Perform detection
    with st.spinner("Performing detection with YOLO..."):
        start_time = time.time()
        results = model(ocr_image)
        end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Extract results
    extracted_texts = []
    confidences = []
    bboxes = []
    
    # Process YOLO results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Convert to the format expected by visualize_ocr_results
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # Get confidence
            confidence = box.conf.item()
            
            # Get class name (if available)
            if hasattr(box, 'cls'):
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                text = class_name
            else:
                text = f"Object {len(extracted_texts)+1}"
            
            extracted_texts.append(text)
            confidences.append(confidence)
            bboxes.append(bbox)
    
    # Calculate metrics
    metrics = {
        "num_detections": len(bboxes),
        "avg_confidence": np.mean(confidences) if confidences else 0,
        "processing_time": processing_time,
        "confidences": confidences,
        "bboxes": bboxes
    }
    
    # Create visualization image with separated top and bottom text
    ocr_visualization = None
    if metrics['bboxes']:
        texts = extracted_texts
        confidences = metrics['confidences']
        bboxes = metrics['bboxes']
        ocr_visualization, clean_top_text, clean_bottom_text, raw_top_text, raw_bottom_text = visualize_ocr_results(
            ocr_image, bboxes, texts, confidences, show_threshold_line
        )
        
        # Store both raw and cleaned versions
        metrics["top_text"] = clean_top_text
        metrics["bottom_text"] = clean_bottom_text
        metrics["raw_top_text"] = raw_top_text
        metrics["raw_bottom_text"] = raw_bottom_text
        
        # Create cleaned full text
        metrics["detected_text"] = clean_top_text + clean_bottom_text
        metrics["raw_detected_text"] = " ".join(extracted_texts)
    else:
        # If no bounding boxes, just return the original image with a line
        ocr_visualization = ocr_image.copy()
        height = ocr_visualization.shape[0]
        threshold_y = height // 3
        if show_threshold_line:
            cv2.line(ocr_visualization, (0, threshold_y), (ocr_visualization.shape[1], threshold_y), (255, 0, 0), 2)
        metrics["top_text"] = ""
        metrics["bottom_text"] = ""
        metrics["raw_top_text"] = ""
        metrics["raw_bottom_text"] = ""
        metrics["detected_text"] = ""
        metrics["raw_detected_text"] = ""
    
    # Add visualization to metrics
    metrics["visualization"] = ocr_visualization
    
    # Clean expected texts for comparison
    clean_expected_top = clean_text(expected_top_text) if expected_top_text else ""
    clean_expected_bottom = clean_text(expected_bottom_text) if expected_bottom_text else ""
    
    # Calculate text similarity for top text if expected text is provided
    if expected_top_text:
        metrics["top_text_similarity"] = calculate_text_similarity(
            metrics["top_text"], clean_expected_top
        )
    
    # Calculate text similarity for bottom text if expected text is provided
    if expected_bottom_text:
        metrics["bottom_text_similarity"] = calculate_text_similarity(
            metrics["bottom_text"], clean_expected_bottom
        )
    
    # Optionally display OCR results
    if display_results:
        col1, col2 = st.columns(2)
        
        # Display detected text and confidence
        col1.markdown(f"**Raw Top Text:** {metrics['raw_top_text']}")
        col1.markdown(f"**Cleaned Top Text:** {metrics['top_text']}")
        col1.markdown(f"**Raw Bottom Text:** {metrics['raw_bottom_text']}")
        col1.markdown(f"**Cleaned Bottom Text:** {metrics['bottom_text']}")
        col1.markdown(f"**Average Confidence:** {metrics['avg_confidence']:.4f}")
        col1.markdown(f"**Number of Detections:** {metrics['num_detections']}")
        col1.markdown(f"**Processing Time:** {metrics['processing_time']:.4f} seconds")
        
        if expected_top_text:
            col1.markdown(f"**Top Text Similarity:** {metrics.get('top_text_similarity', 0):.4f}")
        if expected_bottom_text:
            col1.markdown(f"**Bottom Text Similarity:** {metrics.get('bottom_text_similarity', 0):.4f}")
        
        # Visualize OCR results on image
        if ocr_visualization is not None:
            col2.image(ocr_visualization, use_container_width=True, caption="OCR Visualization")
    
    return metrics


# Calculate text similarity
def calculate_text_similarity(text1, text2):
    # Using Levenshtein distance for string similarity
    try:
        import Levenshtein
        
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1 - (distance / max_len)
        return similarity
    except ImportError:
        # Fallback to a simpler similarity measure if Levenshtein is not installed
        common_chars = sum(1 for c in text1 if c in text2)
        total_chars = max(len(text1), len(text2))
        return common_chars / total_chars if total_chars > 0 else 1.0

# Function to visualize OCR results on image
def visualize_ocr_results(image, bboxes, texts, confidences, show_threshold_line=True):
    result_img = image.copy()
    
    # Draw a horizontal line at 1/3 of the image height if toggle is on
    height = result_img.shape[0]
    threshold_y = height // 3
    
    if show_threshold_line:
        cv2.line(result_img, (0, threshold_y), (result_img.shape[1], threshold_y), (255, 0, 0), 2)  # Red line
    
    # Separate top and bottom text
    top_texts = []
    bottom_texts = []
    
    for bbox, text, confidence in zip(bboxes, texts, confidences):
        # Convert points to integer
        points = np.array(bbox, dtype=np.int32)
        
        # Draw bounding box
        cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
        
        # Add text and confidence
        x, y = points[0]
        cv2.putText(result_img, f"{text} ({confidence:.2f})", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Determine if text is above or below the threshold line
        # Using the y-coordinate of the top-left point of the bbox
        if points[0][1] < threshold_y:
            top_texts.append(text)
        else:
            bottom_texts.append(text)
    
    # Clean and join the texts
    raw_top_text = " ".join(top_texts)
    raw_bottom_text = " ".join(bottom_texts)
    
    # Clean the texts (remove special chars, uppercase, remove spaces)
    clean_top_text = clean_text(raw_top_text)
    clean_bottom_text = clean_text(raw_bottom_text)
    
    # Return both raw and cleaned versions
    return result_img, clean_top_text, clean_bottom_text, raw_top_text, raw_bottom_text
    
def clean_text(text):
    import re
    # Remove all special characters and spaces, convert to uppercase
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    return cleaned_text

# Main app
def main():
    st.title("OCR Preprocessing Visualizer")
    
    # Sidebar for image selection
    st.sidebar.header("Image Selection")
    
    # Get list of folders in the data directory
    data_dir = os.path.join(os.getcwd(), "data")
    data_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not data_folders:
        st.sidebar.warning("No folders found in the data directory.")
        return
    
    # Add folder selection dropdown
    selected_folder = st.sidebar.selectbox("Select data folder", data_folders)
    
    # Get list of images from the selected folder
    image_dir = os.path.join(data_dir, selected_folder)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        st.sidebar.warning(f"No image files found in the {selected_folder} directory.")
        return
    
    selected_image = st.sidebar.selectbox("Select an image", image_files)
    image_path = os.path.join(image_dir, selected_image)

    # Add OCR model selection
    st.sidebar.header("OCR Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Select OCR Model",
        list(ocr_models.keys()),
        index=0
    )
    selected_model = ocr_models[selected_model_name]

    # Add YOLO model file selection if YOLO is selected
    yolo_model_path = None
    if selected_model_name == "YOLO":
        st.sidebar.subheader("YOLO Model Settings")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Option to use default model or select a custom one
        yolo_model_option = st.sidebar.radio(
            "YOLO Model Source",
            ["Use Default Model", "Select Custom Model"]
        )
        
        if yolo_model_option == "Use Default Model":
            yolo_model_path = "yolov8n.pt"  # Default model
            st.sidebar.info(f"Using default model: {yolo_model_path}")
        else:
            # Get list of model files from the models directory
            model_files = [f for f in os.listdir(models_dir) if f.lower().endswith(('.pt', '.pth', '.weights'))]
            
            
            st.sidebar.warning("No model files found in the models directory. Please add .pt, .pth, or .weights files to the models folder.")
            # Add option to upload a model file
            uploaded_model = st.sidebar.file_uploader("Upload YOLO model file", type=["pt", "pth", "weights"])
            if uploaded_model:
                # Save the uploaded model to the models directory
                model_path = os.path.join(models_dir, uploaded_model.name)
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.sidebar.success(f"Model uploaded successfully: {uploaded_model.name}")
                yolo_model_path = model_path
            if model_files:
                selected_model_file = st.sidebar.selectbox("Select YOLO model file", model_files)
                yolo_model_path = os.path.join(models_dir, selected_model_file)
                st.sidebar.info(f"Selected model: {selected_model_file}")

    # Load the selected image
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Failed to load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preview_image = image.copy()
    processed_image = image.copy()

    # Update expected text values from filename whenever a new image is selected
    # Split the filename by underscore to get top and bottom text
    filename_without_ext = os.path.splitext(selected_image)[0]  # Remove extension
    filename_parts = filename_without_ext.split('_')
    
    # Set expected top text from first part
    st.session_state.expected_top_text = filename_parts[0] if len(filename_parts) > 0 else ""
    
    # Set expected bottom text from second part (if available)
    st.session_state.expected_bottom_text = filename_parts[1] if len(filename_parts) > 1 else ""
    # Display original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    
    # Preprocessing pipeline configuration
    st.sidebar.header("Preprocessing Pipeline")
    
    # Initialize session state for pipeline if it doesn't exist
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []

    if 'expected_text' not in st.session_state:
        st.session_state.expected_text = ""

    # Add new preprocessing step
    new_step = st.sidebar.selectbox("Add preprocessing step", 
                                   ["Select..."] + list(preprocessing_functions.keys()))
    
    if new_step != "Select...":
        # Add parameters based on the selected function
        params = {}
        preview_image = image.copy()
        if st.session_state.pipeline:
            for step in st.session_state.pipeline:
                func_name = step["function"]
                step_params = step.get("params", {})
                if func_name in preprocessing_functions:
                    preview_image = preprocessing_functions[func_name](preview_image, **step_params)
        
        st.sidebar.subheader(f"Preview: {new_step}")
        
        if new_step == "Gaussian Blur":
            kernel_size = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
            params = {"kernel_size": kernel_size}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)
        
        elif new_step == "Adaptive Threshold":
            block_size = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
            C = st.sidebar.slider("C", 0, 10, 2)
            params = {"block_size": block_size, "C": C}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)
        
        elif new_step == "Canny Edge Detection":
            threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            params = {"threshold1": threshold1, "threshold2": threshold2}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)

        elif new_step == "Bilateral Filtered Image":
            d = st.sidebar.slider("Diameter", 1, 15, 9)
            sigmaColor = st.sidebar.slider("Sigma Color", 0, 255, 75)
            sigmaSpace = st.sidebar.slider("Sigma Space", 0, 255, 75)
            params = {"d": d, "sigmaColor": sigmaColor, "sigmaSpace": sigmaSpace}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)
        
        elif new_step == "Fast Non-Local Means Denoising":
            h = st.sidebar.slider("h", 0, 100, 10)
            templateWindowSize = st.sidebar.slider("Template Window Size", 3, 21, 7, step=2)
            searchWindowSize = st.sidebar.slider("Search Window Size", 3, 41, 21, step=2)
            params = {"h": h, "templateWindowSize": templateWindowSize, "searchWindowSize": searchWindowSize}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)
        
        elif new_step == "Convert Scale Abs":
            alpha = st.sidebar.slider("Alpha", 0.0, 10.0, 1.0)
            beta = st.sidebar.slider("Beta", -100, 100, 0)
            params = {"alpha": alpha, "beta": beta}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)

        elif new_step == "Fill Black Corners":
            threshold = st.sidebar.slider("Threshold", 0, 255, 5)
            params = {"threshold": threshold}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)
        
        elif new_step in ["Dilate", "Erode", "Morphological Opening", "Morphological Closing"]:
            kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
            if new_step in ["Dilate", "Erode"]:
                iterations = st.sidebar.slider("Iterations", 1, 10, 1)
                params = {"kernel_size": kernel_size, "iterations": iterations}
            else:
                params = {"kernel_size": kernel_size}
            # Generate preview
            preview_image = preprocessing_functions[new_step](preview_image, **params)

        # Display preview image
        if len(preview_image.shape) == 2:
            preview_display = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2RGB)
        else:
            preview_display = preview_image

        # Show the current pipeline steps before the preview
        if st.session_state.pipeline:
            pipeline_steps = ", ".join([step["function"] for step in st.session_state.pipeline])
            st.sidebar.caption(f"Current pipeline: {pipeline_steps}")
        
        st.sidebar.image(preview_display, use_container_width=True, caption="Parameter Preview")

        
        
        if st.sidebar.button("Add Step"):
            st.session_state.pipeline.append({
                "function": new_step,
                "params": params
            })

    # Use session state for expected text
    expected_top_text_input = st.sidebar.text_input(
        "Expected Top Text", 
        value=st.session_state.expected_top_text,
        key="expected_top_text_sidebar"
    )
    st.session_state.expected_top_text = expected_top_text_input
    
    expected_bottom_text_input = st.sidebar.text_input(
        "Expected Bottom Text", 
        value=st.session_state.expected_bottom_text,
        key="expected_bottom_text_sidebar"
    )
    st.session_state.expected_bottom_text = expected_bottom_text_input

    st.sidebar.subheader("Save/Load Pipeline")
    
    pipeline_name = st.sidebar.text_input("Pipeline Name", key="pipeline_name")
    
    if st.sidebar.button("Save Pipeline") and pipeline_name and st.session_state.pipeline:
        # Create presets directory if it doesn't exist
        presets_dir = os.path.join(os.getcwd(), "pipeline_presets")
        os.makedirs(presets_dir, exist_ok=True)
        
        # Save pipeline to JSON file
        pipeline_file = os.path.join(presets_dir, f"{pipeline_name}.json")
        
        with open(pipeline_file, 'w') as f:
            import json
            json.dump(st.session_state.pipeline, f, indent=4)
        
        st.sidebar.success(f"Pipeline '{pipeline_name}' saved successfully!")
    
    # Load saved pipelines
    presets_dir = os.path.join(os.getcwd(), "pipeline_presets")
    os.makedirs(presets_dir, exist_ok=True)
    
    # Get list of saved pipelines
    saved_pipelines = [f[:-5] for f in os.listdir(presets_dir) if f.endswith('.json')]
    
    if saved_pipelines:
        selected_preset = st.sidebar.selectbox(
            "Load Saved Pipeline", 
            ["Select..."] + saved_pipelines,
            key="load_pipeline"
        )
        
        if selected_preset != "Select..." and st.sidebar.button("Load Pipeline"):
            # Load pipeline from JSON file
            pipeline_file = os.path.join(presets_dir, f"{selected_preset}.json")
            
            try:
                with open(pipeline_file, 'r') as f:
                    import json
                    loaded_pipeline = json.load(f)
                
                # Update session state with loaded pipeline
                st.session_state.pipeline = loaded_pipeline
                st.sidebar.success(f"Pipeline '{selected_preset}' loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading pipeline: {str(e)}")
    else:
        st.sidebar.info("No saved pipelines found.")
    
    # Display and manage the current pipeline
    st.sidebar.subheader("Current Pipeline")
    
    for i, step in enumerate(st.session_state.pipeline):
        col1, col2, col3, col4 = st.sidebar.columns([3, 0.5, 0.5, 0.5])
        
        params_str = ", ".join([f"{k}={v}" for k, v in step.get("params", {}).items()])
        step_display = f"{step['function']} ({params_str})" if params_str else step['function']
        
        col1.write(f"{i+1}. {step_display}")
        
        # Add up/down buttons for reordering
        if i > 0 and col2.button("↑", key=f"up_{i}"):
            # Move step up in the pipeline
            st.session_state.pipeline[i], st.session_state.pipeline[i-1] = st.session_state.pipeline[i-1], st.session_state.pipeline[i]
            st.rerun()
            
        if i < len(st.session_state.pipeline) - 1 and col3.button("↓", key=f"down_{i}"):
            # Move step down in the pipeline
            st.session_state.pipeline[i], st.session_state.pipeline[i+1] = st.session_state.pipeline[i+1], st.session_state.pipeline[i]
            st.rerun()
        
        if col4.button("x", key=f"remove_{i}"):
            st.session_state.pipeline.pop(i)
            st.rerun()
    
    if st.sidebar.button("Clear Pipeline"):
        st.session_state.pipeline = []
        st.rerun()
    
    # Apply the preprocessing pipeline and display results
    if st.session_state.pipeline:
        st.subheader("Preprocessing Results")
        
        # Process the image through the pipeline
        
        
        # Create columns for step-by-step visualization
        cols = st.columns(min(3, len(st.session_state.pipeline) + 1))
        
        cols[0].write("Original")
        cols[0].image(image, use_container_width=True)
        
        # Apply each step and show intermediate results
        for i, step in enumerate(st.session_state.pipeline):
            func_name = step["function"]
            params = step.get("params", {})
            
            if func_name in preprocessing_functions:
                processed_image = preprocessing_functions[func_name](processed_image, **params)
                
                # Display intermediate result
                col_idx = (i + 1) % 3
                cols[col_idx].write(f"Step {i+1}: {func_name}")
                
                # Convert to RGB for display if it's grayscale
                if len(processed_image.shape) == 2:
                    display_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                else:
                    display_img = processed_image
                
                cols[col_idx].image(display_img, use_container_width=True)
        
        # Display final result
        st.subheader("Final Result")
        
        # Convert to RGB for display if it's grayscale
        if len(processed_image.shape) == 2:
            final_display = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        else:
            final_display = processed_image
            
        st.image(final_display, use_container_width=True)

        
        # Allow downloading the processed image
        buf = io.BytesIO()
        if len(processed_image.shape) == 2:
            # Convert grayscale to RGB for PIL
            pil_img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB))
        else:
            pil_img = Image.fromarray(processed_image)
        
        pil_img.save(buf, format="PNG")
        
        st.download_button(
            label="Download Processed Image",
            data=buf.getvalue(),
            file_name=f"processed_{selected_image}",
            mime="image/png"
        )
        
        st.subheader("Export All Processed Images")
        
        # Set base export directory path
        base_export_dir = "D:\\SKRIPSI\\CODE\\ocr-lotno-model\\data"
        
        # Create a list of subdirectories for the dropdown
        export_subdirs = [""] + [d for d in os.listdir(base_export_dir) if os.path.isdir(os.path.join(base_export_dir, d))]
        selected_subdir = st.selectbox("Select export subdirectory", export_subdirs, index=0)
        
        # Combine base path with selected subdirectory
        if selected_subdir:
            export_dir = os.path.join(base_export_dir, selected_subdir)
        else:
            export_dir = base_export_dir
            
        # Allow user to add a custom folder name
        custom_folder = st.text_input("Add custom subfolder (optional)", value="")
        if custom_folder:
            export_dir = os.path.join(export_dir, custom_folder)
        
        st.info(f"Images will be exported to: {export_dir}")
        
        if st.button("Export All Processed Images"):
            if not os.path.exists(export_dir):
                try:
                    os.makedirs(export_dir)
                    st.success(f"Created directory: {export_dir}")
                except Exception as e:
                    st.error(f"Failed to create directory: {str(e)}")
                    return
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process and export all images
            for i, img_file in enumerate(image_files):
                status_text.text(f"Processing image {i+1}/{len(image_files)}: {img_file}")
                progress_bar.progress((i) / len(image_files))
                
                # Load image
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    st.warning(f"Failed to load image: {img_file}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process image with current pipeline
                processed_img = img.copy()
                for step in st.session_state.pipeline:
                    func_name = step["function"]
                    params = step.get("params", {})
                    if func_name in preprocessing_functions:
                        processed_img = preprocessing_functions[func_name](processed_img, **params)
                
                # Save processed image
                output_path = os.path.join(export_dir, img_file)
                
                # Convert to RGB if grayscale
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                
                # Save using PIL to ensure proper format handling
                pil_img = Image.fromarray(processed_img)
                pil_img.save(output_path)
                
                # Check if corresponding JSON file exists and copy it
                json_filename = os.path.splitext(img_file)[0] + ".json"
                json_path = os.path.join(image_dir, json_filename)
                
                if os.path.exists(json_path):
                    import shutil
                    output_json_path = os.path.join(export_dir, json_filename)
                    shutil.copy2(json_path, output_json_path)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Export complete!")
            
            st.success(f"Successfully exported {len(image_files)} processed images to {export_dir}")
    
    
    
    # Perform OCR on original image
    with st.spinner("Performing OCR on original image..."):
        # Add toggle for threshold line
        show_threshold_line = st.checkbox("Show Threshold Line", value=True)
        
        if selected_model_name == "YOLO" and yolo_model_path:
            # For YOLO, we need to pass the model path
            original_ocr_results = perform_yolo_on_image(
                image, 
                st.session_state.expected_top_text,
                st.session_state.expected_bottom_text,
                show_threshold_line=show_threshold_line,
                model_path=yolo_model_path
            )

            final_ocr_results = perform_yolo_on_image(
                processed_image, 
                st.session_state.expected_top_text,
                st.session_state.expected_bottom_text,
                show_threshold_line=show_threshold_line,
                model_path=yolo_model_path
            )
        else:
            # For other models, use the standard function
            original_ocr_results = perform_ocr_with_model(
                image, 
                selected_model,
                st.session_state.expected_top_text,
                st.session_state.expected_bottom_text,
                show_threshold_line=show_threshold_line
            )

            final_ocr_results = perform_ocr_with_model(
                processed_image, 
                selected_model,
                st.session_state.expected_top_text,
                st.session_state.expected_bottom_text,
                show_threshold_line=show_threshold_line
            )

        # Create a comparison table for OCR metrics
        st.subheader("OCR Results Comparison")
        
                # Create comparison data
        comparison_data = {
            "Metric": ["Raw Detected Text", "Cleaned Detected Text", "Raw Top Text", "Cleaned Top Text", 
                      "Raw Bottom Text", "Cleaned Bottom Text", "Number of Detections", 
                      "Average Confidence", "Processing Time (s)"],
            "Original Image": [
                original_ocr_results.get("raw_detected_text", ""),
                original_ocr_results["detected_text"],
                original_ocr_results.get("raw_top_text", ""),
                original_ocr_results.get("top_text", ""),
                original_ocr_results.get("raw_bottom_text", ""),
                original_ocr_results.get("bottom_text", ""),
                original_ocr_results["num_detections"],
                f"{original_ocr_results['avg_confidence']:.4f}",
                f"{original_ocr_results['processing_time']:.4f}"
            ],
            "Processed Image": [
                final_ocr_results.get("raw_detected_text", ""),
                final_ocr_results["detected_text"],
                final_ocr_results.get("raw_top_text", ""),
                final_ocr_results.get("top_text", ""),
                final_ocr_results.get("raw_bottom_text", ""),
                final_ocr_results.get("bottom_text", ""),
                final_ocr_results["num_detections"],
                f"{final_ocr_results['avg_confidence']:.4f}",
                f"{final_ocr_results['processing_time']:.4f}"
            ]
        }
        
        # Add text similarity for top text if expected text is provided
        if st.session_state.expected_top_text:
            comparison_data["Metric"].append("Top Text Similarity")
            comparison_data["Original Image"].append(
                f"{original_ocr_results.get('top_text_similarity', 0):.4f}"
            )
            comparison_data["Processed Image"].append(
                f"{final_ocr_results.get('top_text_similarity', 0):.4f}"
            )
        
        # Add text similarity for bottom text if expected text is provided
        if st.session_state.expected_bottom_text:
            comparison_data["Metric"].append("Bottom Text Similarity")
            comparison_data["Original Image"].append(
                f"{original_ocr_results.get('bottom_text_similarity', 0):.4f}"
            )
            comparison_data["Processed Image"].append(
                f"{final_ocr_results.get('bottom_text_similarity', 0):.4f}"
            )
            
        # Create and display the comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)

        image_comparison(
            img1= original_ocr_results['visualization'],
            img2= final_ocr_results['visualization'],
            label1= "Original",
            label2= "Final",
            starting_position= 50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

        st.subheader("Batch Processing")
        
                # Add batch processing button
        if st.button("Process All Images in Folder"):
            # Get all image files in the folder
            all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not all_image_files:
                st.warning(f"No image files found in the {selected_folder} directory.")
                return
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize data collection
            batch_results = {
                "filename": [],
                "original_top_text": [],
                "original_bottom_text": [],
                "processed_top_text": [],
                "processed_bottom_text": [],
                "original_top_similarity": [],
                "original_bottom_similarity": [],
                "processed_top_similarity": [],
                "processed_bottom_similarity": [],
                "original_top_confidence": [],
                "original_bottom_confidence": [],
                "processed_top_confidence": [],
                "processed_bottom_confidence": [],
                "expected_top_text": [],
                "expected_bottom_text": []
            }
            
            # Process each image
            for i, img_file in enumerate(all_image_files):
                status_text.text(f"Processing image {i+1}/{len(all_image_files)}: {img_file}")
                progress_bar.progress((i) / len(all_image_files))
                
                # Load image
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract expected text from filename
                filename_without_ext = os.path.splitext(img_file)[0]
                filename_parts = filename_without_ext.split('_')
                expected_top = filename_parts[0] if len(filename_parts) > 0 else ""
                expected_bottom = filename_parts[1] if len(filename_parts) > 1 else ""
                
                # Process image with current pipeline
                processed_img = img.copy()
                for step in st.session_state.pipeline:
                    func_name = step["function"]
                    params = step.get("params", {})
                    if func_name in preprocessing_functions:
                        processed_img = preprocessing_functions[func_name](processed_img, **params)
                
                # Perform OCR on original and processed images
                if selected_model_name == "YOLO" and yolo_model_path:
                    original_results = perform_yolo_on_image(
                        img, 
                        expected_top, 
                        expected_bottom, 
                        show_threshold_line=False,
                        model_path=yolo_model_path
                    )
                    processed_results = perform_yolo_on_image(
                        processed_img, 
                        expected_top, 
                        expected_bottom, 
                        show_threshold_line=False,
                        model_path=yolo_model_path
                    )
                else:
                    original_results = perform_ocr_with_model(
                        img, 
                        selected_model,
                        expected_top, 
                        expected_bottom, 
                        show_threshold_line=False
                    )
                    processed_results = perform_ocr_with_model(
                        processed_img, 
                        selected_model,
                        expected_top, 
                        expected_bottom, 
                        show_threshold_line=False
                    )

                # Extract top and bottom confidences
                original_top_conf = []
                original_bottom_conf = []
                processed_top_conf = []
                processed_bottom_conf = []
                
                # Process original image confidences
                if original_results['bboxes']:
                    height = img.shape[0]
                    threshold_y = height // 3
                    
                    for bbox, conf in zip(original_results['bboxes'], original_results['confidences']):
                        points = np.array(bbox, dtype=np.int32)
                        if points[0][1] < threshold_y:
                            original_top_conf.append(conf)
                        else:
                            original_bottom_conf.append(conf)
                
                # Process processed image confidences
                if processed_results['bboxes']:
                    height = processed_img.shape[0]
                    threshold_y = height // 3
                    
                    for bbox, conf in zip(processed_results['bboxes'], processed_results['confidences']):
                        points = np.array(bbox, dtype=np.int32)
                        if points[0][1] < threshold_y:
                            processed_top_conf.append(conf)
                        else:
                            processed_bottom_conf.append(conf)
                
                # Calculate average confidences
                avg_original_top_conf = np.mean(original_top_conf) if original_top_conf else 0
                avg_original_bottom_conf = np.mean(original_bottom_conf) if original_bottom_conf else 0
                avg_processed_top_conf = np.mean(processed_top_conf) if processed_top_conf else 0
                avg_processed_bottom_conf = np.mean(processed_bottom_conf) if processed_bottom_conf else 0
                
                # Add data to batch results
                batch_results["filename"].append(img_file)
                batch_results["original_top_text"].append(original_results.get("top_text", ""))
                batch_results["original_bottom_text"].append(original_results.get("bottom_text", ""))
                batch_results["processed_top_text"].append(processed_results.get("top_text", ""))
                batch_results["processed_bottom_text"].append(processed_results.get("bottom_text", ""))
                batch_results["original_top_similarity"].append(original_results.get("top_text_similarity", 0))
                batch_results["original_bottom_similarity"].append(original_results.get("bottom_text_similarity", 0))
                batch_results["processed_top_similarity"].append(processed_results.get("top_text_similarity", 0))
                batch_results["processed_bottom_similarity"].append(processed_results.get("bottom_text_similarity", 0))
                batch_results["original_top_confidence"].append(avg_original_top_conf)
                batch_results["original_bottom_confidence"].append(avg_original_bottom_conf)
                batch_results["processed_top_confidence"].append(avg_processed_top_conf)
                batch_results["processed_bottom_confidence"].append(avg_processed_bottom_conf)
                batch_results["expected_top_text"].append(expected_top)
                batch_results["expected_bottom_text"].append(expected_bottom)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Create DataFrame from results
            batch_df = pd.DataFrame(batch_results)
            
            # Calculate improvement metrics
            batch_df["top_similarity_improvement"] = batch_df["processed_top_similarity"] - batch_df["original_top_similarity"]
            batch_df["bottom_similarity_improvement"] = batch_df["processed_bottom_similarity"] - batch_df["original_bottom_similarity"]
            batch_df["total_improvement"] = batch_df["top_similarity_improvement"] + batch_df["bottom_similarity_improvement"]
            
            # Calculate success rate (similarity > 0.8)
            original_top_success = (batch_df['original_top_similarity'] > 0.8).mean() * 100
            original_bottom_success = (batch_df['original_bottom_similarity'] > 0.8).mean() * 100
            processed_top_success = (batch_df['processed_top_similarity'] > 0.8).mean() * 100
            processed_bottom_success = (batch_df['processed_bottom_similarity'] > 0.8).mean() * 100
            
            # Create distribution plots
            fig = plt.figure(figsize=(12, 20))
            
            # Top Text Similarity Distribution
            ax1 = plt.subplot(4, 1, 1)
            ax1.hist(batch_df['original_top_similarity'], alpha=0.5, bins=20, label='Original')
            ax1.hist(batch_df['processed_top_similarity'], alpha=0.5, bins=20, label='Processed')
            ax1.set_title('Top Text Similarity Distribution')
            ax1.set_xlabel('Similarity Score')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True)
            
            # Bottom Text Similarity Distribution
            ax2 = plt.subplot(4, 1, 2)
            ax2.hist(batch_df['original_bottom_similarity'], alpha=0.5, bins=20, label='Original')
            ax2.hist(batch_df['processed_bottom_similarity'], alpha=0.5, bins=20, label='Processed')
            ax2.set_title('Bottom Text Similarity Distribution')
            ax2.set_xlabel('Similarity Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True)
            
            # Top Text Confidence Distribution
            ax3 = plt.subplot(4, 1, 3)
            ax3.hist(batch_df['original_top_confidence'], alpha=0.5, bins=20, label='Original')
            ax3.hist(batch_df['processed_top_confidence'], alpha=0.5, bins=20, label='Processed')
            ax3.set_title('Top Text Confidence Distribution')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True)
            
            # Bottom Text Confidence Distribution
            ax4 = plt.subplot(4, 1, 4)
            ax4.hist(batch_df['original_bottom_confidence'], alpha=0.5, bins=20, label='Original')
            ax4.hist(batch_df['processed_bottom_confidence'], alpha=0.5, bins=20, label='Processed')
            ax4.set_title('Bottom Text Confidence Distribution')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Create scatter plots for similarity comparison
            fig2 = plt.figure(figsize=(12, 10))
            
            # Top Text Similarity Comparison
            ax1 = plt.subplot(2, 1, 1)
            ax1.scatter(batch_df['original_top_similarity'], batch_df['processed_top_similarity'])
            ax1.plot([0, 1], [0, 1], 'r--')  # Diagonal line
            ax1.set_title('Top Text Similarity: Original vs Processed')
            ax1.set_xlabel('Original Similarity')
            ax1.set_ylabel('Processed Similarity')
            ax1.grid(True)
            
            # Bottom Text Similarity Comparison
            ax2 = plt.subplot(2, 1, 2)
            ax2.scatter(batch_df['original_bottom_similarity'], batch_df['processed_bottom_similarity'])
            ax2.plot([0, 1], [0, 1], 'r--')  # Diagonal line
            ax2.set_title('Bottom Text Similarity: Original vs Processed')
            ax2.set_xlabel('Original Similarity')
            ax2.set_ylabel('Processed Similarity')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Create box plots for confidence scores
            fig3 = plt.figure(figsize=(12, 8))
            
            # Prepare data for box plots
            confidence_data = [
                batch_df['original_top_confidence'],
                batch_df['processed_top_confidence'],
                batch_df['original_bottom_confidence'],
                batch_df['processed_bottom_confidence']
            ]
            
            labels = [
                'Original Top', 
                'Processed Top', 
                'Original Bottom', 
                'Processed Bottom'
            ]
            
            plt.boxplot(confidence_data, labels=labels)
            plt.title('Confidence Score Distribution')
            plt.ylabel('Confidence Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Create correlation heatmap
            corr_columns = [
                'original_top_similarity', 'processed_top_similarity',
                'original_bottom_similarity', 'processed_bottom_similarity',
                'original_top_confidence', 'processed_top_confidence',
                'original_bottom_confidence', 'processed_bottom_confidence'
            ]
            
            corr_df = batch_df[corr_columns]
            correlation = corr_df.corr()
            
            fig4 = plt.figure(figsize=(12, 10))
            plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
            plt.yticks(range(len(correlation.columns)), correlation.columns)
            
            # Add correlation values
            for i in range(len(correlation.columns)):
                for j in range(len(correlation.columns)):
                    plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", 
                             ha="center", va="center", color="black")
            
            plt.title('Correlation Between Metrics')
            plt.tight_layout()
            
            # Store batch results in session state to persist between image selections
            st.session_state.batch_df = batch_df
            st.session_state.batch_figs = {
                'distribution_fig': fig,
                'scatter_fig': fig2,
                'boxplot_fig': fig3,
                'correlation_fig': fig4
            }
            st.session_state.batch_stats = {
                'original_top_success': original_top_success,
                'original_bottom_success': original_bottom_success,
                'processed_top_success': processed_top_success,
                'processed_bottom_success': processed_bottom_success,
                'selected_folder': selected_folder
            }
            
            # Display the results immediately after processing
            st.subheader(f"Batch Processing Results for {selected_folder}")
            
            # Display summary statistics
            st.write("### Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Original Image")
                st.write(f"Average Top Text Similarity: {batch_df['original_top_similarity'].mean():.4f}")
                st.write(f"Average Bottom Text Similarity: {batch_df['original_bottom_similarity'].mean():.4f}")
                st.write(f"Average Top Text Confidence: {batch_df['original_top_confidence'].mean():.4f}")
                st.write(f"Average Bottom Text Confidence: {batch_df['original_bottom_confidence'].mean():.4f}")
            
            with col2:
                st.write("#### Processed Image")
                st.write(f"Average Top Text Similarity: {batch_df['processed_top_similarity'].mean():.4f}")
                st.write(f"Average Bottom Text Similarity: {batch_df['processed_bottom_similarity'].mean():.4f}")
                st.write(f"Average Top Text Confidence: {batch_df['processed_top_confidence'].mean():.4f}")
                st.write(f"Average Bottom Text Confidence: {batch_df['processed_bottom_confidence'].mean():.4f}")
            
            st.write("#### Improvement")
            st.write(f"Average Top Text Similarity Improvement: {batch_df['top_similarity_improvement'].mean():.4f}")
            st.write(f"Average Bottom Text Similarity Improvement: {batch_df['bottom_similarity_improvement'].mean():.4f}")
            
            # Display visualizations
            st.write("### Visualizations")
            st.pyplot(fig)
            st.pyplot(fig2)
            
            # Display full results table
            st.write("### Full Results Table")
            st.dataframe(batch_df)
            
            # Allow downloading the results as CSV
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"batch_results_{selected_folder}.csv",
                mime="text/csv"
            )
            
            # Additional statistical analysis
            st.write("### Additional Statistical Analysis")
            
            # Success rate display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Success Rate (Similarity > 0.8)")
                st.write(f"Original Top Text: {st.session_state.batch_stats['original_top_success']:.2f}%")
                st.write(f"Original Bottom Text: {st.session_state.batch_stats['original_bottom_success']:.2f}%")
            
            with col2:
                st.write("#### Success Rate (Similarity > 0.8)")
                st.write(f"Processed Top Text: {st.session_state.batch_stats['processed_top_success']:.2f}%")
                st.write(f"Processed Bottom Text: {st.session_state.batch_stats['processed_bottom_success']:.2f}%")
            
            # Display additional visualizations
            st.pyplot(st.session_state.batch_figs['boxplot_fig'])
            st.pyplot(st.session_state.batch_figs['correlation_fig'])

if __name__ == "__main__":
    main()