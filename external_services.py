import os
import streamlit as st
from utils.file_utils import validate_directory, get_subdirectories
from services.json_processor import process_json_files
from services.model_service import train_model, convert_labelme_to_yolo

def json_converter_page():
    st.header("JSON Image Data Converter")
    st.write("This app converts images referenced in JSON files to base64 and stores them in the 'imageData' attribute.")
    
    # Default base path
    default_path = r"D:\SKRIPSI\CODE\ocr-lotno-model\data\export"
    
    # Directory selection
    directory_path = st.text_input("Enter directory path", value=default_path)
    
    # Browse button functionality
    if st.button("Browse"):
        subdirs = get_subdirectories(directory_path)
        if subdirs:
            selected_subdir = st.selectbox("Select a subdirectory", subdirs)
            if selected_subdir:
                directory_path = os.path.join(directory_path, selected_subdir)
                st.session_state.directory_path = directory_path
        elif not validate_directory(directory_path):
            st.error("Directory does not exist")
    
    # Update directory path if it's in session state
    if 'directory_path' in st.session_state:
        directory_path = st.session_state.directory_path
        st.text_input("Enter directory path", value=directory_path, key="directory_path_display")
    
    # Process button
    if st.button("Process JSON Files"):
        if validate_directory(directory_path):
            with st.spinner("Processing JSON files..."):
                process_json_files(directory_path)

def model_generator_page():
    st.header("YOLOv8 Model Generator")
    st.write("Train YOLOv8 models with custom datasets")
    
    # First, select the action
    action = st.radio(
        "Select Action",
        options=["Train Only", "Convert Only", "Convert and Train"],
        help="Choose what action to perform"
    )
    
    # Input parameters based on selected action
    with st.form("model_training_form"):
        # JSON directory is needed for all actions
        json_dir = st.text_input(
            "JSON Directory", 
            value="D:\\SKRIPSI\\CODE\\ocr-lotno-model\\data\\more_cap_cropped",
            help="Directory containing LabelMe JSON files"
        )
        
        # Show training parameters only if action involves training
        if action == "Train Only" or action == "Convert and Train":
            st.subheader("Training Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model = st.selectbox(
                    "Base Model",
                    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    help="Base model to use for training"
                )
                
                epochs = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="Number of training epochs"
                )
            
            with col2:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=128,
                    value=16,
                    help="Batch size for training"
                )
                
                project_dir = st.text_input(
                    "Project Directory",
                    value="D:\\SKRIPSI\\CODE\\ocr-lotno-model\\runs\\train",
                    help="Project directory for saving results"
                )
                
                name = st.text_input(
                    "Experiment Name",
                    value="yolov8n_cropped",
                    help="Experiment name"
                )
        
        submit_button = st.form_submit_button("Run")
    
    if submit_button:
        if action == "Convert Only" or action == "Convert and Train":
            with st.spinner("Converting LabelMe JSON to YOLO format..."):
                data_yaml = convert_labelme_to_yolo(json_dir)
                if data_yaml:
                    st.success(f"Dataset converted. YAML file created at: {data_yaml}")
                else:
                    st.error("Conversion failed. Please check the logs.")
                    return
        
        if action == "Train Only" or action == "Convert and Train":
            # Use the default dataset.yaml path if we're only training
            if action == "Train Only":
                data_yaml = os.path.join(json_dir, "YOLODataset", "dataset.yaml")
                if not os.path.exists(data_yaml):
                    st.error(f"Dataset YAML file not found at {data_yaml}. Please convert the dataset first.")
                    return
            
            with st.spinner(f"Training model for {epochs} epochs..."):
                success, results = train_model(
                    data_yaml=data_yaml,
                    model=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    project=project_dir,
                    name=name
                )
                if success:
                    st.success("Model training complete!")
                    # Display some results if available
                    if results:
                        st.write("Training results summary will appear here")

def main():
    st.title("OCR Lot Number Tool")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["JSON Converter", "Model Generator"])
    
    with tab1:
        json_converter_page()
    
    with tab2:
        model_generator_page()

if __name__ == "__main__":
    main()