import os
import json
import base64
from PIL import Image
import io

def update_json_image_paths(directory):
    """
    Updates the imagePath and imageData attributes in JSON files to match 
    the actual image files in the directory.
    
    Args:
        directory (str): Path to the directory containing JSON and image files
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter for JSON files
    json_files = [f for f in files if f.lower().endswith('.json')]
    print("json_files: ", json_files)
    
    # Counter for updated files
    updated_count = 0
    
    # Iterate through each JSON file
    for json_file in json_files:
        json_path = os.path.join(directory, json_file)
        
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(data)
        
        # Get the base filename without extension
        base_name = os.path.splitext(json_file)[0]
        
        # Create the new image path (assuming PNG format)
        new_image_path = f"{base_name}.png"
        image_path = os.path.join(directory, new_image_path)
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # Update the imagePath attribute
        data['imagePath'] = new_image_path
        
        # Generate new imageData from the image file
        try:
            with Image.open(image_path) as img:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Encode as base64
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                # Update the imageData attribute
                data['imageData'] = img_base64
                
                # Write the updated JSON back to the file
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Updated: {json_file}")
                updated_count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nUpdated imagePath and imageData in {updated_count} JSON files")

if __name__ == "__main__":
    # Directory path
    directory = r"d:\SKRIPSI\CODE\ocr-lotno-model\data\more_cap_preprocessed"
    
    # Update JSON files
    update_json_image_paths(directory)