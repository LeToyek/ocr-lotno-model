import os
import shutil

def rename_json_files_based_on_png(directory):
    """
    Renames JSON files based on matching PNG files in the same directory.
    The new name will be the first and second parts of the PNG filename split by underscore.
    
    Args:
        directory (str): Path to the directory containing files to rename
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Separate PNG and JSON files
    png_files = [f for f in files if f.lower().endswith('.png')]
    json_files = [f for f in files if f.lower().endswith('.json')]
    # print(json_files)
    
    # Create a mapping of PNG file patterns to their full names
    png_mapping = {}
    for png_file in png_files:
        parts = png_file.split('_')
        if len(parts) >= 3:
            # Use the third part as the key for matching
            key = parts[2]
            png_mapping[key] = png_file

    # print(png_mapping)
    
    # Counter for renamed files
    renamed_count = 0
    
    # Iterate through each JSON file
    for json_file in json_files:
        # Get the first part of the JSON filename (without extension)
        json_key = json_file.split('_')[0]
        print("json_key ", json_key)

        
        # Find matching PNG file
        if json_key in png_mapping:
            matching_png = png_mapping[json_key]
            png_parts = matching_png.split('_')

            print(matching_png)
            print(png_parts)
            print(png_parts[0])
            
            if len(png_parts) >= 2:
                # Create new JSON filename using first and second parts of PNG filename
                new_json_name = f"{png_parts[0]}_{png_parts[1]}_{json_key}.json"
                
                # Get full paths
                old_path = os.path.join(directory, json_file)
                new_path = os.path.join(directory, new_json_name)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {json_file} -> {new_json_name}")
                renamed_count += 1
    
    print(f"\nRenamed {renamed_count} JSON files based on matching PNG files")
def add_lot_no_to_json_files(directory):
    """
    Adds "_lot_no_1" to all JSON filenames in the directory.
    
    Args:
        directory (str): Path to the directory containing files to rename
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter for JSON files
    json_files = [f for f in files if f.lower().endswith('.json')]
    
    # Counter for renamed files
    renamed_count = 0
    
    # Iterate through each JSON file
    for json_file in json_files:
        # Create new filename with "_lot_no_1" before the extension
        base_name = json_file[:-5]  # Remove .json extension
        new_json_name = f"{base_name}_lot_no_1.json"
        
        # Get full paths
        old_path = os.path.join(directory, json_file)
        new_path = os.path.join(directory, new_json_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {json_file} -> {new_json_name}")
        renamed_count += 1
    
    print(f"\nAdded '_lot_no_1' to {renamed_count} JSON files")

def match_and_rename_preprocessed_files(preprocessed_dir, cropped_dir):
    """
    Matches files in preprocessed directory with files in cropped directory
    and renames the preprocessed files to match the pattern in cropped directory.
    
    Args:
        preprocessed_dir (str): Path to the preprocessed directory
        cropped_dir (str): Path to the cropped directory
    """
    # Check if directories exist
    if not os.path.isdir(preprocessed_dir):
        print(f"Preprocessed directory not found: {preprocessed_dir}")
        return
    
    if not os.path.isdir(cropped_dir):
        print(f"Cropped directory not found: {cropped_dir}")
        return
    
    # Get all files in both directories
    preprocessed_files = os.listdir(preprocessed_dir)
    cropped_files = os.listdir(cropped_dir)
    
    # Create a mapping of file keys to their full names in cropped directory
    cropped_mapping = {}
    for cropped_file in cropped_files:
        parts = cropped_file.split('_')
        if len(parts) >= 3:
            # Use the third part as the key for matching
            key = parts[2].split('.')[0]  # Remove file extension if present
            cropped_mapping[key] = cropped_file
    
    # Counter for renamed files
    renamed_count = 0
    
    
    # Iterate through each preprocessed file
    for preprocessed_file in preprocessed_files:
        # Skip if not a file we want to process (e.g., directories)
        if os.path.isdir(os.path.join(preprocessed_dir, preprocessed_file)):
            continue
            
        # Get the key part of the preprocessed filename
        preprocessed_parts = preprocessed_file.split('_')
        if len(preprocessed_parts) > 0:
            preprocessed_key = preprocessed_parts[0].split('.')[0]  # Remove file extension if present
            
            # Find matching cropped file
            if preprocessed_key in cropped_mapping:
                matching_cropped = cropped_mapping[preprocessed_key]
                cropped_parts = matching_cropped.split('_')
                
                if len(cropped_parts) >= 2:
                    # Get file extension from preprocessed file
                    _, ext = os.path.splitext(preprocessed_file)
                    
                    # Create new filename using first and second parts of cropped filename
                    new_name = f"{cropped_parts[0]}_{cropped_parts[1]}_{preprocessed_key}_lot_no_1{ext}"
                    
                    # Get full paths
                    old_path = os.path.join(preprocessed_dir, preprocessed_file)
                    new_path = os.path.join(preprocessed_dir, new_name)
                    
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed: {preprocessed_file} -> {new_name}")
                    renamed_count += 1
    
    print(f"\nRenamed {renamed_count} files in preprocessed directory to match cropped directory")

if __name__ == "__main__":
    # Directory paths
    source_directory = r"d:\SKRIPSI\CODE\ocr-lotno-model\data\more_cap_preprocessed"
    destination_directory = r"d:\SKRIPSI\CODE\ocr-lotno-model\data\more_cap_cropped"
    
    # Match and rename files in preprocessed directory based on cropped directory
    match_and_rename_preprocessed_files(source_directory, destination_directory)
    
    # If you want to use the other functions instead, uncomment these lines:
    # add_lot_no_to_json_files(destination_directory)
    # rename_json_files_based_on_png(destination_directory)