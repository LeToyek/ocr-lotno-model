import os
import json
import glob

def find_labels():
    # Path to the directory containing JSON files
    json_dir = "d:\\SKRIPSI\\CODE\\ocr-lotno-model\\data\\more_cap_cropped"
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    changes_made = 0
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                
                # Check if the file has labels
                if "shapes" in data:
                    
                    # Process each label
                    for i, label in enumerate(data["shapes"]):
                        
                        # if label['label'] == "NSX":
                        #     print(f"Identified 'NSX' in {os.path.basename(json_file)}")
                        # elif label['label'] == "HSD":
                        #     print(f"Identified 'HSD' in {os.path.basename(json_file)}")
                        if label['label'] == 'HSD':
                            print(f"Identified 'HSD' in {os.path.basename(json_file)}")
                    
                     
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {json_file}")
    

if __name__ == "__main__":
    find_labels()