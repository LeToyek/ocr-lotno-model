import os
import json
import glob

def fix_labels():
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
                    modified = False
                    
                    # Process each label
                    for i, label in enumerate(data["shapes"]):
                        print(label)
                        print("="*50)
                        # Fix "`1" to "1"
                        if label['label'] == "`1":
                            data["shapes"][i]['label'] = "1"
                            modified = True
                            print(f"Changed '`1' to '1' in {os.path.basename(json_file)}")
                        
                        # Fix "k" to "K"
                        elif label['label'] == "k":
                            data["shapes"][i]['label'] = "K"
                            modified = True
                            print(f"Changed 'k' to 'K' in {os.path.basename(json_file)}")
                    
                    # Save the file if changes were made
                    if modified:
                        with open(json_file, 'w') as f_write:
                            json.dump(data, f_write, indent=2)
                        changes_made += 1
                        
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {json_file}")
    
    print(f"Completed! Modified {changes_made} files.")

if __name__ == "__main__":
    fix_labels()