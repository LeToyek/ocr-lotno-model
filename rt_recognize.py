import os
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('./runs/detect/train7/weights/best.pt')  # Replace with your model path

# Directory containing JPEG images
image_folder = './data/soyjoy_raw'

# Create an output folder if it doesn't exist
output_folder = './output/soyjoy_raw'
os.makedirs(output_folder, exist_ok=True)

# Iterate over all JPEG files in the directory
for filename in os.listdir(image_folder):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
        # Construct full file path
        file_path = os.path.join(image_folder, filename)

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error reading image {file_path}")
            continue

        # Perform inference
        results = model(image)

        # Render results on the image
        annotated_image = results[0].plot()  # Adjust according to YOLOv8 results format
        
        # Save annotated image
        output_file_path = os.path.join(output_folder, filename)
        
        cv2.imwrite(output_file_path, annotated_image)
        print(f"Annotated image saved to {output_file_path}")
        
print("Processing complete.")