import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import (
    Annotator,
)  # ultralytics.yolo.utils.plotting is deprecated

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv8 model
model = YOLO("./runs/detect/box_localization/weights/last.pt")  # Replace with your model path

# Directory containing JPEG images
image_folder = "./data/box"

# Create an output folder if it doesn't exist
output_folder = './output/box'
os.makedirs(output_folder, exist_ok=True)

# Iterate over all JPEG files in the directory

for filename in os.listdir(image_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        # Construct full file path
        file_path = os.path.join(image_folder, filename)

        # Read image
        image = cv2.imread(file_path)

        if image is None:
            print(f"Error reading image {file_path}")
            continue

        # Perform inference
        results = model(image)

        for r in results:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                print("bbox ",b)
                print("Name ",model.names[int(c)])
                
        # save annotated image
        annotated_image = annotator.result()
        cv2.imshow('YOLOv8 Camera Feed', annotated_image)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")
print("Processing complete.")
