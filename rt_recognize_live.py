import os
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('./runs/detect/train7/weights/best.pt')  # Replace with your model path
# model = YOLO('./runs/detect/cap_lot_number_full/weights/best.pt')  # Replace with your model path
# model = YOLO('./yolov8n.pt')  # Replace with your model path

# Open the camera (use 0 for default camera or specify camera index if multiple cameras are connected)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Process frames from the camera in real-time
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results[0].plot()  # Adjust according to YOLOv8 results format
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Camera Feed', annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
