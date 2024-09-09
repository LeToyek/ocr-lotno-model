from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('./runs/detect/train3/weights/best.pt')  # Replace with your model path

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Get the result (assuming it's a list of images)
    # For YOLOv8, this should be the frame with annotations
    annotated_frame = results[0].plot()  # Use the `.plot()` method if available

    # Display the annotated frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
