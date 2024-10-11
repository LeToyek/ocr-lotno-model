import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

def train_yolov8(data_yaml, model='yolov8n.pt', epochs=100, batch_size=16):
    # Initialize the model
    model = YOLO(model)
    
    # Train the model
    model.train(data=data_yaml, epochs=epochs, batch=batch_size)

if __name__ == "__main__":
    data_yaml = './data/box/YOLODataset/dataset.yaml'  # Path to your dataset.yaml
    train_yolov8(data_yaml)
