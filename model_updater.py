import os
import argparse
from ultralytics import YOLO

# Ensure any library linking errors are handled on Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_yolov8(data_yaml, model_path='yolov8n.pt', epochs=100, batch_size=16, save_dir="./runs/train"):
    """
    Function to train YOLOv8 model with specified dataset and settings.

    Args:
        data_yaml (str): Path to the dataset configuration file (YAML format).
        model_path (str): Path to the pre-trained model or checkpoint for transfer learning.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        save_dir (str): Directory to save training weights and logs.
    """
    # Initialize the model (can load pre-trained weights or an existing model)
    model = YOLO(model_path)
    
    # Train the model with the new dataset
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, save_dir=save_dir)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with custom datasets.")
    
    # Arguments
    parser.add_argument("--data_yaml", type=str, default='./data/soyjoy_raw/YOLODataset/dataset.yaml', required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--model_path", type=str, default="./runs/detect/combined_lot_no/weights/best.pt", help="Path to the model file to be used for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--save_dir", type=str, default="./runs/train", help="Directory to save training results and model checkpoints.")

    # Parse arguments
    args = parser.parse_args()
    
    # Train the model using parsed arguments
    train_yolov8(data_yaml=args.data_yaml, model_path=args.model_path, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_dir)
