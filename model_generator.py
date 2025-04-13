import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
import argparse
import subprocess

def train_yolov8(data_yaml, model='yolov8n.pt', epochs=100, batch_size=16, project='./runs/train', name='yolov8n_cropped'):
    # Initialize the model
    model = YOLO(model)
    
    # Train the model
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, project=project, name=name)

def convert_labelme_to_yolo(json_dir):
    # Run labelme2yolo command
    cmd = f"labelme2yolo --json_dir {json_dir}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # Return the path to the dataset.yaml file
    return os.path.join(json_dir, "YOLODataset", "dataset.yaml")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train YOLOv8 model with custom dataset')
    parser.add_argument('--json_dir', type=str, default='./data/more_cap_cropped/', 
                        help='Directory containing LabelMe JSON files')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Base model to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--project', type=str, default='./runs/train',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='yolov8n_cropped',
                        help='Experiment name')
    parser.add_argument('--convert', action='store_true',
                        help='Convert LabelMe JSON to YOLO format before training')
    parser.add_argument('--full', action='store_true',
                        help='Convert LabelMe JSON to YOLO format and then train the model')
    
    args = parser.parse_args()
    
    # Convert LabelMe JSON to YOLO format if requested
    if args.convert or args.full:
        data_yaml = convert_labelme_to_yolo(args.json_dir)
        print(f"Dataset converted. YAML file created at: {data_yaml}")
    else:
        # Use the default dataset.yaml path
        data_yaml = os.path.join(args.json_dir, "YOLODataset", "dataset.yaml")
    
    # Train the model if not just converting or if full pipeline is requested
    if not args.convert or args.full:
        train_yolov8(
            data_yaml=data_yaml,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            project=args.project,
            name=args.name
        )
