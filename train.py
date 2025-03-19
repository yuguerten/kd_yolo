import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 models')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model name or path (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (cuda device id or "cpu")')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create data.yaml if it doesn't exist
    if not os.path.exists(os.path.join(args.data_path, 'data.yaml')):
        from utils.data_utils import create_data_yaml
        create_data_yaml(args.data_path)
    
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    
    # Train the model
    model.train(
        data=os.path.join(args.data_path, 'data.yaml'),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    # Evaluate the model
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    print(f"Training completed! Model saved to {os.path.join(args.project, args.name)}")

if __name__ == '__main__':
    main()
