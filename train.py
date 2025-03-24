import os
import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
from utils.distill import apply_distillation


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with optional knowledge distillation')
    
    # Dataset arguments
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model name or path')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='Device to use (CUDA device ID or "cpu")')
    parser.add_argument('--project', default='runs/train', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    
    # Knowledge distillation arguments
    parser.add_argument('--distill', action='store_true', help='Enable knowledge distillation')
    parser.add_argument('--teacher-model', type=str, help='Path to the teacher model')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature parameter for distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for balancing distillation and task loss')
    
    return parser.parse_args()


def prepare_data_yaml(data_path):
    """Create data.yaml file if it doesn't exist"""
    data_yaml_path = os.path.join(data_path, 'data.yaml')
    
    if os.path.exists(data_yaml_path):
        print(f"Using existing data.yaml at {data_yaml_path}")
        return data_yaml_path
    
    # Create data.yaml if it doesn't exist
    print("Creating data.yaml file...")
    
    # Check if images and labels directories exist
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Dataset structure not valid. Expected directories: {images_dir}, {labels_dir}")
    
    # Count classes by checking all label files
    classes = set()
    label_files = Path(labels_dir).glob('*.txt')
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    classes.add(class_id)
    
    num_classes = len(classes)
    
    if num_classes == 0:
        raise ValueError("No classes found in the dataset.")
    
    # Create class names as placeholders
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Create train/val/test split paths
    train_dir = os.path.join(images_dir, 'train') if os.path.exists(os.path.join(images_dir, 'train')) else images_dir
    val_dir = os.path.join(images_dir, 'val') if os.path.exists(os.path.join(images_dir, 'val')) else images_dir
    
    # Create the data.yaml content
    data_dict = {
        'path': data_path,
        'train': train_dir,
        'val': val_dir,
        'test': '',
        'nc': num_classes,
        'names': class_names
    }
    
    # Write data.yaml file
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_dict, f, sort_keys=False)
    
    print(f"Created data.yaml at {data_yaml_path} with {num_classes} classes.")
    return data_yaml_path


def train_with_distillation(args):
    """Train the model with knowledge distillation"""
    # Check if teacher model is specified
    if not args.teacher_model:
        raise ValueError("Teacher model path must be specified for distillation.")
    
    # Prepare data config
    data_yaml_path = prepare_data_yaml(args.data_path)
    
    # Load the teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = YOLO(args.teacher_model)
    
    # Load the student model
    print(f"Loading student model: {args.model}")
    student_model = YOLO(args.model)
    
    # Create a custom training loop for distillation
    class DistillationTrainer:
        def __init__(self, student_model, teacher_model, args):
            self.student_model = student_model
            self.teacher_model = teacher_model
            self.args = args
            self.device = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else "cpu")
            
            # Set teacher model to eval mode
            self.teacher_model.model.eval()
            
            # Pre-process the student model for training (make it ready to accept our custom loss)
            self.student_model.add_callback("on_train_batch_end", self._on_batch_end_callback)
        
        def _on_batch_end_callback(self, trainer):
            """Custom callback to modify training process for distillation"""
            # Here we can add custom logging for the distillation process
            if hasattr(trainer, 'distill_loss_dict'):
                # Log distillation loss components
                for name, value in trainer.distill_loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        trainer.metrics[name] = value.item()
        
        def train(self):
            """Start training with knowledge distillation"""
            # Modify student model's loss computation
            original_loss_fn = self.student_model.model.loss
            
            # Create a new loss function that incorporates distillation
            def distillation_loss_fn(preds, batch):
                # Get original loss and predictions
                loss, loss_items = original_loss_fn(preds, batch)
                
                # If we're in validation mode, just return the original loss
                if not self.student_model.model.training:
                    return loss, loss_items
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_preds = self.teacher_model.model(batch["img"])
                
                # Apply distillation
                inputs = batch["img"]
                targets = batch["cls"] if "cls" in batch else None
                
                # Create output dictionaries for distillation function
                student_outputs = {"pred": preds[0], "loss": loss}
                teacher_outputs = {"pred": teacher_preds[0]}
                
                # Compute distillation loss
                distill_loss, loss_dict = apply_distillation(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    targets=targets,
                    temperature=self.args.temperature,
                    alpha=self.args.alpha
                )
                
                # Store the loss dict for logging
                if not hasattr(self.student_model.trainer, 'distill_loss_dict'):
                    self.student_model.trainer.distill_loss_dict = {}
                self.student_model.trainer.distill_loss_dict = loss_dict
                
                return distill_loss, loss_items
            
            # Replace the loss function
            self.student_model.model.loss = distillation_loss_fn
            
            # Start training
            results = self.student_model.train(
                data=data_yaml_path,
                epochs=self.args.epochs,
                batch=self.args.batch_size,
                imgsz=self.args.img_size,
                device=self.args.device,
                project=self.args.project,
                name=f"{self.args.name}_distill",
                resume=self.args.resume
            )
            
            return results
    
    # Create trainer and start training
    trainer = DistillationTrainer(student_model, teacher_model, args)
    results = trainer.train()
    
    return results


def train_standard(args):
    """Train the model without knowledge distillation"""
    # Prepare data config
    data_yaml_path = prepare_data_yaml(args.data_path)
    
    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Train the model using standard YOLO training
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    return results


def main():
    args = parse_args()
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Dataset: {args.data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Knowledge Distillation: {'Enabled' if args.distill else 'Disabled'}")
    
    if args.distill:
        print(f"Teacher Model: {args.teacher_model}")
        print(f"Temperature: {args.temperature}")
        print(f"Alpha: {args.alpha}")
    print("============================\n")
    
    # Check for CUDA availability
    if args.device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = "cpu"
    
    # Train with or without distillation
    if args.distill:
        results = train_with_distillation(args)
    else:
        results = train_standard(args)
    
    print("\n=== Training Completed ===")
    print(f"Results saved to {args.project}/{args.name}")
    return results


if __name__ == "__main__":
    main()
