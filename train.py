import os
import argparse
import yaml
from pathlib import Path
import torch
import csv
from ultralytics import YOLO
from utils.distill import apply_distillation
from utils.data_utils import create_data_yaml


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


def train_with_distillation(args):
    """Train the model with knowledge distillation"""
    # Check if teacher model is specified
    if not args.teacher_model:
        raise ValueError("Teacher model path must be specified for distillation.")
    
    # Prepare data config
    data_yaml_path = create_data_yaml(args.data_path)
    
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
            self.student_model.add_callback("on_train_epoch_end", self._on_epoch_end_callback)
            
            # Create CSV file for logging metrics
            self.csv_path = Path(self.args.project) / f"{self.args.name}_distill" / "metrics.csv"
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            # Initialize CSV with headers
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 
                                'Box Loss', 'Cls Loss', 'Obj Loss', 'KD Loss', 
                                'Student Loss', 'Teacher mAP50', 'Teacher mAP50-95'])
        
        def _on_batch_end_callback(self, trainer):
            """Custom callback to modify training process for distillation"""
            # Here we can add custom logging for the distillation process
            if hasattr(trainer, 'distill_loss_dict'):
                # Log distillation loss components
                for name, value in trainer.distill_loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        trainer.metrics[name] = value.item()
        
        def _on_epoch_end_callback(self, trainer):
            """Save metrics at the end of each epoch"""
            # Extract metrics
            epoch = trainer.epoch
            metrics = trainer.metrics
            
            # Calculate teacher model performance on validation set
            val_loader = trainer.validator.dataloader
            teacher_metrics = self._evaluate_teacher(val_loader)
            
            # Prepare metric values for CSV
            row = [
                epoch,
                metrics.get('metrics/mAP50(B)', 0),
                metrics.get('metrics/mAP50-95(B)', 0),
                metrics.get('metrics/precision(B)', 0),
                metrics.get('metrics/recall(B)', 0),
                metrics.get('train/box_loss', 0),
                metrics.get('train/cls_loss', 0),
                metrics.get('train/dfl_loss', 0),  # Object loss
                metrics.get('kd_loss', 0),
                metrics.get('student_loss', 0),
                teacher_metrics.get('mAP50', 0),
                teacher_metrics.get('mAP50-95', 0)
            ]
            
            # Save to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"Epoch {epoch}: Student mAP50: {row[1]:.4f}, Teacher mAP50: {row[9]:.4f}, KD Loss: {row[7]:.4f}")
        
        def _evaluate_teacher(self, dataloader):
            """Evaluate teacher model on validation set"""
            metrics = {}
            try:
                # Run validation on teacher model
                results = self.teacher_model.val(data=dataloader)
                # Extract metrics
                metrics['mAP50'] = results.box.map50
                metrics['mAP50-95'] = results.box.map
            except Exception as e:
                print(f"Error evaluating teacher model: {e}")
            
            return metrics
        
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
            
            print(f"Training completed. Metrics saved to {self.csv_path}")
            return results
    
    # Create trainer and start training
    trainer = DistillationTrainer(student_model, teacher_model, args)
    results = trainer.train()
    
    return results


def train_standard(args):
    """Train the model without knowledge distillation"""
    # Prepare data config
    data_yaml_path = create_data_yaml(args.data_path)
    
    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Setup CSV logging for standard training
    csv_path = Path(args.project) / args.name / "metrics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Initialize CSV with headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 
                        'Box Loss', 'Cls Loss', 'Obj Loss'])
    
    # Add epoch end callback for logging
    def on_epoch_end(trainer):
        metrics = trainer.metrics
        epoch = trainer.epoch
        
        # Prepare row for CSV
        row = [
            epoch,
            metrics.get('metrics/mAP50(B)', 0),
            metrics.get('metrics/mAP50-95(B)', 0),
            metrics.get('metrics/precision(B)', 0),
            metrics.get('metrics/recall(B)', 0),
            metrics.get('train/box_loss', 0),
            metrics.get('train/cls_loss', 0),
            metrics.get('train/dfl_loss', 0)  # Object loss
        ]
        
        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    # Add the callback
    model.add_callback("on_train_epoch_end", on_epoch_end)
    
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
    
    print(f"Training completed. Metrics saved to {csv_path}")
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
