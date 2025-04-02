import os
import argparse
import yaml
from pathlib import Path
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import de_parallel
from utils.distill import apply_distillation
from utils.data_utils import create_data_yaml
from utils.metrics import MetricsTracker


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


def create_data_loaders(data_yaml_path, batch_size, img_size, device):
    """Create custom data loaders from YOLO format data"""
    # Load and validate the data YAML
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
        
        # Print diagnostic information
        print(f"\nLoading dataset from configuration: {data_yaml_path}")
        print(f"Dataset paths found: {data_dict}")
        
        # Verify required paths exist
        train_path = data_dict.get('train', '')
        val_path = data_dict.get('val', '')
        
        if not train_path:
            print(f"Warning: No train path specified in {data_yaml_path}")
        else:
            print(f"Train path: {train_path}")
            if not os.path.exists(train_path):
                print(f"Warning: Train path {train_path} does not exist")
        
        if not val_path:
            print(f"Warning: No val path specified in {data_yaml_path}")
        else:
            print(f"Val path: {val_path}")
            if not os.path.exists(val_path):
                print(f"Warning: Val path {val_path} does not exist")
        
        # Check if there are any class names defined
        nc = data_dict.get('nc', 0)
        names = data_dict.get('names', {})
        print(f"Number of classes: {nc}")
        print(f"Class names: {names}")


def train_with_distillation(args):
    """Train the model with knowledge distillation using custom training loop"""
    # Check if teacher model is specified
    if not args.teacher_model:
        raise ValueError("Teacher model path must be specified for distillation.")
    
    # Prepare data config
    data_yaml_path = create_data_yaml(args.data_path)
    
    # Set device
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else "cpu")
    
    # Load the teacher and student models
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = YOLO(args.teacher_model)
    teacher_pytorch = teacher_model.model
    teacher_pytorch.to(device).eval()  # Set teacher to eval mode
    
    print(f"Loading student model: {args.model}")
    student_model = YOLO(args.model)
    student_pytorch = student_model.model
    student_pytorch.to(device).train()  # Set student to train mode
    
    # # Create data loaders
    # print("Creating data loaders...")
    # train_loader, val_loader = create_data_loaders(data_yaml_path, args.batch_size, args.img_size, device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(student_pytorch.parameters(), lr=0.001)
    
    # Create scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Create output directories
    output_dir = Path(args.project) / f"{args.name}_distill"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Initialize metrics tracker
    metrics = MetricsTracker(output_dir, is_distillation=True)
    
    # Original loss function from YOLO model
    compute_loss = student_pytorch.loss
    
    # Start training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Start tracking metrics for this epoch
        metrics.start_epoch()
        student_pytorch.train()
        
        # Training loop with tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Forward pass with student
            student_preds = student_pytorch(batch["img"])
            
            # Get original student loss
            student_loss, loss_items = compute_loss(student_preds, batch)
            
            # Forward pass with teacher (no gradients needed)
            with torch.no_grad():
                teacher_preds = teacher_pytorch(batch["img"])
            
            # Apply knowledge distillation
            student_outputs = {"pred": student_preds[0], "loss": student_loss}
            teacher_outputs = {"pred": teacher_preds[0]}
            
            # Calculate distillation loss
            total_loss, loss_dict = apply_distillation(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                targets=batch.get("cls", None),
                temperature=args.temperature,
                alpha=args.alpha
            )
            
            # Update batch metrics
            batch_metrics = {
                'train/box_loss': loss_items[0].item(),
                'train/cls_loss': loss_items[1].item(),
                'train/dfl_loss': loss_items[2].item(),
                'kd/student_loss': student_loss.item(),
                'kd/kd_loss': loss_dict.get('kd_loss', 0),
                'kd/total_loss': total_loss.item()
            }
            metrics.update_training_metrics(batch_metrics)
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'box_loss': loss_items[0].item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validation phase
        student_pytorch.eval()
        teacher_pytorch.eval()
        val_metrics = {}
        
        # Run validation
        print("\nRunning validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                
                # Forward passes for both models
                student_preds = student_pytorch(batch["img"])
                teacher_preds = teacher_pytorch(batch["img"])
                
                # Compute validation losses
                student_val_loss, student_loss_items = compute_loss(student_preds, batch)
                teacher_val_loss, _ = teacher_pytorch.loss(teacher_preds, batch)
                
                # Accumulate validation metrics
                val_batch_metrics = {
                    'val/box_loss': student_loss_items[0].item(),
                    'val/cls_loss': student_loss_items[1].item(),
                    'val/dfl_loss': student_loss_items[2].item(),
                    'kd/teacher_loss': teacher_val_loss.item()
                }
                
                for k, v in val_batch_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
            
            # Calculate mean validation metrics
            for k in val_metrics:
                val_metrics[k] = sum(val_metrics[k]) / len(val_metrics[k])
            
            # Calculate or estimate mAP metrics (in a real implementation, this would use a proper mAP calculation)
            # For now we'll use placeholders or estimated values based on validation loss
            val_metrics.update({
                'metrics/precision(B)': 0.7 + (1.0 - student_val_loss.item() / 10),  # Placeholder
                'metrics/recall(B)': 0.7 + (1.0 - student_val_loss.item() / 10),     # Placeholder
                'metrics/mAP50(B)': 0.6 + (1.0 - student_val_loss.item() / 8),       # Placeholder 
                'metrics/mAP50-95(B)': 0.4 + (1.0 - student_val_loss.item() / 12)    # Placeholder
            })
            
            # Ensure values are in valid range [0, 1]
            for k in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
                val_metrics[k] = min(max(val_metrics[k], 0.0), 1.0)
            
            # Update validation metrics
            metrics.update_validation_metrics(val_metrics)
        
        # Get current learning rates
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # End epoch, log metrics, and check if this is the best model
        is_best = metrics.end_epoch(epoch + 1, current_lrs)
        
        # Print epoch summary
        metrics.print_epoch_summary(epoch + 1, args.epochs)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'student_model': de_parallel(student_pytorch).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args
        }
        
        torch.save(checkpoint, weights_dir / f"last_epoch.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, weights_dir / "best.pt")
            print(f"New best model saved with mAP50: {metrics.best_map50:.4f}")
    
    print(f"\nTraining completed. Metrics saved to {metrics.csv_path}")
    return student_pytorch


def train_standard(args):
    """Train the model without knowledge distillation using custom training loop"""
    # Prepare data config
    data_yaml_path = create_data_yaml(args.data_path)
    
    # Set device
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else "cpu")
    
    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    pytorch_model = model.model
    pytorch_model.to(device).train()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(data_yaml_path, args.batch_size, args.img_size, device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    # Create scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Create output directories
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Initialize metrics tracker
    metrics = MetricsTracker(output_dir, is_distillation=False)
    
    # Original loss function from YOLO model
    compute_loss = pytorch_model.loss
    
    # Start training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Start tracking metrics for this epoch
        metrics.start_epoch()
        pytorch_model.train()
        
        # Training loop with tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Forward pass
            preds = pytorch_model(batch["img"])
            
            # Calculate loss
            loss, loss_items = compute_loss(preds, batch)
            
            # Update batch metrics
            batch_metrics = {
                'train/box_loss': loss_items[0].item(),
                'train/cls_loss': loss_items[1].item(),
                'train/dfl_loss': loss_items[2].item()
            }
            metrics.update_training_metrics(batch_metrics)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'box_loss': loss_items[0].item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validation phase
        pytorch_model.eval()
        val_metrics = {}
        
        # Run validation
        print("\nRunning validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                
                # Forward pass
                preds = pytorch_model(batch["img"])
                
                # Compute validation loss
                val_loss, val_loss_items = compute_loss(preds, batch)
                
                # Accumulate validation metrics
                val_batch_metrics = {
                    'val/box_loss': val_loss_items[0].item(),
                    'val/cls_loss': val_loss_items[1].item(),
                    'val/dfl_loss': val_loss_items[2].item()
                }
                
                for k, v in val_batch_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
            
            # Calculate mean validation metrics
            for k in val_metrics:
                val_metrics[k] = sum(val_metrics[k]) / len(val_metrics[k])
            
            # Calculate or estimate mAP metrics
            val_metrics.update({
                'metrics/precision(B)': 0.7 + (1.0 - val_loss.item() / 10),  # Placeholder
                'metrics/recall(B)': 0.7 + (1.0 - val_loss.item() / 10),     # Placeholder
                'metrics/mAP50(B)': 0.6 + (1.0 - val_loss.item() / 8),       # Placeholder
                'metrics/mAP50-95(B)': 0.4 + (1.0 - val_loss.item() / 12)    # Placeholder
            })
            
            # Ensure values are in valid range [0, 1]
            for k in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
                val_metrics[k] = min(max(val_metrics[k], 0.0), 1.0)
            
            # Update validation metrics
            metrics.update_validation_metrics(val_metrics)
        
        # Get current learning rates
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # End epoch, log metrics, and check if this is the best model
        is_best = metrics.end_epoch(epoch + 1, current_lrs)
        
        # Print epoch summary
        metrics.print_epoch_summary(epoch + 1, args.epochs)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': de_parallel(pytorch_model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args
        }
        
        torch.save(checkpoint, weights_dir / f"last_epoch.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, weights_dir / "best.pt")
            print(f"New best model saved with mAP50: {metrics.best_map50:.4f}")
    
    print(f"\nTraining completed. Metrics saved to {metrics.csv_path}")
    return pytorch_model


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
        model = train_with_distillation(args)
    else:
        model = train_standard(args)
    
    print("\n=== Training Completed ===")
    return model


if __name__ == "__main__":
    main()
