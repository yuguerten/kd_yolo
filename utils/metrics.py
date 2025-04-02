import csv
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

class MetricsTracker:
    """
    Custom metrics tracking class for YOLO training.
    Handles both standard training and knowledge distillation metrics.
    """
    def __init__(self, output_dir, is_distillation=False):
        """
        Initialize metrics tracker with output directory
        
        Args:
            output_dir (Path): Directory to save metrics
            is_distillation (bool): Whether training with knowledge distillation
        """
        self.output_dir = Path(output_dir)
        self.is_distillation = is_distillation
        self.csv_path = self.output_dir / "results.csv"
        self.current_metrics = {}
        self.best_map50 = 0
        self.epoch_start_time = None
        
        # Initialize CSV file with header
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with appropriate header"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Define columns based on training type
            columns = ['epoch', 'time']
            
            # Add training loss metrics
            columns.extend(['train/box_loss', 'train/cls_loss', 'train/dfl_loss'])
            
            # Add validation metrics
            columns.extend([
                'metrics/precision(B)', 'metrics/recall(B)', 
                'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
            ])
            
            # Add validation loss metrics
            columns.extend(['val/box_loss', 'val/cls_loss', 'val/dfl_loss'])
            
            # Add learning rate columns
            columns.extend(['lr/pg0', 'lr/pg1', 'lr/pg2'])
            
            # Add distillation-specific metrics if needed
            if self.is_distillation:
                columns.extend(['kd/student_loss', 'kd/teacher_loss', 'kd/kd_loss', 'kd/total_loss'])
            
            writer.writerow(columns)
    
    def start_epoch(self):
        """Start timing for a new epoch"""
        self.epoch_start_time = time.time()
        self.current_metrics = {}
        
    def update_training_metrics(self, loss_dict):
        """Update training metrics from a single batch"""
        # Initialize metrics if not present
        for k, v in loss_dict.items():
            if k not in self.current_metrics:
                self.current_metrics[k] = []
            self.current_metrics[k].append(v)
    
    def update_validation_metrics(self, metrics_dict):
        """Update validation metrics"""
        for k, v in metrics_dict.items():
            self.current_metrics[k] = v
    
    def end_epoch(self, epoch, learning_rates):
        """
        End epoch, compute averages, and log to CSV
        
        Args:
            epoch (int): Current epoch number
            learning_rates (list): List of learning rates for each parameter group
        """
        epoch_time = time.time() - self.epoch_start_time
        
        # Compute average for metrics that are lists
        for k in self.current_metrics:
            if isinstance(self.current_metrics[k], list):
                self.current_metrics[k] = np.mean(self.current_metrics[k])
        
        # Construct row for CSV
        row = [epoch, epoch_time]
        
        # Training loss metrics (compute averages if lists)
        row.extend([
            self.current_metrics.get('train/box_loss', 0),
            self.current_metrics.get('train/cls_loss', 0),
            self.current_metrics.get('train/dfl_loss', 0),
        ])
        
        # Validation metrics
        row.extend([
            self.current_metrics.get('metrics/precision(B)', 0),
            self.current_metrics.get('metrics/recall(B)', 0),
            self.current_metrics.get('metrics/mAP50(B)', 0),
            self.current_metrics.get('metrics/mAP50-95(B)', 0),
        ])
        
        # Validation loss metrics
        row.extend([
            self.current_metrics.get('val/box_loss', 0),
            self.current_metrics.get('val/cls_loss', 0),
            self.current_metrics.get('val/dfl_loss', 0),
        ])
        
        # Learning rates
        row.extend(learning_rates)
        
        # Distillation metrics
        if self.is_distillation:
            row.extend([
                self.current_metrics.get('kd/student_loss', 0),
                self.current_metrics.get('kd/teacher_loss', 0),
                self.current_metrics.get('kd/kd_loss', 0),
                self.current_metrics.get('kd/total_loss', 0),
            ])
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Check for best mAP
        current_map = self.current_metrics.get('metrics/mAP50(B)', 0)
        if current_map > self.best_map50:
            self.best_map50 = current_map
            return True  # Indicates new best model
        return False
    
    def print_epoch_summary(self, epoch, total_epochs):
        """Print a summary of the epoch metrics to console"""
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{total_epochs} completed in {self.current_metrics.get('time', 0):.2f}s")
        
        # Print training metrics
        print(f"Training: box_loss={self.current_metrics.get('train/box_loss', 0):.4f}, "
              f"cls_loss={self.current_metrics.get('train/cls_loss', 0):.4f}, "
              f"dfl_loss={self.current_metrics.get('train/dfl_loss', 0):.4f}")
        
        # Print validation metrics
        print(f"Validation: mAP50={self.current_metrics.get('metrics/mAP50(B)', 0):.4f}, "
              f"mAP50-95={self.current_metrics.get('metrics/mAP50-95(B)', 0):.4f}, "
              f"precision={self.current_metrics.get('metrics/precision(B)', 0):.4f}, "
              f"recall={self.current_metrics.get('metrics/recall(B)', 0):.4f}")
        
        # Print distillation metrics if applicable
        if self.is_distillation:
            print(f"Distillation: student_loss={self.current_metrics.get('kd/student_loss', 0):.4f}, "
                  f"kd_loss={self.current_metrics.get('kd/kd_loss', 0):.4f}, "
                  f"total_loss={self.current_metrics.get('kd/total_loss', 0):.4f}")
        
        print(f"{'='*50}")
