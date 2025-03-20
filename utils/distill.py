import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        """
        Initialize the distillation loss.
        
        Args:
            temperature (float): Temperature for softening the teacher's output distribution
            alpha (float): Weight balancing the distillation loss and original loss
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_outputs, teacher_outputs, original_loss):
        """
        Calculate the distillation loss.
        
        Args:
            student_outputs: Output from the student model
            teacher_outputs: Output from the teacher model
            original_loss: Original detection loss
            
        Returns:
            Combined loss with distillation
        """
        # For YOLOv8, we focus on distilling the detection head outputs
        distill_loss = 0
        
        # Apply distillation to each output layer
        for i in range(len(student_outputs)):
            student_out = student_outputs[i]
            teacher_out = teacher_outputs[i].detach()  # No gradient for teacher
            
            # Apply temperature scaling
            soft_student = torch.sigmoid(student_out / self.temperature)
            soft_teacher = torch.sigmoid(teacher_out / self.temperature)
            
            # Calculate KL divergence loss for classification part (first 80 channels in YOLOv8)
            cls_channels = min(student_out.shape[1] - 5, 80)  # Classification channels
            distill_loss += F.mse_loss(
                soft_student[..., :cls_channels],
                soft_teacher[..., :cls_channels]
            ) * (self.temperature ** 2)
        
        # Combine losses: distillation loss and original detection loss
        return self.alpha * distill_loss + (1 - self.alpha) * original_loss


class DistillationTrainer:
    def __init__(self, teacher_model_path, student_model, temperature=4.0, alpha=0.5):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model_path (str): Path to the teacher model
            student_model: Student model to be trained
            temperature (float): Temperature for softening the outputs
            alpha (float): Weight for the distillation loss
        """
        self.teacher_model = self.load_teacher(teacher_model_path)
        self.student_model = student_model
        self.distill_loss = DistillationLoss(temperature, alpha)
        
    def load_teacher(self, model_path):
        """Load the teacher model and set it to evaluation mode."""
        teacher = DetectionModel(model_path)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def forward_teacher(self, imgs):
        """Get teacher predictions for the input images."""
        with torch.no_grad():
            return self.teacher_model(imgs)
    
    def distill_batch(self, batch, compute_loss):
        """
        Apply knowledge distillation to a batch.
        
        Args:
            batch: Input batch (images and targets)
            compute_loss: Loss computation function from YOLOv8
            
        Returns:
            Combined loss with distillation
        """
        imgs, targets = batch
        
        # Get predictions from teacher model
        teacher_outputs = self.forward_teacher(imgs)
        
        # Get predictions and loss from student model
        student_outputs, student_loss = compute_loss(self.student_model(imgs), targets)
        
        # Calculate distillation loss
        combined_loss = self.distill_loss(student_outputs, teacher_outputs, student_loss)
        
        return combined_loss

def apply_distillation_training(student_model, teacher_model_path, data_path,
                               epochs, batch_size, img_size, device, project,
                               name, temperature, alpha, resume=False):
    """
    Apply knowledge distillation during training.
    
    Args:
        student_model: The student YOLO model
        teacher_model_path: Path to the teacher model
        data_path: Path to the data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Training device
        project: Project directory
        name: Run name
        temperature: Temperature for distillation
        alpha: Weight for distillation loss
        resume: Whether to resume training from a checkpoint
    """
    # Import the necessary YOLO modules
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    
    class DistillDetectionTrainer(DetectionTrainer):
        def __init__(self, teacher_model_path, temperature, alpha, **kwargs):
            super().__init__(**kwargs)
            self.teacher = self.load_teacher(teacher_model_path)
            self.temperature = temperature
            self.alpha = alpha
            
        def load_teacher(self, model_path):
            """Load the teacher model."""
            from ultralytics.nn.tasks import DetectionModel
            import torch
            
            teacher = DetectionModel(model_path)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            return teacher
            
        def get_teacher_preds(self, batch):
            """Get predictions from the teacher model."""
            with torch.no_grad():
                imgs = batch[0].to(self.device)
                return self.teacher(imgs)
        
        def _do_train_step(self, batch, batch_idx):
            """Custom training step with distillation."""
            imgs, targets = batch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_preds = self.teacher(imgs)
                
            # Forward pass with student model
            preds = self.model(imgs)
            
            # Calculate original loss
            loss, loss_items = self.criterion(preds, targets.to(self.device))
            
            # Apply distillation loss to the outputs
            if self.epoch >= 0:  # Apply distillation from the beginning
                distill_loss = torch.tensor(0.0, device=self.device)
                
                # Apply distillation to each detection layer
                for i, (student_out, teacher_out) in enumerate(zip(preds, teacher_preds)):
                    # Apply temperature scaling
                    soft_student = torch.sigmoid(student_out / self.temperature)
                    soft_teacher = torch.sigmoid(teacher_out / self.temperature)
                    
                    # For classification outputs
                    cls_channels = min(student_out.shape[1] - 5, 80)  # YOLO has cls channels first
                    
                    distill_loss += torch.nn.functional.mse_loss(
                        soft_student[..., :cls_channels],
                        soft_teacher[..., :cls_channels]
                    ) * (self.temperature ** 2)
                
                # Combine losses
                loss = (1 - self.alpha) * loss + self.alpha * distill_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss, loss_items
            
    # Set up trainer with distillation
    trainer = DistillDetectionTrainer(
        teacher_model_path=teacher_model_path,
        temperature=temperature,
        alpha=alpha,
        cfg=student_model.model.yaml,
        overrides={
            'data': data_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'project': project,
            'name': name,
            'resume': resume
        }
    )
    
    # Train the model
    trainer.train()
    
    # Update the student model with the trained weights
    student_model.model = trainer.model
