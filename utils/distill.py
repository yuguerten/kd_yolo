import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_distillation_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute the knowledge distillation loss combining soft and hard target losses.
    
    Args:
        student_outputs: Dictionary containing student model outputs
        teacher_outputs: Dictionary containing teacher model outputs
        targets: Ground truth targets
        temperature: Temperature parameter to soften probability distributions
        alpha: Weight balancing distillation loss vs. task loss
        
    Returns:
        total_loss: Combined distillation loss
        loss_dict: Dictionary containing individual loss components
    """
    # Extract logits from model outputs
    student_logits = student_outputs['logits'] if 'logits' in student_outputs else student_outputs['pred']
    teacher_logits = teacher_outputs['logits'] if 'logits' in teacher_outputs else teacher_outputs['pred']
    
    # Original hard target loss - this depends on YOLOv8's loss structure
    # We assume the student model already computed this
    hard_target_loss = student_outputs['loss'] if 'loss' in student_outputs else 0
    
    # Compute soft target loss (distillation loss)
    # Apply temperature scaling
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence loss
    soft_target_loss = F.kl_div(
        soft_student, 
        soft_teacher,
        reduction='batchmean',
        log_target=False
    ) * (temperature ** 2)
    
    # Combine losses using alpha as the weighting factor
    total_loss = alpha * soft_target_loss + (1 - alpha) * hard_target_loss
    
    # Create a dictionary of losses for monitoring
    loss_dict = {
        'total_loss': total_loss,
        'soft_target_loss': soft_target_loss,
        'hard_target_loss': hard_target_loss if isinstance(hard_target_loss, torch.Tensor) else torch.tensor(0.0)
    }
    
    return total_loss, loss_dict


def apply_distillation(
    student_model,
    teacher_model,
    inputs: torch.Tensor,
    targets=None,
    temperature: float = 4.0,
    alpha: float = 0.5
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply knowledge distillation during training.
    
    Args:
        student_model: The student model being trained
        teacher_model: The teacher model for knowledge transfer
        inputs: Input tensor
        targets: Ground truth targets
        temperature: Temperature parameter to soften probability distributions
        alpha: Weight balancing distillation loss vs task loss
        
    Returns:
        loss: Total loss value
        loss_dict: Dictionary with individual loss components
    """
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    # Forward pass through teacher model (no gradient tracking needed)
    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)
    
    # Forward pass through student model
    student_outputs = student_model(inputs)
    
    # Compute distillation loss
    loss, loss_dict = compute_distillation_loss(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        targets=targets,
        temperature=temperature,
        alpha=alpha
    )
    
    return loss, loss_dict
