# YOLOv8 Training Framework

This repository contains code for training YOLOv8 models on custom datasets using Google Colab.

## Setup

1. Fork/clone this repository to your GitHub account
2. Upload your dataset to Google Drive (in YOLO format)
3. Open the `train_yolov8.ipynb` notebook in Google Colab
4. Follow the steps in the notebook to:
   - Clone your repository
   - Install dependencies
   - Train the model
   - Save results

## Directory Structure

```
kd_yolo/
├── train_yolov8.ipynb  # Main Colab notebook
├── train.py            # Training script with argparse
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── data_utils.py   # Dataset preparation utilities
│   └── distill.py      # Knowledge distillation functionality
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Training Arguments

The `train.py` script accepts the following arguments:

- `--data-path`: Path to dataset directory (required)
- `--model`: Model name or path (default: yolov8n.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--device`: Device to use, either a CUDA device ID or "cpu" (default: 0)
- `--project`: Project directory (default: runs/train)
- `--name`: Experiment name (default: exp)
- `--resume`: Resume training from last checkpoint

### Knowledge Distillation Arguments

For training with knowledge distillation:

- `--distill`: Enable knowledge distillation (flag)
- `--teacher-model`: Path to the teacher model (required for distillation)
- `--temperature`: Temperature parameter for distillation (default: 4.0)
- `--alpha`: Weight for balancing distillation and task loss (default: 0.5)

## Dataset Format

The dataset should be in YOLO format:
- images/ - Contains all images
- labels/ - Contains corresponding .txt label files
- data.yaml - Dataset configuration (will be created automatically if missing)

## Knowledge Distillation

Knowledge distillation helps a smaller student model learn from a larger teacher model. This implementation:

1. Uses response-based distillation focused on the output layer
2. Applies temperature scaling to soften probability distributions
3. Balances distillation loss with the original detection loss

### Distillation Implementation

Our knowledge distillation approach combines:

- **Soft Target Loss**: KL divergence between the teacher and student predictions
  - Temperature scaling (T=4.0) softens probability distributions
  - Higher temperature reveals more dark knowledge from teacher
- **Hard Target Loss**: Original task loss from labeled data
- **Combined Loss**: `total_loss = α * soft_target_loss + (1-α) * hard_target_loss`
  - `α` controls the balance between mimicking the teacher and learning from ground truth

For YOLOv8, we apply distillation to the detection outputs, helping the student model learn the nuanced prediction patterns of the larger teacher model.

### Recommended Model Pairs

- Student: YOLOv8n (Nano) - Teacher: YOLOv8l (Large)
- Student: YOLOv8s (Small) - Teacher: YOLOv8x (XLarge)
- Student: YOLOv8m (Medium) - Teacher: YOLOv8x (XLarge)

### Example usage:

```bash
python train.py --data-path /path/to/dataset --model yolov8s.pt \
    --distill --teacher-model yolov8x.pt --temperature 4.0 --alpha 0.5 \
    --epochs 100 --batch-size 16
```

This trains a YOLOv8-Small model (student) with knowledge from a YOLOv8-XLarge model (teacher).