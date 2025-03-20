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
│   └── data_utils.py   # Dataset preparation utilities
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

## Dataset Format

The dataset should be in YOLO format:
- images/ - Contains all images
- labels/ - Contains corresponding .txt label files
- data.yaml - Dataset configuration (will be created automatically if missing)

## Example Usage

```python
!python train.py --data-path /content/drive/MyDrive/your_dataset \
                 --model yolov8n.pt \
                 --epochs 50 \
                 --batch-size 16 \
                 --img-size 640
```