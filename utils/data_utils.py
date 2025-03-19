import os
import yaml
from pathlib import Path

def create_data_yaml(dataset_path):
    """
    Create a data.yaml file for YOLOv8 training
    """
    dataset_path = Path(dataset_path)
    
    # Find class names from the dataset
    class_names = []
    labels_dir = dataset_path / 'train' / 'labels'
    
    if not labels_dir.exists():
        labels_dir = dataset_path / 'labels'
    
    if labels_dir.exists():
        # Try to find classes.txt or similar
        classes_file = dataset_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            # Try to infer classes from label files
            print("Looking for label files to infer class names...")
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            while len(class_names) <= class_id:
                                class_names.append(f"class_{len(class_names)}")
    
    if not class_names:
        print("Warning: No class names found. Using 'class_0' as default.")
        class_names = ['class_0']
    
    # Create the data.yaml content
    data = {
        'path': str(dataset_path),
        'train': str(dataset_path / 'train' / 'images' if (dataset_path / 'train' / 'images').exists() else dataset_path / 'images'),
        'val': str(dataset_path / 'val' / 'images' if (dataset_path / 'val' / 'images').exists() else dataset_path / 'images'),
        'test': str(dataset_path / 'test' / 'images' if (dataset_path / 'test' / 'images').exists() else ''),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    # Write the data.yaml file
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

def prepare_colab_dataset(gdrive_path, dataset_name):
    """
    Prepare a dataset from Google Drive for training
    """
    from google.colab import drive
    drive.mount('/content/drive')
    
    dataset_path = Path(f'/content/drive/MyDrive/{gdrive_path}/{dataset_name}')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Dataset found at {dataset_path}")
    create_data_yaml(dataset_path)
    
    return dataset_path
