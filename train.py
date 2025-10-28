"""
Script để train YOLOv8 model cho object detection
"""
from ultralytics import YOLO
from pathlib import Path
import torch

def detect_device():
    """Auto detect available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = '0'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = 'cpu'
        print("No GPU available, using CPU")
        return device

def train_model(data_yaml, epochs=100, batch_size=16, imgsz=640, device=None):
    """
    Train YOLOv8n model
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        device: Device to use ('0' for GPU, 'cpu' for CPU, None for auto-detect)
    """
    # Auto detect device if not specified
    if device is None:
        device = detect_device()
    
    # Load YOLOv8n model (nano - nhẹ nhất, phù hợp cho Jetson)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        val=True,
        project='runs/detect',
        name='object_detection',
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: runs/detect/object_detection/weights/best.pt")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--data", default="dataset_yolo/data.yaml",
                       help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--device", default=None,
                       help="Device to use ('0' for GPU, 'cpu' for CPU, None for auto-detect)")
    
    args = parser.parse_args()
    
    # Check if data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        print("Please run prepare_dataset.py first to create the dataset.")
        exit(1)
    
    train_model(
        data_yaml=str(data_path),
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
