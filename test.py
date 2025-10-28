"""
Script để inference và tạo predictions từ trained model
"""
from ultralytics import YOLO
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

def inference_video(model_path, video_path, conf_threshold=0.5):
    """
    Run inference trên một video
    
    Args:
        model_path: Path đến trained model
        video_path: Path đến video
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections theo format: [{"frame": num, "x1": x, "y1": y, "x2": x, "y2": y}, ...]
    """
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_detections = []
    frame_number = 0
    
    with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = model.predict(
                frame,
                conf=conf_threshold,
                verbose=False,
                imgsz=640
            )
            
            # Parse results
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get coordinates trong format x1, y1, x2, y2
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            "frame": frame_number,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                        all_detections.append(detection)
            
            frame_number += 1
            pbar.update(1)
    
    cap.release()
    
    # Group consecutive detections
    grouped_detections = group_consecutive_detections(all_detections)
    
    return grouped_detections

def group_consecutive_detections(detections):
    """
    Group các detections liên tiếp thành các intervals
    
    Returns: list of bboxes lists theo format yêu cầu
    """
    if not detections:
        return []
    
    # Sort by frame number
    detections = sorted(detections, key=lambda x: x['frame'])
    
    grouped = []
    current_group = []
    
    for i, det in enumerate(detections):
        if not current_group:
            current_group.append(det)
        else:
            # Check if consecutive (within 10 frames)
            last_frame = current_group[-1]['frame']
            if det['frame'] - last_frame <= 10:
                current_group.append(det)
            else:
                # New group
                if current_group:
                    grouped.append({"bboxes": current_group})
                current_group = [det]
    
    # Add last group
    if current_group:
        grouped.append({"bboxes": current_group})
    
    return grouped

def generate_predictions(model_path, test_samples_dir, output_file="predictions.json"):
    """
    Generate predictions cho tất cả videos trong test samples
    
    Args:
        model_path: Path đến trained model
        test_samples_dir: Directory chứa test samples
        output_file: Output file cho predictions
    """
    samples_dir = Path(test_samples_dir)
    
    if not samples_dir.exists():
        print(f"Error: {samples_dir} not found!")
        return
    
    # Get all video samples
    video_samples = sorted(list(samples_dir.glob("*")))
    
    print(f"Found {len(video_samples)} video samples")
    
    all_predictions = []
    
    for sample_dir in tqdm(video_samples, desc="Processing test samples"):
        video_path = sample_dir / "drone_video.mp4"
        
        if not video_path.exists():
            print(f"Warning: {video_path} not found")
            continue
        
        video_id = sample_dir.name
        
        # Run inference
        detections = inference_video(model_path, video_path)
        
        # Format theo yêu cầu
        prediction = {
            "video_id": video_id,
            "detections": detections if detections else []
        }
        
        all_predictions.append(prediction)
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")
    
    # Print summary
    total_with_detections = sum(1 for pred in all_predictions if pred['detections'])
    print(f"Summary:")
    print(f"  Total videos: {len(all_predictions)}")
    print(f"  Videos with detections: {total_with_detections}")
    print(f"  Videos without detections: {len(all_predictions) - total_with_detections}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions from trained model")
    parser.add_argument("--model", default="runs/detect/object_detection/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--samples", default="public_test/public_test/samples",
                       help="Path to test samples directory")
    parser.add_argument("--output", default="predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found!")
        print("Please train the model first using train.py")
        exit(1)
    
    generate_predictions(
        model_path=str(model_path),
        test_samples_dir=args.samples,
        output_file=args.output
    )

