"""
Script để chuẩn bị dataset từ video và annotations cho YOLOv8 training
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, frame_numbers, output_dir):
    """Extract các frames cụ thể từ video"""
    cap = cv2.VideoCapture(video_path)
    frames_saved = []
    
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"frame_{frame_num}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames_saved.append(frame_num)
    
    cap.release()
    return frames_saved

def create_yolo_label(frame_bboxes, img_width, img_height):
    """Convert bbox format từ (x1, y1, x2, y2) sang YOLO format (class, x_center, y_center, width, height) normalized"""
    labels = []
    
    for bbox_info in frame_bboxes:
        x1 = bbox_info['x1']
        y1 = bbox_info['y1']
        x2 = bbox_info['x2']
        y2 = bbox_info['y2']
        
        # Convert to normalized center coordinates
        x_center = ((x1 + x2) / 2.0) / img_width
        y_center = ((y1 + y2) / 2.0) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Class 0 cho object mục tiêu
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return labels

def prepare_dataset(annotations_path, samples_dir, output_dir, use_all_frames=False, sampling_rate=10):
    """
    Chuẩn bị dataset theo format YOLO từ annotations và videos
    
    Args:
        annotations_path: Path đến annotations.json
        samples_dir: Directory chứa tất cả samples
        output_dir: Directory output cho dataset
        use_all_frames: Nếu True, extract tất cả frames, nếu False chỉ extract frames có annotation
        sampling_rate: Nếu use_all_frames=True, lấy frame mỗi N frames
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Tạo output directories
    output_dir = Path(output_dir)
    train_images_dir = output_dir / "images" / "train"
    train_labels_dir = output_dir / "labels" / "train"
    val_images_dir = output_dir / "images" / "val"
    val_labels_dir = output_dir / "labels" / "val"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process mỗi video
    total_videos = len(annotations)
    train_split = 0.8  # 80% train, 20% val
    
    video_idx = 0
    for ann in tqdm(annotations, desc="Processing videos"):
        video_id = ann['video_id']
        video_path = Path(samples_dir) / video_id / "drone_video.mp4"
        
        if not video_path.exists():
            print(f"Warning: {video_path} not found")
            continue
        
        # Open video để lấy dimensions
        cap = cv2.VideoCapture(str(video_path))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Quyết định train hay val
        is_train = video_idx < int(total_videos * train_split)
        
        if is_train:
            images_dir = train_images_dir
            labels_dir = train_labels_dir
        else:
            images_dir = val_images_dir
            labels_dir = val_labels_dir
        
        # Collect all frames that need to be extracted
        frames_to_extract = set()
        
        if use_all_frames:
            # Extract frames với sampling rate
            for frame_num in range(0, total_frames, sampling_rate):
                frames_to_extract.add(frame_num)
        else:
            # Chỉ extract frames có annotations
            for annotation_set in ann['annotations']:
                for bbox_info in annotation_set['bboxes']:
                    frames_to_extract.add(bbox_info['frame'])
        
        # Sort frames
        frames_to_extract = sorted(list(frames_to_extract))
        
        # Extract frames từ video
        cap = cv2.VideoCapture(str(video_path))
        
        # Tạo dictionary để lưu bboxes theo frame number
        bbox_dict = {}
        for annotation_set in ann['annotations']:
            for bbox_info in annotation_set['bboxes']:
                frame_num = bbox_info['frame']
                if frame_num not in bbox_dict:
                    bbox_dict[frame_num] = []
                bbox_dict[frame_num].append(bbox_info)
        
        # Extract và save frames
        for frame_num in tqdm(frames_to_extract, desc=f"  {video_id}", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Save image
            img_filename = f"{video_id}_frame_{frame_num}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), frame)
            
            # Create label file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = labels_dir / label_filename
            
            # Get bboxes cho frame này
            if frame_num in bbox_dict:
                labels = create_yolo_label(bbox_dict[frame_num], img_width, img_height)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
            else:
                # Frame không có object, tạo empty file
                with open(label_path, 'w') as f:
                    pass
        
        cap.release()
        video_idx += 1
    
    # Tạo data.yaml cho YOLOv8
    data_yaml = f"""
path: {output_dir.absolute()}
train: images/train
val: images/val

names:
  0: target_object

nc: 1
"""
    
    with open(output_dir / "data.yaml", 'w') as f:
        f.write(data_yaml)
    
    print(f"\nDataset đã được chuẩn bị tại: {output_dir}")
    print(f"Train images: {len(list(train_images_dir.glob('*.jpg')))}")
    print(f"Val images: {len(list(val_images_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training")
    parser.add_argument("--annotations", default="observing/train/annotations/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--samples", default="observing/train/samples",
                       help="Path to samples directory")
    parser.add_argument("--output", default="dataset_yolo",
                       help="Output directory for dataset")
    parser.add_argument("--use-all-frames", action="store_true",
                       help="Extract all frames instead of only annotated frames")
    parser.add_argument("--sampling-rate", type=int, default=10,
                       help="Frame sampling rate when using --use-all-frames")
    
    args = parser.parse_args()
    
    prepare_dataset(
        annotations_path=args.annotations,
        samples_dir=args.samples,
        output_dir=args.output,
        use_all_frames=args.use_all_frames,
        sampling_rate=args.sampling_rate
    )

