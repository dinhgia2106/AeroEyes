"""
Script để visualize annotations trên video
"""
import cv2
import json
import argparse
from pathlib import Path

def draw_bboxes_on_frame(frame, bboxes):
    """Vẽ bounding boxes lên frame"""
    for bbox in bboxes:
        x1 = bbox['x1']
        y1 = bbox['y1']
        x2 = bbox['x2']
        y2 = bbox['y2']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

def visualize_video(video_path, annotations):
    """Visualize một video với annotations"""
    cap = cv2.VideoCapture(str(video_path))
    
    # Tạo dictionary cho bboxes theo frame
    bbox_dict = {}
    for annotation_set in annotations['annotations']:
        for bbox_info in annotation_set['bboxes']:
            frame_num = bbox_info['frame']
            if frame_num not in bbox_dict:
                bbox_dict[frame_num] = []
            bbox_dict[frame_num].append(bbox_info)
    
    frame_number = 0
    
    print(f"\nĐang hiển thị video: {video_path.name}")
    print("Nhấn 'q' để quit, 'p' để pause, space để step")
    print("="*60)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        # Hiển thị thông tin frame
        if frame_number in bbox_dict:
            bboxes = bbox_dict[frame_number]
            frame = draw_bboxes_on_frame(frame.copy(), bboxes)
            
            # Add text
            text = f"Frame {frame_number}: {len(bboxes)} objects"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            text = f"Frame {frame_number}: No objects"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize để hiển thị
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        cv2.imshow('Video with Annotations', frame)
        
        key = cv2.waitKey(0 if paused else 30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            if not paused:
                paused = True
            else:
                frame_number += 1
                continue
        
        if not paused:
            frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize annotations on videos")
    parser.add_argument("--annotations", default="observing/train/annotations/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--samples", default="observing/train/samples",
                       help="Path to samples directory")
    parser.add_argument("--video-id", default=None,
                       help="Specific video_id to visualize (optional)")
    
    args = parser.parse_args()
    
    # Load annotations
    with open(args.annotations, 'r') as f:
        all_annotations = json.load(f)
    
    # Filter video nếu cần
    if args.video_id:
        all_annotations = [ann for ann in all_annotations if ann['video_id'] == args.video_id]
    
    # Visualize từng video
    samples_dir = Path(args.samples)
    
    for ann in all_annotations:
        video_id = ann['video_id']
        video_path = samples_dir / video_id / "drone_video.mp4"
        
        if not video_path.exists():
            print(f"Warning: {video_path} not found")
            continue
        
        visualize_video(video_path, ann)
        
        # Hỏi xem có tiếp tục không
        if len(all_annotations) > 1:
            continue_video = input(f"\nXem video tiếp theo? (y/n): ")
            if continue_video.lower() != 'y':
                break

if __name__ == "__main__":
    main()

