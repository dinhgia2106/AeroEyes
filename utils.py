"""
Các utility functions hữu ích
"""
import json
from pathlib import Path

def load_annotations(annotations_path):
    """Load annotations từ file JSON"""
    with open(annotations_path, 'r') as f:
        return json.load(f)

def save_annotations(annotations, output_path):
    """Save annotations ra file JSON"""
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

def get_video_info(video_id, annotations):
    """Lấy thông tin về một video từ annotations"""
    for ann in annotations:
        if ann['video_id'] == video_id:
            return ann
    return None

def count_total_bboxes(annotations):
    """Đếm tổng số bboxes trong annotations"""
    total = 0
    for ann in annotations:
        for annotation_set in ann['annotations']:
            total += len(annotation_set['bboxes'])
    return total

def get_unique_frames_count(annotations):
    """Đếm số unique frames có annotations"""
    unique_frames = set()
    for ann in annotations:
        for annotation_set in ann['annotations']:
            for bbox_info in annotation_set['bboxes']:
                unique_frames.add((ann['video_id'], bbox_info['frame']))
    return len(unique_frames)

def analyze_dataset(annotations_path, samples_dir):
    """Analyze dataset và hiển thị statistics"""
    annotations = load_annotations(annotations_path)
    samples_dir = Path(samples_dir)
    
    print("="*60)
    print("📊 Dataset Statistics")
    print("="*60)
    
    print(f"\n📹 Videos: {len(annotations)}")
    print(f"🔲 Total bboxes: {count_total_bboxes(annotations)}")
    print(f"🎯 Unique frames with objects: {get_unique_frames_count(annotations)}")
    
    # Count by video
    print(f"\n📋 Detections per video:")
    for ann in annotations[:10]:  # Show first 10
        bbox_count = sum(len(aset['bboxes']) for aset in ann['annotations'])
        detections_count = len(ann['annotations'])
        print(f"  {ann['video_id']}: {bbox_count} bboxes in {detections_count} intervals")
    
    if len(annotations) > 10:
        print(f"  ... and {len(annotations) - 10} more videos")
    
    # Check video files
    missing_videos = []
    for ann in annotations:
        video_path = samples_dir / ann['video_id'] / "drone_video.mp4"
        if not video_path.exists():
            missing_videos.append(ann['video_id'])
    
    if missing_videos:
        print(f"\n⚠️  Missing videos: {len(missing_videos)}")
        for vid in missing_videos[:5]:
            print(f"  - {vid}")
    else:
        print(f"\n✅ All video files exist")
    
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset")
    parser.add_argument("--annotations", default="observing/train/annotations/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--samples", default="observing/train/samples",
                       help="Path to samples directory")
    
    args = parser.parse_args()
    
    analyze_dataset(args.annotations, args.samples)

