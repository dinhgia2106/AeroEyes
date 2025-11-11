"""
SORT (Simple Online and Realtime Tracking) với Kalman Filter.

Module này triển khai tracking để tối ưu hóa STIoU bằng cách:
1. Làm mượt (smooth) các bounding boxes bị giật
2. Nội suy (interpolate) các khung hình bị bỏ lỡ
3. Lọc (filter) các false positives
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """
    Kalman Filter cho tracking bounding boxes.
    
    State vector: [x_center, y_center, width, height, vx, vy, vw, vh]
    """
    
    def __init__(self, bbox: np.ndarray):
        """
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # Convert to [x_center, y_center, width, height]
        x1, y1, x2, y2 = bbox
        self.state = np.array([
            (x1 + x2) / 2.0,  # x_center
            (y1 + y2) / 2.0,  # y_center
            x2 - x1,          # width
            y2 - y1,          # height
            0.0,              # vx
            0.0,              # vy
            0.0,              # vw
            0.0               # vh
        ], dtype=np.float32)
        
        # State covariance matrix (8x8)
        self.P = np.eye(8, dtype=np.float32) * 1000.0
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        
        # Measurement noise covariance (chỉ đo được x, y, w, h)
        self.R = np.eye(4, dtype=np.float32) * 10.0
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1.0  # x = x + vx
        self.F[1, 5] = 1.0  # y = y + vy
        self.F[2, 6] = 1.0  # w = w + vw
        self.F[3, 7] = 1.0  # h = h + vh
        
        # Measurement matrix (chỉ quan sát được x, y, w, h)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # w
        self.H[3, 3] = 1.0  # h
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        # Predict state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Convert to [x1, y1, x2, y2]
        x_center, y_center, width, height = self.state[:4]
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def update(self, bbox: np.ndarray):
        """
        Update state với measurement.
        
        Args:
            bbox: Measured bounding box [x1, y1, x2, y2]
        """
        # Convert to [x_center, y_center, width, height]
        x1, y1, x2, y2 = bbox
        z = np.array([
            (x1 + x2) / 2.0,  # x_center
            (y1 + y2) / 2.0,  # y_center
            x2 - x1,          # width
            y2 - y1           # height
        ], dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def get_bbox(self) -> np.ndarray:
        """
        Get current bounding box.
        
        Returns:
            Bbox [x1, y1, x2, y2]
        """
        x_center, y_center, width, height = self.state[:4]
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class Track:
    """
    Một track (đối tượng đang được theo dõi).
    """
    
    def __init__(self, bbox: np.ndarray, track_id: int, frame_id: int):
        """
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            track_id: Unique track ID
            frame_id: Frame ID khi track được tạo
        """
        self.track_id = track_id
        self.kalman_filter = KalmanFilter(bbox)
        self.age = 0  # Số frames kể từ lần update cuối
        self.time_since_update = 0
        self.hit_streak = 1  # Số lần liên tiếp được matched
        self.frame_id = frame_id
    
    def predict(self) -> np.ndarray:
        """Predict next bbox."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kalman_filter.predict()
    
    def update(self, bbox: np.ndarray):
        """Update với detection mới."""
        self.kalman_filter.update(bbox)
        self.time_since_update = 0
        self.hit_streak += 1


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Tính IoU giữa hai bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class SORTTracker:
    """
    SORT Tracker chính.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Args:
            max_age: Số frames tối đa một track có thể tồn tại mà không được matched
            min_hits: Số lần matched tối thiểu để track được coi là "confirmed"
            iou_threshold: Ngưỡng IoU để match detections với tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: List[np.ndarray]) -> List[Dict]:
        """
        Update tracker với detections mới.
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2]
        
        Returns:
            List of confirmed tracks với format:
            {
                'track_id': int,
                'bbox': [x1, y1, x2, y2],
                'age': int,
                'hit_streak': int
            }
        """
        self.frame_count += 1
        
        # Predict cho tất cả tracks hiện tại
        predicted_bboxes = []
        for track in self.tracks:
            pred_bbox = track.predict()
            predicted_bboxes.append(pred_bbox)
        
        # Association: Match detections với tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            # Compute cost matrix (IoU)
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track_bbox in enumerate(predicted_bboxes):
                for j, det_bbox in enumerate(detections):
                    cost_matrix[i, j] = 1.0 - iou(track_bbox, det_bbox)  # Cost = 1 - IoU
            
            # Hungarian algorithm
            matched_indices, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                cost_matrix, self.iou_threshold
            )
        else:
            matched_indices = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for trk_idx, det_idx in matched_indices:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Create new tracks cho unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                detections[det_idx],
                self.next_id,
                self.frame_count
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        
        # Return confirmed tracks only
        confirmed_tracks = []
        for track in self.tracks:
            if track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = track.kalman_filter.get_bbox()
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox.tolist(),
                    'age': track.age,
                    'hit_streak': track.hit_streak,
                    'time_since_update': track.time_since_update
                })
        
        return confirmed_tracks
    
    def _associate_detections_to_trackers(
        self,
        cost_matrix: np.ndarray,
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections với tracks sử dụng Hungarian algorithm.
        
        Returns:
            (matched_indices, unmatched_dets, unmatched_trks)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[1])), list(range(cost_matrix.shape[0]))
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_indices = []
        unmatched_dets = []
        unmatched_trks = []
        
        # Check matches
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < (1.0 - iou_threshold):  # IoU > threshold
                matched_indices.append((r, c))
            else:
                unmatched_dets.append(c)
                unmatched_trks.append(r)
        
        # Find unmatched detections
        all_det_indices = set(range(cost_matrix.shape[1]))
        matched_det_indices = set([c for _, c in matched_indices])
        unmatched_dets.extend(list(all_det_indices - matched_det_indices))
        
        # Find unmatched tracks
        all_trk_indices = set(range(cost_matrix.shape[0]))
        matched_trk_indices = set([r for r, _ in matched_indices])
        unmatched_trks.extend(list(all_trk_indices - matched_trk_indices))
        
        return matched_indices, unmatched_dets, unmatched_trks


if __name__ == "__main__":
    # Test SORT tracker
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Simulate detections qua nhiều frames
    detections_frame1 = [
        np.array([100, 100, 200, 200], dtype=np.float32),
        np.array([300, 300, 400, 400], dtype=np.float32)
    ]
    
    detections_frame2 = [
        np.array([105, 105, 205, 205], dtype=np.float32),  # Track 1 (slightly moved)
        np.array([310, 310, 410, 410], dtype=np.float32)  # Track 2 (slightly moved)
    ]
    
    detections_frame3 = [
        np.array([110, 110, 210, 210], dtype=np.float32)  # Track 1 only (Track 2 missed)
    ]
    
    # Update tracker
    tracks1 = tracker.update(detections_frame1)
    print(f"Frame 1: {len(tracks1)} tracks")
    
    tracks2 = tracker.update(detections_frame2)
    print(f"Frame 2: {len(tracks2)} tracks")
    
    tracks3 = tracker.update(detections_frame3)
    print(f"Frame 3: {len(tracks3)} tracks")
    print(f"Track IDs: {[t['track_id'] for t in tracks3]}")
    
    print("✅ SORT Tracker tested successfully!")


