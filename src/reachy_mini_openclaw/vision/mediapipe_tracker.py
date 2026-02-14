"""MediaPipe-based head tracker for face detection.

Uses MediaPipe Face Detection for lightweight face tracking.
Falls back to this if YOLO is not available.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "To use MediaPipe head tracker, install: pip install mediapipe"
    ) from e


logger = logging.getLogger(__name__)


class HeadTracker:
    """Lightweight head tracker using MediaPipe for face detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
    ) -> None:
        """Initialize MediaPipe-based head tracker.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            model_selection: 0 for short-range (2m), 1 for long-range (5m)
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )
        logger.info("MediaPipe face detection initialized")

    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[float]]:
        """Get head position from face detection.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (eye_center in [-1,1] coords, roll_angle in radians)
        """
        h, w = img.shape[:2]

        try:
            logger.debug(f"Processing image shape: {img.shape}")
            # Convert BGR to RGB for MediaPipe and ensure contiguous memory
            rgb_img = np.ascontiguousarray(img[:, :, ::-1])
            
            # Run face detection
            results = self.face_detection.process(rgb_img)
            
            if not results.detections:
                return None, None
            
            # Get the first (most confident) detection
            detection = results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate center of face
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            # Convert to [-1, 1] range
            norm_x = center_x * 2.0 - 1.0
            norm_y = center_y * 2.0 - 1.0
            
            face_center = np.array([norm_x, norm_y], dtype=np.float32)
            
            # Estimate roll from key points if available
            roll = 0.0
            keypoints = detection.location_data.relative_keypoints
            if len(keypoints) >= 2:
                # Use left and right eye positions to estimate roll
                left_eye = keypoints[0]  # LEFT_EYE
                right_eye = keypoints[1]  # RIGHT_EYE
                
                dx = right_eye.x - left_eye.x
                dy = right_eye.y - left_eye.y
                roll = np.arctan2(dy, dx)
            
            logger.debug(f"Face detected at ({norm_x:.2f}, {norm_y:.2f}), roll: {np.degrees(roll):.1f}°")
            
            return face_center, roll

        except Exception as e:
            logger.error(f"Error in head position detection: {e}")
            return None, None
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
