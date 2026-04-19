"""
Privacy Protection Module
Face detection and blurring for GDPR / privacy compliance.
Uses OpenCV Haar cascades — no external API dependencies.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """
    Applies privacy protection to video frames by detecting and
    blurring human faces before display or storage.
    """

    def __init__(self, blur_strength: int = 51,
                 scale_factor: float = 1.15,
                 min_neighbors: int = 5,
                 min_face_size: Tuple[int, int] = (20, 20),
                 enabled: bool = True):
        """
        Args:
            blur_strength: Gaussian blur kernel size (must be odd)
            scale_factor: Haar cascade scale factor
            min_neighbors: Min detections for a positive hit
            min_face_size: Minimum face size in pixels (w, h)
            enabled: Whether the filter is active
        """
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.enabled = enabled

        # Load cascades
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_path)

        if self.face_cascade.empty():
            logger.error("Failed to load frontal face cascade")
        if self.profile_cascade.empty():
            logger.warning("Profile face cascade not available — using frontal only")

        self._face_count = 0
        self._frames_processed = 0

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.

        Returns:
            List of (x, y, w, h) tuples for each detected face.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )

        # Also try profile faces
        profiles = []
        if not self.profile_cascade.empty():
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors + 1,
                minSize=self.min_face_size
            )

        all_faces = list(frontal) if len(frontal) > 0 else []
        if len(profiles) > 0:
            all_faces.extend(list(profiles))

        # Remove duplicates (faces overlapping >50%)
        all_faces = self._non_max_suppress(all_faces, overlap_thresh=0.5)
        return all_faces

    def blur_faces(self, frame: np.ndarray,
                   method: str = 'gaussian') -> np.ndarray:
        """
        Detect and blur all faces in a frame.

        Args:
            frame: Input BGR frame
            method: 'gaussian', 'pixelate', or 'black'

        Returns:
            Frame with faces blurred/obscured
        """
        if not self.enabled:
            return frame

        self._frames_processed += 1
        result = frame.copy()
        faces = self.detect_faces(frame)
        self._face_count += len(faces)

        for (x, y, w, h) in faces:
            # Add padding around face (15% each side)
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            face_roi = result[y1:y2, x1:x2]

            if method == 'gaussian':
                blurred = cv2.GaussianBlur(
                    face_roi,
                    (self.blur_strength, self.blur_strength),
                    30
                )
            elif method == 'pixelate':
                # Pixelation effect
                small = cv2.resize(face_roi, (8, 8),
                                   interpolation=cv2.INTER_LINEAR)
                blurred = cv2.resize(small, (x2 - x1, y2 - y1),
                                     interpolation=cv2.INTER_NEAREST)
            elif method == 'black':
                blurred = np.zeros_like(face_roi)
            else:
                blurred = cv2.GaussianBlur(
                    face_roi,
                    (self.blur_strength, self.blur_strength),
                    30
                )

            result[y1:y2, x1:x2] = blurred

        return result

    def _non_max_suppress(self, boxes: list,
                          overlap_thresh: float = 0.5) -> list:
        """Simple non-maximum suppression to remove duplicate detections."""
        if len(boxes) == 0:
            return []

        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = x1 + boxes_array[:, 2]
        y2 = y1 + boxes_array[:, 3]
        areas = boxes_array[:, 2] * boxes_array[:, 3]

        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        keep = []

        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[indices[1:]]

            remaining = np.where(overlap < overlap_thresh)[0]
            indices = indices[remaining + 1]

        return [tuple(boxes_array[i]) for i in keep]

    def get_stats(self) -> Dict:
        """Get privacy filter statistics."""
        return {
            'enabled': self.enabled,
            'frames_processed': self._frames_processed,
            'total_faces_detected': self._face_count,
            'avg_faces_per_frame': (
                self._face_count / self._frames_processed
                if self._frames_processed > 0 else 0
            )
        }

    def toggle(self, enabled: bool = None):
        """Toggle privacy filter on/off."""
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = not self.enabled
        logger.info(f"Privacy filter {'enabled' if self.enabled else 'disabled'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pf = PrivacyFilter(blur_strength=51, enabled=True)

    # Test with a synthetic frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pf.blur_faces(test_frame)
    print(f"Privacy filter stats: {pf.get_stats()}")
    print(f"Output frame shape: {result.shape}")
