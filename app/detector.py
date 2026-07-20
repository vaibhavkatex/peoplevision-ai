import logging
from typing import Optional
import cv2
import supervision as sv
from ultralytics import YOLO
from app.config import settings

logger = logging.getLogger("peoplevision-ai.detector")

class YoloDetector:
    """Wrapper class for YOLO-based person detection using Ultralytics and Supervision."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        target_class_id: Optional[int] = None
    ):
        """Initializes the YOLO detector.

        Args:
            model_name (Optional[str]): Path or weight name for YOLO.
            device (Optional[str]): Device to run the model on (cpu, cuda, mps).
            target_class_id (Optional[int]): Target class ID (defaults to person=0).
        """
        self.model_name = model_name or settings.YOLO_MODEL
        self.device = device or settings.DEVICE
        self.target_class_id = target_class_id if target_class_id is not None else settings.TARGET_CLASS_ID

        logger.info(f"Loading YOLO model: {self.model_name} on device: {self.device}")
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: cv2.Mat, confidence_threshold: float = settings.CONFIDENCE_THRESHOLD) -> sv.Detections:
        """Runs inference on a frame and returns filtered detections.

        Args:
            frame (cv2.Mat): Input image frame from camera.
            confidence_threshold (float): Confidence threshold to keep detections.

        Returns:
            sv.Detections: Filtered supervision Detections object containing bounding boxes,
                           class IDs, and confidence scores.
        """
        # Run inference at 320px resolution for 3x speedup on CPU
        results = self.model(frame, imgsz=320, verbose=False)[0]

        # Convert Ultralytics results to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections: only target class (person) AND above confidence threshold
        mask = (detections.class_id == self.target_class_id) & (detections.confidence >= confidence_threshold)
        filtered_detections = detections[mask]

        logger.debug(f"Detected {len(filtered_detections)} people (threshold={confidence_threshold}).")
        return filtered_detections
