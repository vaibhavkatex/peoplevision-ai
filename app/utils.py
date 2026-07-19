import csv
import logging
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("peoplevision-ai.utils")

class FpsCalculator:
    """Calculates smoothed real-time Frames Per Second (FPS)."""

    def __init__(self, buffer_size: int = 30):
        """Initializes FPS calculator.

        Args:
            buffer_size (int): Size of rolling window of timestamps to calculate FPS.
        """
        self.buffer_size = buffer_size
        self.timestamps: List[float] = []

    def tick(self) -> float:
        """Records a frame timestamp and returns the current rolling average FPS.

        Returns:
            float: Calculated average FPS.
        """
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.buffer_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


# Global instances of annotators
box_annotator = sv.BoxAnnotator(
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.5,
    text_padding=4
)

def draw_detections(frame: cv2.Mat, detections: sv.Detections) -> cv2.Mat:
    """Draws bounding boxes and labels with tracker IDs on the frame.

    Args:
        frame (cv2.Mat): OpenCV image frame.
        detections (sv.Detections): Detections containing coordinates and tracker IDs.

    Returns:
        cv2.Mat: Annotated image frame.
    """
    if len(detections) == 0:
        return frame

    # Generate custom labels detailing tracker ID and confidence
    labels = []
    for tracker_id, confidence in zip(detections.tracker_id, detections.confidence):
        id_str = f"ID: {tracker_id}" if tracker_id is not None else "Detecting..."
        labels.append(f"{id_str} ({confidence:.2f})")

    # Annotate frame
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


def draw_overlay_metrics(frame: cv2.Mat, fps: float, current_count: int, status: str) -> cv2.Mat:
    """Overlays basic system performance statistics on the top-left of the video frame.

    Args:
        frame (cv2.Mat): Frame to write text onto.
        fps (float): Current processed FPS.
        current_count (int): Current people count visible in the frame.
        status (str): Camera stream connection status.

    Returns:
        cv2.Mat: Output frame with stats drawn on top.
    """
    overlay = frame.copy()
    
    # Text metadata
    stats = [
        f"FPS: {fps:.1f}",
        f"In Frame: {current_count}",
        f"Camera: {status.upper()}"
    ]
    
    # Draw simple background block for readability
    cv2.rectangle(overlay, (10, 10), (180, 85), (0, 0, 0), -1)
    # Blend with original image
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y_offset = 30
    for stat in stats:
        cv2.putText(
            frame,
            stat,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y_offset += 20
        
    return frame


def export_events_to_csv(events: List[Dict], filepath: str) -> None:
    """Exports a list of serialized crossing events to a CSV file.

    Args:
        events (List[Dict]): List of dictionary-serialized crossing events.
        filepath (str): Target CSV destination file path.
    """
    logger.info(f"Exporting {len(events)} events to CSV: {filepath}")
    fields = ["id", "timestamp", "direction", "tracker_id", "camera_id"]
    
    try:
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for event in events:
                writer.writerow({
                    "id": event.get("id"),
                    "timestamp": event.get("timestamp"),
                    "direction": event.get("direction"),
                    "tracker_id": event.get("tracker_id"),
                    "camera_id": event.get("camera_id"),
                })
        logger.info("CSV export completed successfully.")
    except Exception as e:
        logger.error(f"Error during CSV export: {e}")
        raise
