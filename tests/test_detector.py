from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import supervision as sv
from app.detector import YoloDetector

def test_detector_initialization():
    """Test that the detector initializes YOLO with correct arguments."""
    with patch("app.detector.YOLO") as mock_yolo:
        detector = YoloDetector(model_name="yolov8n.pt", device="cpu", target_class_id=0)
        assert detector.model_name == "yolov8n.pt"
        assert detector.device == "cpu"
        assert detector.target_class_id == 0
        mock_yolo.assert_called_once_with("yolov8n.pt")


def test_detector_inference_filtering():
    """Test that detection output is filtered by confidence and target class."""
    with patch("app.detector.YOLO") as mock_yolo:
        # Instantiate detector
        detector = YoloDetector(model_name="yolov8n.pt", device="cpu", target_class_id=0)
        
        # Setup mock inference return
        mock_result = MagicMock()
        detector.model.return_value = [mock_result]

        # Define mock detections returned from supervision.Detections.from_ultralytics
        # Index 0: Person, high confidence (should keep)
        # Index 1: Person, low confidence (should filter out)
        # Index 2: Bicycle, high confidence (should filter out class mismatch)
        mock_raw_detections = sv.Detections(
            xyxy=np.array([
                [10, 10, 50, 50],
                [60, 60, 100, 100],
                [10, 60, 50, 100]
            ]),
            confidence=np.array([0.85, 0.25, 0.90]),
            class_id=np.array([0, 0, 1])
        )

        with patch("supervision.Detections.from_ultralytics", return_value=mock_raw_detections):
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            filtered = detector.detect(dummy_frame, confidence_threshold=0.5)

            # Assert only 1 target detection remains (index 0)
            assert len(filtered) == 1
            assert filtered.class_id[0] == 0
            assert filtered.confidence[0] == 0.85
            np.testing.assert_array_equal(filtered.xyxy[0], [10, 10, 50, 50])
