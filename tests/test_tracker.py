from unittest.mock import MagicMock, patch
import numpy as np
import supervision as sv
from app.tracker import PersonTracker

def test_tracker_initialization():
    """Test that the tracker initializes Supervision ByteTrack with configured parameters."""
    with patch("app.tracker.sv.ByteTrack") as mock_bytetrack:
        PersonTracker(fps=25, buffer=45)
        # Ensure ByteTrack constructor is triggered with settings-derived args
        mock_bytetrack.assert_called_once()


def test_tracker_update_delegation():
    """Test that tracker.update delegates bounding box tracking updates properly."""
    with patch("app.tracker.sv.ByteTrack") as mock_bytetrack:
        mock_bt_instance = MagicMock()
        mock_bytetrack.return_value = mock_bt_instance

        tracker = PersonTracker()

        # Inputs
        raw_detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )

        # Expected output from the tracker
        expected_tracked_detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            tracker_id=np.array([42]) # tracker ID assigned
        )
        mock_bt_instance.update_with_detections.return_value = expected_tracked_detections

        # Run tracker update
        tracked_out = tracker.update(raw_detections)

        # Verify tracking coordinates are passed to inner supervision tracker
        mock_bt_instance.update_with_detections.assert_called_once_with(detections=raw_detections)
        assert tracked_out.tracker_id is not None
        assert tracked_out.tracker_id[0] == 42
