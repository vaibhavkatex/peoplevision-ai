from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import supervision as sv
from sqlalchemy.orm import Session
from app.counter import PeopleCounter

def test_counter_lazy_initialization():
    """Test that absolute line coordinates are set correctly based on frame shape."""
    counter = PeopleCounter(
        camera_id="test_cam",
        start_pct=(0.0, 0.4),
        end_pct=(1.0, 0.6)
    )
    assert counter.line_zone is None

    # Shape: height=400, width=600
    dummy_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    empty_detections = sv.Detections.empty()

    # Trigger first update to trigger initialization
    counter.update(empty_detections, dummy_frame)

    assert counter.line_zone is not None
    assert counter.line_annotator is not None
    assert counter.width == 600
    assert counter.height == 400

    # Retrieve coordinates from sv.Point objects (internal to LineZone)
    assert counter.line_zone.vector.start.x == 0
    assert counter.line_zone.vector.start.y == 160  # 400 * 0.4
    assert counter.line_zone.vector.end.x == 600
    assert counter.line_zone.vector.end.y == 240    # 400 * 0.6


def test_counter_crossing_event_dispatch():
    """Test that detection crossing trigger increments counts and records DB events."""
    counter = PeopleCounter(camera_id="test_cam")
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock supervision LineZone to control crossing outcomes
    with patch("app.counter.sv.LineZone") as mock_line_zone_cls:
        mock_lz_instance = MagicMock()
        mock_line_zone_cls.return_value = mock_lz_instance
        
        # Setup crossing output:
        # First track (ID 11): crossed in (entry)
        # Second track (ID 22): crossed out (exit)
        mock_lz_instance.trigger.return_value = (
            np.array([True, False]),  # crossed_in
            np.array([False, True])   # crossed_out
        )

        # Build dummy detections
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 30, 30], [40, 40, 60, 60]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0]),
            tracker_id=np.array([11, 22])
        )

        # Mock database session
        mock_db = MagicMock(spec=Session)

        # Execute
        counter.update(detections, dummy_frame, db=mock_db)

        # Check local counts
        assert counter.entries == 1
        assert counter.exits == 1

        # Check that DB operations were invoked for both directions
        assert mock_db.add.call_count == 2
        mock_db.commit.assert_called()
