"""Verifies crossing directions programmatically using a simulated environment."""
import numpy as np
import cv2
import supervision as sv
from app.counter import PeopleCounter

def run_verification():
    print("=" * 60)
    print("  Verifying Line Crossing Direction Logic")
    print("=" * 60)

    # 1. Create a simulated frame (640x480)
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. Initialize PeopleCounter (Vertical line at 50% width)
    # LINE_START_X_PCT=0.5, LINE_START_Y_PCT=0.0
    # LINE_END_X_PCT=0.5, LINE_END_Y_PCT=1.0
    counter = PeopleCounter(
        camera_id="test_cam",
        start_pct=(0.5, 0.0),
        end_pct=(0.5, 1.0)
    )

    print("\n--- Phase 1: Left to Right Crossing (ENTRY/IN) ---")
    
    # Frame 1: Person (ID 9) is on the LEFT side (x = 200)
    # Bounding box coordinates: [x_min, y_min, x_max, y_max]
    box_frame1 = np.array([[180, 200, 220, 300]], dtype=np.float32)
    detections_frame1 = sv.Detections(
        xyxy=box_frame1,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([9])
    )
    counter.update(detections_frame1, frame, db=None)
    print(f"Frame 1 (ID 9 at x=200): Entries={counter.entries}, Exits={counter.exits} (Expected: 0, 0)")

    # Frame 2: Person (ID 9) has moved to the RIGHT side (x = 450)
    box_frame2 = np.array([[430, 200, 470, 300]], dtype=np.float32)
    detections_frame2 = sv.Detections(
        xyxy=box_frame2,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([9])
    )
    counter.update(detections_frame2, frame, db=None)
    print(f"Frame 2 (ID 9 at x=450): Entries={counter.entries}, Exits={counter.exits} (Expected: 1, 0)")
    
    # Assert check
    assert counter.entries == 1 and counter.exits == 0, "Left-to-Right crossing failed to count as Entry!"
    print("[PASS] Left-to-Right registered as ENTRY (IN) successfully.")

    print("\n--- Phase 2: Right to Left Crossing (EXIT/OUT) ---")
    
    # Frame 3: A new person (ID 12) is on the RIGHT side (x = 450)
    box_frame3 = np.array([[430, 200, 470, 300]], dtype=np.float32)
    detections_frame3 = sv.Detections(
        xyxy=box_frame3,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([12])
    )
    counter.update(detections_frame3, frame, db=None)
    print(f"Frame 3 (ID 12 at x=450): Entries={counter.entries}, Exits={counter.exits} (Expected: 1, 0)")

    # Frame 4: Person (ID 12) has moved to the LEFT side (x = 200)
    box_frame4 = np.array([[180, 200, 220, 300]], dtype=np.float32)
    detections_frame4 = sv.Detections(
        xyxy=box_frame4,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([12])
    )
    counter.update(detections_frame4, frame, db=None)
    print(f"Frame 4 (ID 12 at x=200): Entries={counter.entries}, Exits={counter.exits} (Expected: 1, 1)")

    # Assert check
    assert counter.entries == 1 and counter.exits == 1, "Right-to-Left crossing failed to count as Exit!"
    print("[PASS] Right-to-Left registered as EXIT (OUT) successfully.")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
