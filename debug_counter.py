"""Debug script: Shows live detections, tracker IDs, and line crossing events."""
import cv2
import supervision as sv
from ultralytics import YOLO
from app.config import settings

print("=" * 60)
print("  PeopleVision Debug - Line Crossing Test")
print("=" * 60)

# Load model
model = YOLO(settings.YOLO_MODEL)
print(f"[OK] Model loaded: {settings.YOLO_MODEL}")

# Open camera
src = settings.parsed_camera_source
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    print(f"[FAIL] Cannot open camera source: {src}")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("[FAIL] Cannot read frame from camera")
    exit(1)

h, w = frame.shape[:2]
print(f"[OK] Camera opened: {w}x{h}")

# Build vertical line
x1 = int(settings.LINE_START_X_PCT * w)
y1 = int(settings.LINE_START_Y_PCT * h)
x2 = int(settings.LINE_END_X_PCT * w)
y2 = int(settings.LINE_END_Y_PCT * h)
print(f"[OK] Line from ({x1},{y1}) to ({x2},{y2})")

line_zone = sv.LineZone(
    start=sv.Point(x=x1, y=y1),
    end=sv.Point(x=x2, y=y2)
)

tracker = sv.ByteTrack(
    frame_rate=settings.TRACKER_FPS,
    track_activation_threshold=settings.CONFIDENCE_THRESHOLD
)

line_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.8)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

print("\n[INFO] Running live loop... Press 'q' to quit.\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model(frame, verbose=False, conf=settings.CONFIDENCE_THRESHOLD)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter person class only
    person_mask = detections.class_id == 0
    detections = detections[person_mask]

    # Track
    detections = tracker.update_with_detections(detections)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"  Frame {frame_count}: {len(detections)} person(s) detected, tracker_ids={detections.tracker_id}")

    # Check crossings
    crossed_in, crossed_out = line_zone.trigger(detections=detections)
    
    for i in range(len(detections)):
        if crossed_in[i]:
            tid = int(detections.tracker_id[i])
            print(f"  >>> ENTRY (in): Person {tid} crossed to RIGHT side!")
        if crossed_out[i]:
            tid = int(detections.tracker_id[i])
            print(f"  <<< EXIT (out): Person {tid} crossed to LEFT side!")

    # Annotate
    labels = [f"ID:{tid}" for tid in detections.tracker_id] if detections.tracker_id is not None else []
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

    cv2.imshow("PeopleVision Debug", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal IN: {line_zone.in_count}, Total OUT: {line_zone.out_count}")
