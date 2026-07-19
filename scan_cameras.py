"""Scans all available camera indices (0-9) and reports which ones are active."""
import cv2

print("=" * 50)
print("  Camera Index Scanner")
print("=" * 50)

found = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            print(f"\n  [Camera {i}] ACTIVE")
            print(f"    Resolution : {w}x{h}")
            print(f"    FPS        : {fps}")
            print(f"    Backend    : {backend}")
            found.append(i)
        else:
            print(f"\n  [Camera {i}] Opened but no frame")
        cap.release()
    else:
        pass  # Camera index not available

print("\n" + "=" * 50)
if len(found) == 0:
    print("  No cameras found!")
elif len(found) == 1:
    print(f"  Only 1 camera found at index {found[0]}.")
    print(f"  Use CAMERA_SOURCE={found[0]} in .env")
else:
    print(f"  Found {len(found)} cameras at indices: {found}")
    print(f"  Laptop camera is usually index 0.")
    print(f"  DroidCam is likely index {found[-1]}.")
    print(f"  Set CAMERA_SOURCE={found[-1]} in .env")
print("=" * 50)
