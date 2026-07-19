import logging
import threading
import time
from typing import Tuple, Union, Optional
import cv2

logger = logging.getLogger("peoplevision-ai.camera")

class CameraStream:
    """Thread-safe stream reader for USB webcams, RTSP streams, or local files."""

    def __init__(self, source: Union[int, str], is_video_file: bool = False):
        """Initializes the camera stream.

        Args:
            source (Union[int, str]): USB camera index, RTSP URL, or path to video file.
            is_video_file (bool): Whether the source is a video file that should be looped.
        """
        self.source = source
        self.is_video_file = is_video_file
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[cv2.Mat] = None
        self.ret: bool = False
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Read metadata parameters
        self.width: int = 640
        self.height: int = 480
        self.fps: float = 30.0

    def start(self) -> "CameraStream":
        """Starts the background frame capture thread.

        Returns:
            CameraStream: Self reference.
        """
        if self.running:
            logger.warning("Camera stream thread already running.")
            return self

        logger.info(f"Opening camera source: {self.source} (is_video_file={self.is_video_file})")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
        else:
            self._update_metadata()
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, args=(), name="CameraReaderThread", daemon=True)
        self.thread.start()
        logger.info("Camera stream reader thread started.")
        return self

    def _update_metadata(self) -> None:
        """Retrieves and updates camera metadata (width, height, FPS)."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = self.cap.get(cv2.CAP_PROP_FPS)
            
            if w > 0: self.width = w
            if h > 0: self.height = h
            if f > 0: self.fps = f
            logger.info(f"Source metadata: Resolution={self.width}x{self.height}, FPS={self.fps}")

    def _update_loop(self) -> None:
        """Continuously reads frames from VideoCapture and stores the latest frame."""
        consecutive_failures = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                logger.warning(f"Video capture is not open. Reconnecting to: {self.source}")
                time.sleep(2.0)
                with self.lock:
                    self.cap = cv2.VideoCapture(self.source)
                    if self.cap.isOpened():
                        self._update_metadata()
                        consecutive_failures = 0
                continue

            ret, frame = self.cap.read()
            if ret:
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame
                    self.ret = True
            else:
                consecutive_failures += 1
                if self.is_video_file:
                    # Reset video file back to the first frame
                    logger.debug("Looping video file to start.")
                    with self.lock:
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    if consecutive_failures > 50:
                        logger.error(f"Failed to read frame {consecutive_failures} times. Reconnecting...")
                        with self.lock:
                            if self.cap:
                                self.cap.release()
                        time.sleep(2.0)
                # Introduce a small sleep to prevent CPU starvation
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Returns the latest captured frame in a thread-safe manner.

        Returns:
            Tuple[bool, Optional[cv2.Mat]]: Capture success status and copy of the frame.
        """
        with self.lock:
            # We copy the frame to avoid race conditions when the thread overwrites it
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def get_status(self) -> str:
        """Gets a human-readable camera connection status.

        Returns:
            str: Connection status ('connected' or 'disconnected').
        """
        with self.lock:
            if self.cap and self.cap.isOpened() and self.ret:
                return "connected"
            return "disconnected"

    def stop(self) -> None:
        """Stops the camera stream thread and releases resources."""
        logger.info("Stopping camera stream...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        with self.lock:
            if self.cap:
                self.cap.release()
                logger.info("Camera capture source released.")
            self.ret = False
            self.frame = None
