import logging
from typing import Optional
import supervision as sv
from app.config import settings

logger = logging.getLogger("peoplevision-ai.tracker")

class PersonTracker:
    """Wrapper class for object tracking using Supervision's ByteTrack."""

    def __init__(
        self,
        fps: Optional[int] = None,
        buffer: Optional[int] = None
    ):
        """Initializes the ByteTrack tracker.

        Args:
            fps (Optional[int]): Frame rate of the input stream.
            buffer (Optional[int]): Number of frames to keep lost tracks in memory.
        """
        tracker_fps = fps or settings.TRACKER_FPS
        track_buffer = buffer or settings.TRACK_BUFFER

        logger.info(f"Initializing ByteTrack with FPS={tracker_fps}, buffer={track_buffer}")
        try:
            # supervision.ByteTrack parameters:
            # - track_activation_threshold (default 0.25)
            # - lost_track_buffer (default 30)
            # - minimum_matching_threshold (default 0.8)
            # - frame_rate (default 30)
            self.tracker = sv.ByteTrack(
                track_activation_threshold=settings.CONFIDENCE_THRESHOLD,
                lost_track_buffer=track_buffer,
                frame_rate=tracker_fps
            )
            logger.info("ByteTrack initialized successfully.")
        except Exception as e:
            logger.warning(f"Error initializing ByteTrack with custom arguments: {e}. Falling back to default ByteTrack.")
            self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Updates tracks with the latest detections.

        Args:
            detections (sv.Detections): Detections from the detector module.

        Returns:
            sv.Detections: Detections with 'tracker_id' field populated.
        """
        # ByteTrack updates detections in-place or returns a copy with tracker_id populated.
        tracked_detections = self.tracker.update_with_detections(detections=detections)
        
        # Log active tracker IDs for debugging if verbose
        active_ids = tracked_detections.tracker_id.tolist() if tracked_detections.tracker_id is not None else []
        logger.debug(f"Tracked IDs in current frame: {active_ids}")
        
        return tracked_detections
