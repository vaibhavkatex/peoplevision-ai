import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import supervision as sv
from sqlalchemy.orm import Session
from app.config import settings
from app.database import add_crossing_event

logger = logging.getLogger("peoplevision-ai.counter")

class PeopleCounter:
    """Manages virtual line crossing logic for counting entries and exits of tracked people."""

    def __init__(
        self,
        camera_id: str = "camera_0",
        start_pct: Tuple[float, float] = (settings.LINE_START_X_PCT, settings.LINE_START_Y_PCT),
        end_pct: Tuple[float, float] = (settings.LINE_END_X_PCT, settings.LINE_END_Y_PCT)
    ):
        """Initializes the people counter.

        Args:
            camera_id (str): Identifier of the camera source.
            start_pct (Tuple[float, float]): (x_pct, y_pct) start coordinates of the line.
            end_pct (Tuple[float, float]): (x_pct, y_pct) end coordinates of the line.
        """
        self.camera_id = camera_id
        self.start_pct = start_pct
        self.end_pct = end_pct

        # Lazy-initialized once frame size is known
        self.line_zone: Optional[sv.LineZone] = None
        self.line_annotator: Optional[sv.LineZoneAnnotator] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

        # Cumulative stats cached in memory (synced to DB during updates)
        self.entries: int = 0
        self.exits: int = 0

    def _init_line_zone(self, width: int, height: int) -> None:
        """Initializes the virtual LineZone using absolute pixel coordinates.

        Args:
            width (int): Frame width.
            height (int): Frame height.
        """
        self.width = width
        self.height = height

        x1 = int(self.start_pct[0] * width)
        y1 = int(self.start_pct[1] * height)
        x2 = int(self.end_pct[0] * width)
        y2 = int(self.end_pct[1] * height)

        logger.info(f"Initializing LineZone from ({x1}, {y1}) to ({x2}, {y2}) based on percentages.")
        
        self.line_zone = sv.LineZone(
            start=sv.Point(x=x1, y=y1),
            end=sv.Point(x=x2, y=y2)
        )
        # Sync initial counts (e.g. from DB) into the line_zone for consistent rendering
        self.line_zone._in_count_per_class[0] = self.entries
        self.line_zone._out_count_per_class[0] = self.exits
        
        # LineZoneAnnotator draws the line with direction arrows and count overlays
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=3,
            text_thickness=2,
            text_scale=0.8,
            text_padding=6
        )

    def update(self, detections: sv.Detections, frame: cv2.Mat, db: Optional[Session] = None) -> None:
        """Checks for line crossings and records them in memory and the database.

        Args:
            detections (sv.Detections): Tracker-associated detections.
            frame (cv2.Mat): Current frame image (used to get dimensions if uninitialized).
            db (Optional[Session]): SQLAlchemy database session to save crossing events.
        """
        if self.line_zone is None:
            h, w = frame.shape[:2]
            self._init_line_zone(w, h)

        # Line crossing detection requires tracking IDs to be populated
        if detections.tracker_id is None or len(detections) == 0:
            return

        # Trigger LineZone crossing logic.
        # crossed_in: Boolean array where True means the object crossed in the entry direction.
        # crossed_out: Boolean array where True means the object crossed in the exit direction.
        crossed_in, crossed_out = self.line_zone.trigger(detections=detections)

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            
            if crossed_in[i]:
                self.entries += 1
                logger.info(f"Person {tracker_id} crossed ENTRY line.")
                if db:
                    try:
                        add_crossing_event(db, direction="entry", tracker_id=tracker_id, camera_id=self.camera_id)
                    except Exception as e:
                        logger.error(f"Failed to record entry crossing to database: {e}")

            elif crossed_out[i]:
                self.exits += 1
                logger.info(f"Person {tracker_id} crossed EXIT line.")
                if db:
                    try:
                        add_crossing_event(db, direction="exit", tracker_id=tracker_id, camera_id=self.camera_id)
                    except Exception as e:
                        logger.error(f"Failed to record exit crossing to database: {e}")

    def annotate(self, frame: cv2.Mat) -> cv2.Mat:
        """Annotates the line and text summaries on the frame.

        Args:
            frame (cv2.Mat): Image frame.

        Returns:
            cv2.Mat: Annotated frame.
        """
        if self.line_zone is not None and self.line_annotator is not None:
            return self.line_annotator.annotate(frame=frame, line_counter=self.line_zone)
        return frame

    def reset_counts(self) -> None:
        """Resets local cumulative in-memory counts."""
        self.entries = 0
        self.exits = 0
        if self.line_zone:
            # supervision LineZone stores counts inside in_count and out_count properties
            # reset these properties using internal Counter dictionaries
            self.line_zone._in_count_per_class.clear()
            self.line_zone._out_count_per_class.clear()
            # Also clear the internal tracker states to allow re-entry of the same objects
            self.line_zone.crossing_state_history.clear()
        logger.info("Local counter statistics reset.")
