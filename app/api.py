import logging
import time
from typing import Dict, Generator, List, Optional
import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db, get_crossing_stats, get_historical_events, clear_database_logs
from app.config import settings

logger = logging.getLogger("peoplevision-ai.api")
router = APIRouter()


class ConfigUpdateSchema(BaseModel):
    """Schema for updating runtime settings."""
    confidence_threshold: float = Field(..., ge=0.0, le=1.0, description="Confidence threshold between 0.0 and 1.0")


@router.get("/health")
def get_health(db: Session = Depends(get_db)) -> Dict:
    """Service health status check endpoint.

    Args:
        db (Session): Database session dependency.

    Returns:
        Dict: JSON object detailing api service states.
    """
    db_status = "healthy"
    try:
        # Perform simple test query
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    return {
        "status": "online",
        "timestamp": time.time(),
        "database": db_status
    }


@router.get("/count")
def get_count(request: Request) -> Dict:
    """Retrieve live tracking and crossing counters.

    Args:
        request (Request): FastAPI request to access application state.

    Returns:
        Dict: Current metrics in frame.
    """
    app_state = request.app.state
    return {
        "current_occupancy": getattr(app_state, "current_count", 0),
        "entries": getattr(app_state, "entries", 0),
        "exits": getattr(app_state, "exits", 0),
        "fps": round(getattr(app_state, "fps", 0.0), 2),
        "confidence_threshold": getattr(app_state, "confidence_threshold", settings.CONFIDENCE_THRESHOLD)
    }


@router.get("/stats")
def get_stats(request: Request, db: Session = Depends(get_db)) -> Dict:
    """Retrieve historical aggregated stats and event logs.

    Args:
        request (Request): Request context.
        db (Session): Database session.

    Returns:
        Dict: Combined database totals and recent event records.
    """
    db_stats = get_crossing_stats(db)
    history = get_historical_events(db, limit=50)
    
    # Sync memory totals with DB totals
    app_state = request.app.state
    if hasattr(app_state, "entries"):
        app_state.entries = db_stats["entries"]
    if hasattr(app_state, "exits"):
        app_state.exits = db_stats["exits"]

    return {
        "total_entries": db_stats["entries"],
        "total_exits": db_stats["exits"],
        "history": [event.to_dict() for event in history]
    }


@router.post("/reset")
def post_reset(request: Request, clear_db: bool = False, db: Session = Depends(get_db)) -> Dict:
    """Resets tracking counts in memory and optionally in the database.

    Args:
        request (Request): Request context.
        clear_db (bool): If true, deletes all crossing event rows from the database.
        db (Session): Database session.

    Returns:
        Dict: Status message.
    """
    app_state = request.app.state
    
    # Reset the in-memory counter
    if hasattr(app_state, "counter") and app_state.counter is not None:
        app_state.counter.reset_counts()
    
    app_state.entries = 0
    app_state.exits = 0
    app_state.current_count = 0

    if clear_db:
        clear_database_logs(db)
        logger.info("Cleared all crossing events from database on reset request.")

    return {
        "status": "success",
        "message": "System counts and trackers reset successfully."
    }


@router.post("/config")
def post_config(request: Request, config: ConfigUpdateSchema) -> Dict:
    """Updates runtime configuration settings.

    Args:
        request (Request): Request context.
        config (ConfigUpdateSchema): Payload containing updated parameters.

    Returns:
        Dict: Confirmation of new setting.
    """
    request.app.state.confidence_threshold = config.confidence_threshold
    logger.info(f"Runtime confidence threshold updated to: {config.confidence_threshold}")
    return {
        "status": "success",
        "confidence_threshold": config.confidence_threshold
    }


# Frame generator for MJPEG streaming
def frame_generator(request: Request) -> Generator[bytes, None, None]:
    """Yields processed JPEGs from application state as multipart boundary blocks.

    Args:
        request (Request): FastAPI request context.

    Yields:
        bytes: Frame payload formatted for MJPEG boundary.
    """
    app_state = request.app.state
    # Maintain active streaming loop as long as the client request connection persists
    while True:
        frame = getattr(app_state, "annotated_frame", None)
        if frame is not None:
            ret, encoded_jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + encoded_jpeg.tobytes() + b'\r\n'
                )
        time.sleep(0.04)  # Limit stream chunk delivery rate (~25 fps max)


@router.get("/video_feed")
def get_video_feed(request: Request) -> StreamingResponse:
    """MJPEG streaming endpoint.

    Args:
        request (Request): FastAPI request.

    Returns:
        StreamingResponse: Stream of frames to be consumed by streamlit or a browser.
    """
    return StreamingResponse(
        frame_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
