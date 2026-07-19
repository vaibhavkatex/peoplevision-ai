import logging
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.config import settings, logger
from app.database import init_db, SessionLocal, get_crossing_stats, get_db
from app.camera import CameraStream
from app.detector import YoloDetector
from app.tracker import PersonTracker
from app.counter import PeopleCounter
from app.utils import FpsCalculator, draw_detections, draw_overlay_metrics
from app.api import router, get_health, get_stats, get_count, post_reset

# Logger setup
logger = logging.getLogger("peoplevision-ai.main")

def process_camera_loop(app: FastAPI) -> None:
    """Background thread function that runs the computer vision pipeline.

    It reads frames from the camera, runs detection, updates tracks, checks for
    line crossings, updates database, and annotates the frame for streaming.

    Args:
        app (FastAPI): The FastAPI application instance to write states to.
    """
    logger.info("Starting background vision processing loop...")
    fps_calc = FpsCalculator()

    while app.state.loop_running:
        camera_stream = getattr(app.state, "camera", None)
        if camera_stream is None:
            time.sleep(0.1)
            continue

        ret, frame = camera_stream.read()
        app.state.camera_status = camera_stream.get_status()

        if not ret or frame is None:
            # Wait a short duration for camera stream buffer to populate
            time.sleep(0.01)
            continue

        try:
            # 1. Update performance metrics (FPS)
            fps = fps_calc.tick()
            app.state.fps = fps

            # 2. Get the current slider configuration from application state
            conf_thresh = getattr(app.state, "confidence_threshold", settings.CONFIDENCE_THRESHOLD)

            # 3. Detect individuals (Class ID 0: Person)
            detections = app.state.detector.detect(frame, confidence_threshold=conf_thresh)

            # 4. Update track matching
            tracked_detections = app.state.tracker.update(detections)
            app.state.current_count = len(tracked_detections)

            # 5. Check line crossings and record database events using a session
            with SessionLocal() as db:
                app.state.counter.update(tracked_detections, frame, db=db)
                
            # 6. Synchronize cumulative totals back to application state
            app.state.entries = app.state.counter.entries
            app.state.exits = app.state.counter.exits

            # 7. Annotate visuals: boxes, labels, path traces, line overlay, metrics overlay
            annotated = draw_detections(frame, tracked_detections)
            annotated = app.state.counter.annotate(annotated)
            annotated = draw_overlay_metrics(
                annotated,
                fps=fps,
                current_count=app.state.current_count,
                status=app.state.camera_status
            )

            # 8. Save output frame references for MJPEG client streaming
            app.state.annotated_frame = annotated
            app.state.latest_frame = frame

        except Exception as e:
            logger.error(f"Error encountered in frame processing loop: {e}", exc_info=True)
            time.sleep(0.1)

        # Yield execution slightly to avoid thread starvation
        time.sleep(0.002)

    logger.info("Background vision processing loop stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager managing the lifespan of the FastAPI application.

    Initializes database tables, creates core model components, and launches
    the background processing worker thread on startup. Shuts them down on stop.
    """
    # Startup tasks
    logger.info("Initializing PeopleVision AI Startup Operations...")
    init_db()

    # Seed application states
    app.state.camera_status = "disconnected"
    app.state.fps = 0.0
    app.state.entries = 0
    app.state.exits = 0
    app.state.current_count = 0
    app.state.confidence_threshold = settings.CONFIDENCE_THRESHOLD
    app.state.annotated_frame = None
    app.state.latest_frame = None

    # Load starting values from database persistence
    with SessionLocal() as db:
        try:
            stats = get_crossing_stats(db)
            app.state.entries = stats["entries"]
            app.state.exits = stats["exits"]
            logger.info(f"Synchronized statistics with DB: Entries={app.state.entries}, Exits={app.state.exits}")
        except Exception as e:
            logger.error(f"Failed to seed initial count statistics from DB: {e}")

    # Launch camera stream input
    app.state.camera = CameraStream(
        source=settings.parsed_camera_source,
        is_video_file=settings.IS_VIDEO_FILE
    ).start()

    # Load models
    app.state.detector = YoloDetector()
    app.state.tracker = PersonTracker()
    app.state.counter = PeopleCounter()

    # Set cumulative targets in local cache matching DB records
    app.state.counter.entries = app.state.entries
    app.state.counter.exits = app.state.exits

    # Start loop thread
    app.state.loop_running = True
    app.state.processing_thread = threading.Thread(
        target=process_camera_loop,
        args=(app,),
        name="VisionLoopWorker",
        daemon=True
    )
    app.state.processing_thread.start()

    yield

    # Shutdown tasks
    logger.info("Shutting down background services...")
    app.state.loop_running = False
    
    if hasattr(app.state, "processing_thread") and app.state.processing_thread.is_alive():
        app.state.processing_thread.join(timeout=3.0)
        logger.info("Processing loop thread joined.")

    if hasattr(app.state, "camera") and app.state.camera is not None:
        app.state.camera.stop()
        logger.info("Camera feed released.")

    logger.info("Application shutdown sequence complete.")


# Initialize app
app = FastAPI(
    title="PeopleVision AI REST API",
    description="Real-time computer vision API for tracking and counting people.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include endpoint routes
app.include_router(router, prefix="/api/v1")

# Legacy/Root endpoints for backward compatibility
@app.get("/")
def read_root():
    return {
        "app": "peoplevision-ai",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "count": "/api/v1/count",
            "stats": "/api/v1/stats",
            "reset": "/api/v1/reset",
            "video_feed": "/api/v1/video_feed"
        }
    }

# Mapping specific root routes requested by the user
@app.get("/health")
def root_health(db: Session = Depends(get_db)):
    return get_health(db)

@app.get("/stats")
def root_stats(request: Request, db: Session = Depends(get_db)):
    return get_stats(request, db)

@app.get("/count")
def root_count(request: Request):
    return get_count(request)

@app.post("/reset")
def root_reset(request: Request, clear_db: bool = False, db: Session = Depends(get_db)):
    return post_reset(request, clear_db, db)
