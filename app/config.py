import logging
import os
from typing import Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings class using Pydantic BaseSettings."""
    
    # Allow settings to be overridden by environment variables or a .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, etc.)")

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./peoplevision.db",
        description="SQLAlchemy database URL connection string"
    )

    # Camera / Video Source
    CAMERA_SOURCE: Union[int, str] = Field(
        default=0,
        description="USB camera index (e.g. 0), RTSP URL, or local video path"
    )
    IS_VIDEO_FILE: bool = Field(
        default=False,
        description="Flag indicating if the source is a local video file (for loop simulation)"
    )

    # YOLO Detector
    YOLO_MODEL: str = Field(default="yolov8n.pt", description="YOLO weights file")
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.4,
        description="Confidence threshold for person class filtering (0.0 to 1.0)"
    )
    TARGET_CLASS_ID: int = Field(
        default=0,
        description="COCO class identifier for person class (default is 0)"
    )
    DEVICE: str = Field(default="cpu", description="Hardware device ('cpu', 'cuda', 'mps')")

    # Tracker Settings
    TRACKER_FPS: int = Field(default=30, description="FPS for ByteTrack speed estimation")
    TRACK_BUFFER: int = Field(default=30, description="ByteTrack memory buffer length")

    # Counting Line (relative percentages of frame dimensions)
    LINE_START_X_PCT: float = Field(default=0.0, ge=0.0, le=1.0)
    LINE_START_Y_PCT: float = Field(default=0.5, ge=0.0, le=1.0)
    LINE_END_X_PCT: float = Field(default=1.0, ge=0.0, le=1.0)
    LINE_END_Y_PCT: float = Field(default=0.5, ge=0.0, le=1.0)

    # Server / REST API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_URL: str = Field(default="http://localhost:8000")

    @property
    def parsed_camera_source(self) -> Union[int, str]:
        """Parses the camera source string into integer if it is a digit.

        Returns:
            Union[int, str]: Int if USB camera index, else string.
        """
        src = str(self.CAMERA_SOURCE).strip()
        if src.isdigit():
            return int(src)
        return src

# Instantiate global settings
settings = Settings()

# Configure global logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("peoplevision-ai")
logger.info(f"Loaded configuration settings. Database URL: {settings.DATABASE_URL}")
