import logging
from datetime import datetime, timezone
from typing import Dict, Generator, List, Tuple
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

logger = logging.getLogger("peoplevision-ai.database")

# Handle SQLite specific threading requirements
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Initialize SQLalchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CrossingEvent(Base):
    """SQLAlchemy model representing a line crossing event."""
    __tablename__ = "crossing_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    direction = Column(String(10), nullable=False)  # "entry" or "exit"
    tracker_id = Column(Integer, nullable=False)
    camera_id = Column(String(50), nullable=False)

    def to_dict(self) -> Dict:
        """Serializes the model into a standard dictionary.

        Returns:
            Dict: Dictionary representation of the event.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "tracker_id": self.tracker_id,
            "camera_id": self.camera_id
        }


def init_db() -> None:
    """Initializes database tables by creating them if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """Dependency injection generator to yield a database session.

    Yields:
        Session: Database session transaction block.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_crossing_event(db: Session, direction: str, tracker_id: int, camera_id: str) -> CrossingEvent:
    """Inserts a new line-crossing detection event into the database.

    Args:
        db (Session): Database session.
        direction (str): 'entry' or 'exit'.
        tracker_id (int): Unique tracking ID of the person.
        camera_id (str): Identifier of the camera source.

    Returns:
        CrossingEvent: The created database model instance.
    """
    event = CrossingEvent(
        direction=direction,
        tracker_id=tracker_id,
        camera_id=camera_id,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    logger.debug(f"Saved crossing event: ID={tracker_id}, direction={direction}")
    return event


def get_crossing_stats(db: Session) -> Dict[str, int]:
    """Aggregates total entry and exit counts from the database.

    Args:
        db (Session): Database session.

    Returns:
        Dict[str, int]: Dictionary with 'entries' and 'exits' keys.
    """
    entries = db.query(func.count(CrossingEvent.id)).filter(CrossingEvent.direction == "entry").scalar() or 0
    exits = db.query(func.count(CrossingEvent.id)).filter(CrossingEvent.direction == "exit").scalar() or 0
    return {"entries": entries, "exits": exits}


def get_historical_events(db: Session, limit: int = 100) -> List[CrossingEvent]:
    """Retrieves list of recent crossing events.

    Args:
        db (Session): Database session.
        limit (int): Maximum number of entries to fetch.

    Returns:
        List[CrossingEvent]: List of crossing database events.
    """
    return db.query(CrossingEvent).order_by(CrossingEvent.timestamp.desc()).limit(limit).all()


def clear_database_logs(db: Session) -> None:
    """Removes all recorded line-crossing records from the database table.

    Args:
        db (Session): Database session.
    """
    try:
        db.query(CrossingEvent).delete()
        db.commit()
        logger.info("Cleared all database logs.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing database logs: {e}")
        raise
