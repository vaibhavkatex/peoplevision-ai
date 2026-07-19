from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db, CrossingEvent

# Configure a temporary file database for testing to avoid SQLite memory pooling isolation issues
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_temp.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override FastAPI db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create testing client
client = TestClient(app)

import os

@pytest.fixture(autouse=True)
def setup_db():
    """Initializes schema before each test and drops it afterward."""
    # Ensure any residual test file is removed
    if os.path.exists("./test_temp.db"):
        try:
            os.remove("./test_temp.db")
        except Exception:
            pass
            
    Base.metadata.create_all(bind=engine)
    # Seed default state attributes on app.state
    app.state.current_count = 0
    app.state.entries = 0
    app.state.exits = 0
    app.state.fps = 0.0
    app.state.confidence_threshold = 0.4
    app.state.counter = MagicMock()
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists("./test_temp.db"):
        try:
            os.remove("./test_temp.db")
        except Exception:
            pass


def test_health_endpoint():
    """Checks that /health endpoint is operational and reports database status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database"] == "healthy"


def test_count_endpoint_values():
    """Checks that /count returns live numbers cached in app state."""
    app.state.current_count = 4
    app.state.entries = 25
    app.state.exits = 12
    app.state.fps = 24.5

    response = client.get("/api/v1/count")
    assert response.status_code == 200
    data = response.json()
    assert data["current_occupancy"] == 4
    assert data["entries"] == 25
    assert data["exits"] == 12
    assert data["fps"] == 24.5


def test_config_update_endpoint():
    """Checks that posting to /config updates the confidence threshold in app state."""
    response = client.post("/api/v1/config", json={"confidence_threshold": 0.65})
    assert response.status_code == 200
    assert response.json()["confidence_threshold"] == 0.65
    assert app.state.confidence_threshold == 0.65

    # Check validation error for bad ranges
    response = client.post("/api/v1/config", json={"confidence_threshold": 1.5})
    assert response.status_code == 422  # Validation error


def test_reset_counts_endpoint():
    """Checks that /reset clears in-memory tracking numbers and triggers counter reset."""
    app.state.current_count = 10
    app.state.entries = 50
    app.state.exits = 40

    response = client.post("/api/v1/reset?clear_db=false")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify memory variables reset
    assert app.state.current_count == 0
    assert app.state.entries == 0
    assert app.state.exits == 0
    # Verify counter object's reset method was triggered
    app.state.counter.reset_counts.assert_called_once()


def test_reset_counts_with_database_truncate():
    """Checks that /reset with clear_db=true deletes rows in the database."""
    # Seed an event
    db = TestingSessionLocal()
    event = CrossingEvent(direction="entry", tracker_id=99, camera_id="test_cam")
    db.add(event)
    db.commit()
    db.close()

    # Verify event exists in stats
    resp_before = client.get("/api/v1/stats")
    assert resp_before.json()["total_entries"] == 1

    # Trigger reset with database clearing
    response = client.post("/api/v1/reset?clear_db=true")
    assert response.status_code == 200

    # Verify event is deleted in database
    resp_after = client.get("/api/v1/stats")
    assert resp_after.json()["total_entries"] == 0
    assert len(resp_after.json()["history"]) == 0
