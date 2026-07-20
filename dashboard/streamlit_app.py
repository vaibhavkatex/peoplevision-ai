import os
import time
from datetime import datetime

try:
    import pandas as pd  # type: ignore[import-not-found]
except ImportError:
    pd = None

import requests
import streamlit as st

# Setup page layout and style
st.set_page_config(
    page_title="PeopleVision AI - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Read API base URL from environment
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
logger_url = f"{API_URL}/api/v1"

# Custom CSS for modern premium dashboard styling (Dark theme integration)
st.markdown(
    """
    <style>
    .main {
        background-color: #0f1115;
        color: #e6e8eb;
    }
    .stMetric {
        background-color: #1b1e24;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric:nth-child(2) {
        border-left-color: #10b981;
    }
    .stMetric:nth-child(3) {
        border-left-color: #ef4444;
    }
    .camera-active {
        color: #10b981;
        font-weight: bold;
    }
    .camera-inactive {
        color: #ef4444;
        font-weight: bold;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_api_health() -> bool:
    """Verifies backend API connectivity.

    Returns:
        bool: Connection success.
    """
    try:
        r = requests.get(f"{logger_url}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def get_live_metrics() -> dict:
    """Retrieves live counting stats from the backend.

    Returns:
        dict: Live frame metrics.
    """
    try:
        r = requests.get(f"{logger_url}/count", timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"current_occupancy": 0, "entries": 0, "exits": 0, "fps": 0.0, "confidence_threshold": 0.4}


def get_historical_data() -> dict:
    """Fetches full aggregate databases statistics and history logs.

    Returns:
        dict: Full crossing logs.
    """
    try:
        r = requests.get(f"{logger_url}/stats", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"total_entries": 0, "total_exits": 0, "history": []}


# Header
st.markdown("## 👁️ PeopleVision AI — Real-time Video Analytics")
st.markdown("---")

# Verify connection
if not get_api_health():
    st.error(f"⚠️ Unable to connect to the PeopleVision API at `{API_URL}`. Please check if the backend is running.")
    if st.button("Retry Connection"):
        st.rerun()
    st.stop()

# Load current config state
current_metrics = get_live_metrics()
conf_default = float(current_metrics.get("confidence_threshold", 0.4))

# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure detector parameters and reset counters.")

# 1. Slider: YOLO Confidence
confidence_slider = st.sidebar.slider(
    "YOLO Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=conf_default,
    step=0.05,
    help="Minimum score needed to register a person detection bounding box."
)

# Update confidence threshold if slider changed
if confidence_slider != conf_default:
    try:
        requests.post(f"{logger_url}/config", json={"confidence_threshold": confidence_slider}, timeout=1.0)
        st.sidebar.success(f"Updated threshold to {confidence_slider:.2f}")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to update config: {e}")

# 2. Reset Counters
st.sidebar.subheader("Reset Counts")
clear_db_checkbox = st.sidebar.checkbox("Truncate Database Logs", value=False)
if st.sidebar.button("Reset System Counts", type="primary"):
    try:
        r = requests.post(f"{logger_url}/reset?clear_db={str(clear_db_checkbox).lower()}", timeout=2.0)
        if r.status_code == 200:
            st.sidebar.success("Counters successfully reset!")
            time.sleep(1.0)
            st.rerun()
        else:
            st.sidebar.error("Failed to reset counts.")
    except Exception as e:
        st.sidebar.error(f"Error resetting: {e}")

# 3. Export CSV Data
st.sidebar.subheader("Export Data")
hist_res = get_historical_data()
history_list = hist_res.get("history", [])

if len(history_list) > 0:
    df_history = pd.DataFrame(history_list)
    # Re-order columns for display
    if "timestamp" in df_history.columns:
        df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
    
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Crossing CSV",
        data=csv_data,
        file_name=f"peoplevision_crossing_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.sidebar.info("No crossing data available to export.")


# ------------------------------------------------------------------------------
# Main Dashboard Layout
# ------------------------------------------------------------------------------
col_feed, col_metrics = st.columns([5, 2])

# Left Column - Camera Feed
with col_feed:
    st.subheader("Live Processing Stream")
    # Stream the MJPEG video feed URL from the API
    video_url = f"{logger_url}/video_feed"
    st.image(video_url, caption="Real-time object tracking feed", use_container_width=True)

# Right Column - Metrics Cards
with col_metrics:
    st.subheader("Key Performance Metrics")
    
    # Placeholders for metric cards
    occupancy_placeholder = st.empty()
    entries_placeholder = st.empty()
    exits_placeholder = st.empty()
    fps_placeholder = st.empty()
    status_placeholder = st.empty()


# Bottom Section - Analytics Charts
st.subheader("📊 Analytics & History")
chart_placeholder = st.empty()
table_placeholder = st.empty()

# ------------------------------------------------------------------------------
# Live Update Loop for numbers & charts
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Dashboard auto-updates every 1 second.")

# We run a continuous loop to fetch numbers from the FastAPI state.
# If the user interacts with the slider or buttons, Streamlit interrupts the loop automatically,
# re-runs the script, and re-starts this loop.
while True:
    metrics = get_live_metrics()
    hist = get_historical_data()
    
    # Update metrics columns
    with col_metrics:
        occupancy_placeholder.metric(
            label="In Bounding Frame",
            value=metrics.get("current_occupancy", 0)
        )
        entries_placeholder.metric(
            label="Total Entries",
            value=metrics.get("entries", 0)
        )
        exits_placeholder.metric(
            label="Total Exits",
            value=metrics.get("exits", 0)
        )
        fps_placeholder.metric(
            label="Processing speed (FPS)",
            value=f"{metrics.get('fps', 0.0):.1f}"
        )
        
        # Camera Status
        try:
            status_check = requests.get(f"{logger_url}/health", timeout=1.0).json()
            is_connected = status_check.get("database") == "healthy" # proxy for backend
            status_text = "ACTIVE" if is_connected else "ERROR"
            status_class = "camera-active" if is_connected else "camera-inactive"
        except Exception:
            status_text = "DISCONNECTED"
            status_class = "camera-inactive"
            
        status_placeholder.markdown(
            f"Camera Source Status: <span class='{status_class}'>{status_text}</span>",
            unsafe_allow_html=True
        )

    # Render History Logs and Charts
    events = hist.get("history", [])
    if len(events) > 0:
        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Group crossings by time (e.g., last 10 minutes) or show simply count line chart
        df_sorted = df.sort_values(by="timestamp")
        
        # Draw dynamic chart
        # Build cumulative sums over time to graph trend lines
        df_sorted["entry_inc"] = df_sorted["direction"].apply(lambda x: 1 if x == "entry" else 0)
        df_sorted["exit_inc"] = df_sorted["direction"].apply(lambda x: 1 if x == "exit" else 0)
        df_sorted["Cumulative Entries"] = df_sorted["entry_inc"].cumsum()
        df_sorted["Cumulative Exits"] = df_sorted["exit_inc"].cumsum()
        
        chart_data = df_sorted[["timestamp", "Cumulative Entries", "Cumulative Exits"]].set_index("timestamp")
        
        with chart_placeholder:
            st.line_chart(chart_data, height=220)
            
        with table_placeholder:
            st.dataframe(
                df_sorted[["timestamp", "direction", "tracker_id", "camera_id"]]
                .sort_values(by="timestamp", ascending=False)
                .head(10),
                use_container_width=True
            )
    else:
        with chart_placeholder:
            st.info("No historical crossing trends logged yet.")
        with table_placeholder:
            st.info("Waiting for crossing events...")

    time.sleep(1.0)
