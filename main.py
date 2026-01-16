import os
import time
import numpy as np
import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2

# Try to import MediaPipe for face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="DrowseGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except FileNotFoundError:
    pass

# Load Audio
@st.cache_data
def load_alert_sound():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(script_dir, "music.wav")
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

alert_sound_b64 = load_alert_sound()

# MediaPipe face mesh indices for eyes
# Left eye indices (from MediaPipe face mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# =============================================================================
# CORE LOGIC
# =============================================================================

def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio using 6 eye landmark points"""
    # Vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    # EAR formula
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.alert_status = False
        self.ear_value = 0.0
        self.ear_threshold = 0.25
        self.frame_check = 20

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None

    def update_settings(self, threshold, frames):
        self.ear_threshold = threshold
        self.frame_check = frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            cv2.putText(img, "Face detection unavailable", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(img, "Running in demo mode", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return frame.from_ndarray(img, format="bgr24")

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        current_alert = False
        h, w = img.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                landmarks = face_landmarks.landmark

                # Get left eye points
                left_eye_points = np.array([
                    [landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE
                ])

                # Get right eye points
                right_eye_points = np.array([
                    [landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE
                ])

                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                ear = (left_ear + right_ear) / 2.0
                self.ear_value = ear

                # Draw eye contours
                left_eye_int = left_eye_points.astype(np.int32)
                right_eye_int = right_eye_points.astype(np.int32)
                cv2.polylines(img, [left_eye_int], True, (0, 255, 0), 1)
                cv2.polylines(img, [right_eye_int], True, (0, 255, 0), 1)

                # Check for drowsiness
                if ear < self.ear_threshold:
                    self.frame_count += 1
                    if self.frame_count >= self.frame_check:
                        current_alert = True
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.frame_count = 0

        self.alert_status = current_alert
        return frame.from_ndarray(img, format="bgr24")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_brand_header():
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🛡️</div>
        <div class="brand-text">
            <h1>DrowseGuard</h1>
            <p>Driver Fatigue Monitoring System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    render_brand_header()

    if not MEDIAPIPE_AVAILABLE:
        st.warning("Face detection libraries are not available. Running in demo mode.")

    col_video, col_stats = st.columns([2, 1], gap="medium")

    with col_video:
        st.markdown("""
        <div class="video-container">
            <div class="video-header">
                <div class="live-indicator">● Live Feed</div>
                <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">Camera 0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_stats:
        status_placeholder = st.empty()
        audio_placeholder = st.empty()

        m_col1, m_col2 = st.columns(2)
        with m_col1:
             ear_placeholder = st.empty()
             ear_placeholder.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Eye Aspect Ratio</div>
                    <div class="metric-value">--</div>
                </div>
             """, unsafe_allow_html=True)
        with m_col2:
             alert_count_placeholder = st.empty()
             alert_count_placeholder.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Total Alerts</div>
                    <div class="metric-value">0</div>
                </div>
             """, unsafe_allow_html=True)

        session_placeholder = st.empty()
        session_placeholder.markdown("""
            <div class="metric-box">
                <div class="metric-label">Session Time</div>
                <div class="metric-value" style="color:white;">00:00:00</div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("Configuration", expanded=True):
            threshold = st.slider("EAR Sensitivity", 0.15, 0.35, 0.25, 0.01)
            frames = st.slider("Alert Delay (frames)", 5, 50, 20)

            if ctx.video_processor:
                ctx.video_processor.update_settings(threshold, frames)

        if ctx.state.playing:
            start_time = time.time() if 'start_time' not in st.session_state else st.session_state.start_time
            if 'start_time' not in st.session_state:
                st.session_state.start_time = start_time
                st.session_state.total_alerts = 0

            while ctx.state.playing:
                if ctx.video_processor:
                    ear_val = ctx.video_processor.ear_value
                    ear_placeholder.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Eye Aspect Ratio</div>
                            <div class="metric-value">{ear_val:.3f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    if ctx.video_processor.alert_status:
                        status_placeholder.markdown("""
                            <div class="css-card" style="text-align:center; border-color: #ff4757;">
                                <div class="status-big-icon" style="color:#ff4757;">💤</div>
                                <div class="status-text-large" style="color:#ff4757;">DROWSINESS!</div>
                                <div style="color:rgba(255,255,255,0.5); font-size:0.9rem; margin-top:0.5rem;">Wake Up!</div>
                            </div>
                        """, unsafe_allow_html=True)

                        if alert_sound_b64:
                            unique_id = f"audio_{int(time.time() * 10)}"
                            audio_placeholder.markdown(
                                f'<div id="{unique_id}"><audio autoplay><source src="data:audio/wav;base64,{alert_sound_b64}"></audio></div>',
                                unsafe_allow_html=True
                            )

                    else:
                        status_placeholder.markdown("""
                            <div class="css-card" style="text-align:center;">
                                <div class="status-big-icon">👁️</div>
                                <div class="status-text-large">ACTIVE</div>
                                <div style="color:rgba(255,255,255,0.5); font-size:0.9rem; margin-top:0.5rem;">Monitoring Driver</div>
                            </div>
                        """, unsafe_allow_html=True)
                        audio_placeholder.empty()

                    elapsed = int(time.time() - st.session_state.start_time)
                    mins, secs = divmod(elapsed, 60)
                    hrs, mins = divmod(mins, 60)
                    session_placeholder.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Session Time</div>
                            <div class="metric-value" style="color:white;">{hrs:02d}:{mins:02d}:{secs:02d}</div>
                        </div>
                    """, unsafe_allow_html=True)

                time.sleep(0.1)
        else:
            if 'start_time' in st.session_state:
                del st.session_state.start_time

            status_placeholder.markdown("""
                <div class="css-card" style="text-align:center;">
                    <div class="status-big-icon" style="opacity:0.5;">📷</div>
                    <div class="status-text-large" style="opacity:0.5;">STANDBY</div>
                    <div style="color:rgba(255,255,255,0.3); font-size:0.9rem; margin-top:0.5rem;">Ready to monitor</div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
