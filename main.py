import os
import time
import numpy as np
import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2

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
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("assets/style.css")

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

# =============================================================================
# LOAD OPENCV CASCADE CLASSIFIERS
# =============================================================================
@st.cache_resource
def load_cascades():
    """Load OpenCV's pre-trained cascade classifiers"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

FACE_CASCADE, EYE_CASCADE = load_cascades()

# =============================================================================
# CORE LOGIC
# =============================================================================

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.alert_status = False
        self.ear_value = 0.0
        self.eye_threshold = 2  # Minimum eyes to detect (2 = both eyes open)
        self.frame_check = 20
        self.eyes_detected = 0

    def update_settings(self, threshold, frames):
        self.eye_threshold = threshold
        self.frame_check = frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        current_alert = False
        self.eyes_detected = 0

        # Detect faces
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Region of interest for eyes (upper half of face)
            roi_gray = gray[y:y+int(h*0.65), x:x+w]
            roi_color = img[y:y+int(h*0.65), x:x+w]

            # Detect eyes within face region
            eyes = EYE_CASCADE.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(25, 25)
            )

            self.eyes_detected = len(eyes)

            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Calculate pseudo-EAR based on eye detection
            # If both eyes detected = high EAR (awake), if fewer = low EAR (drowsy)
            if self.eyes_detected >= 2:
                self.ear_value = 0.35  # Eyes open
                self.frame_count = 0
            elif self.eyes_detected == 1:
                self.ear_value = 0.22  # One eye detected
                self.frame_count += 1
            else:
                self.ear_value = 0.15  # No eyes detected (closed or looking away)
                self.frame_count += 1

            # Check for drowsiness
            if self.frame_count >= self.frame_check:
                current_alert = True
                cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show eye count on frame
            status_text = f"Eyes: {self.eyes_detected}"
            cv2.putText(img, status_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # No face detected
        if len(faces) == 0:
            cv2.putText(img, "No face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            self.ear_value = 0.0
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
                    <div class="metric-label">Eye Status</div>
                    <div class="metric-value">--</div>
                </div>
             """, unsafe_allow_html=True)
        with m_col2:
             alert_count_placeholder = st.empty()
             alert_count_placeholder.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Eyes Detected</div>
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
            frames = st.slider("Alert Delay (frames)", 5, 50, 20,
                help="Number of consecutive frames with closed eyes before alert")

            if ctx.video_processor:
                ctx.video_processor.update_settings(2, frames)

        if ctx.state.playing:
            start_time = time.time() if 'start_time' not in st.session_state else st.session_state.start_time
            if 'start_time' not in st.session_state:
                st.session_state.start_time = start_time
                st.session_state.total_alerts = 0

            while ctx.state.playing:
                if ctx.video_processor:
                    ear_val = ctx.video_processor.ear_value
                    eyes = ctx.video_processor.eyes_detected

                    # Update eye status
                    if eyes >= 2:
                        status = "Open"
                        color = "#00d4ff"
                    elif eyes == 1:
                        status = "Partial"
                        color = "#ffa502"
                    else:
                        status = "Closed"
                        color = "#ff4757"

                    ear_placeholder.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Eye Status</div>
                            <div class="metric-value" style="color:{color};">{status}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    alert_count_placeholder.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Eyes Detected</div>
                            <div class="metric-value">{eyes}</div>
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
