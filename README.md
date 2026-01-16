---
title: DrowseGuard - Driver Drowsiness Detection
emoji: 🛡️
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: main.py
pinned: false
license: mit
---

# DrowseGuard - Driver Drowsiness Detection

Real-time driver fatigue monitoring system using computer vision

## About

DrowseGuard monitors eye movements in real-time to detect fatigue and prevent accidents. Using MediaPipe Face Mesh and the Eye Aspect Ratio (EAR) algorithm, it provides instant alerts when drowsiness is detected.

### Features

- Real-time Detection - WebRTC-based browser video processing
- Eye Aspect Ratio (EAR) - Scientifically validated drowsiness metric
- Multi-modal Alerts - Visual overlays + audio warnings
- Configurable Sensitivity - Adjustable thresholds
- Live Dashboard - Monitor EAR values, alerts, and session time
- Browser-based - No installation required

## Quick Start

1. Click START in the video feed section
2. Grant webcam permission when prompted
3. Position your face in front of the camera
4. The system will:
   - Detect your face and eyes
   - Draw green contours around eyes
   - Monitor your Eye Aspect Ratio (EAR)
   - Alert you if drowsiness is detected

### Configuration

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| EAR Sensitivity | 0.15 - 0.35 | 0.25 | Lower = more sensitive |
| Alert Delay | 5 - 50 frames | 20 | Frames before alert |

## How It Works

### Eye Aspect Ratio (EAR) Algorithm

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

When eyes close, the vertical distance decreases while horizontal distance remains constant, causing EAR to drop (~40% reduction).

### Detection Pipeline

1. Face Detection - MediaPipe Face Mesh
2. Landmark Extraction - 468-point facial landmarks
3. EAR Calculation - Compute eye openness for both eyes
4. Threshold Check - Compare against threshold (default: 0.25)
5. Alert Trigger - If EAR < threshold for 20+ consecutive frames

## Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | Streamlit |
| Video Streaming | streamlit-webrtc, WebRTC |
| Computer Vision | OpenCV |
| Face Detection | MediaPipe Face Mesh |

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari (may have WebRTC issues)

## Privacy

- All video processing happens locally in your browser
- No video data is sent to servers or stored
- No data collection or analytics

## Local Development

```bash
# Clone repository
git clone <repo-url>
cd Driver-Drowsiness-Detection

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

## Disclaimer

This is a demonstration tool for educational purposes. It should NOT be used as the sole safety mechanism in vehicles. Always prioritize adequate rest and avoid driving when fatigued.

## License

MIT License
