from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import math as m
import posture_alert as pa
import os
import sys

# Inisialisasi FastAPI
app = FastAPI(title="Ctrl+Sit Posture Detection API")

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

good_frames = 0
bad_frames = 0
fps = 30  # asumsi sementara

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate the angle between three points (x1,y1), (x2,y2), (x3,y3) with (x2,y2) as the vertex.
    Returns the angle in degrees.
    """
    vector1_x = x1 - x2
    vector1_y = y1 - y2
    vector2_x = x3 - x2
    vector2_y = y3 - y2
    
    dot_product = vector1_x * vector2_x + vector1_y * vector2_y
    magnitude1 = m.sqrt(vector1_x ** 2 + vector1_y ** 2)
    magnitude2 = m.sqrt(vector2_x ** 2 + vector2_y ** 2)
    
    if magnitude1 * magnitude2 == 0:
        return 0
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle = m.degrees(m.acos(cos_theta))
    return abs(angle)

def detect_posture(image, pose, fps, good_frames, bad_frames):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    height, width, _ = image_bgr.shape

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]

        shoulder_x = int((l_shoulder.x + r_shoulder.x) / 2 * width)
        shoulder_y = int((l_shoulder.y + r_shoulder.y) / 2 * height)
        ear_x = int((l_ear.x + r_ear.x) / 2 * width)
        ear_y = int((l_ear.y + r_ear.y) / 2 * height)
        hip_x = int((l_hip.x + r_hip.x) / 2 * width)
        hip_y = int((l_hip.y + r_hip.y) / 2 * height)

        cv2.circle(image_bgr, (shoulder_x, shoulder_y), 7, (0, 255, 0), -1)
        cv2.circle(image_bgr, (ear_x, ear_y), 7, (0, 255, 0), -1)
        cv2.circle(image_bgr, (hip_x, hip_y), 7, (0, 255, 0), -1)
        
        cv2.line(image_bgr, (ear_x, ear_y), (shoulder_x, shoulder_y), (255, 255, 0), 2)
        cv2.line(image_bgr, (shoulder_x, shoulder_y), (hip_x, hip_y), (255, 0, 255), 2)

        neck_angle = calculate_angle(ear_x, ear_y, shoulder_x, shoulder_y, shoulder_x, shoulder_y + 100)
        torso_angle = calculate_angle(shoulder_x, shoulder_y, hip_x, hip_y, hip_x, hip_y + 100)

        if neck_angle > 20 or torso_angle > 10:
            bad_frames += 1
        else:
            good_frames += 1

        posture = "Good" if good_frames > bad_frames else "Bad"
        color = (0, 255, 0) if posture == "Good" else (0, 0, 255)

        good_time = good_frames / fps
        bad_time = bad_frames / fps

        cv2.putText(image_bgr, f"Posture: {posture}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image_bgr, f"Neck Inclination: {int(neck_angle)} deg", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_bgr, f"Torso Inclination: {int(torso_angle)} deg", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_bgr, f"Good Posture Time: {int(good_time)} sec", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image_bgr, f"Bad Posture Time: {int(bad_time)} sec", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        try:
            if bad_time > 20:
                if not st.session_state.get('alert_sent', False):
                    pa.handle_posture_audio(True)
                    st.session_state['alert_sent'] = True
        except NameError:
            if bad_time > 20:
                pa.handle_posture_audio(True)

    return image_bgr, good_frames, bad_frames

@app.post("/detect-posture")
async def detect_posture_endpoint(file: UploadFile = File(...)):
    global good_frames, bad_frames
    good_frames = 0
    bad_frames = 0

    if file.content_type not in ["image/jpeg", "image/png", "video/mp4"]:
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png) or video (mp4)")

    file_bytes = await file.read()

    if file.content_type.startswith("image"):
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        frame, good_frames, bad_frames = detect_posture(image, pose, fps, good_frames, bad_frames)
        posture = "Good" if good_frames > bad_frames else "Bad"
        
        if posture == "Bad":
            pa.handle_posture_audio(True)
        elif posture == "Good":
            pa.handle_posture_audio(False)

        return JSONResponse(content={
            "status": "success",
            "posture": posture,
            "message": "Posture detection completed"
        })

    elif file.content_type.startswith("video"):
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_file)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail="Invalid video file")

        frame, good_frames, bad_frames = detect_posture(frame, pose, fps, good_frames, bad_frames)
        posture = "Good" if good_frames > bad_frames else "Bad"
        cap.release()
        os.remove(temp_file)

        if posture == "Bad":
            pa.handle_posture_audio(True)
        elif posture == "Good":
            pa.handle_posture_audio(False)

        return JSONResponse(content={
            "status": "success",
            "posture": posture,
            "message": "Posture detection completed (first frame analyzed)"
        })

# Fungsi untuk menjalankan aplikasi Streamlit
def run_streamlit_app():
    import streamlit as st
    global good_frames, bad_frames

    good_frames = 0
    bad_frames = 0

    st.set_page_config(page_title="Ctrl+Sit - Posture Monitor", layout="centered")
    st.title("🪑 Ctrl+Sit - Real-time Posture Detection")
    st.markdown("Mulai kamera dan pastikan postur dudukmu tetap baik.")

    # Tambahkan opsi untuk unggah file
    uploaded_file = st.file_uploader("Unggah gambar atau video", type=["jpg", "png", "mp4"])

    if uploaded_file is not None:
        # Proses file yang diunggah
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if uploaded_file.type.startswith("image"):
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                frame, good_frames, bad_frames = detect_posture(image, pose, fps, good_frames, bad_frames)
                posture = "Good" if good_frames > bad_frames else "Bad"
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Detected Posture: {posture}", use_column_width=True)
                if posture == "Bad":
                    pa.handle_posture_audio(True)
                    st.warning("⚠️ Bad posture detected. Please adjust your sitting position.")
                elif posture == "Good":
                    pa.handle_posture_audio(False)
                    st.success("✅ Good posture detected. Keep it up!")
            else:
                st.error("Gagal memproses gambar.")
        elif uploaded_file.type.startswith("video"):
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())
            cap = cv2.VideoCapture(temp_file)
            ret, frame = cap.read()
            if ret:
                frame, good_frames, bad_frames = detect_posture(frame, pose, fps, good_frames, bad_frames)
                posture = "Good" if good_frames > bad_frames else "Bad"
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Detected Posture: {posture} (Frame 1)", use_column_width=True)
                if posture == "Bad":
                    pa.handle_posture_audio(True)
                    st.warning("⚠️ Bad posture detected. Please adjust your sitting position.")
                elif posture == "Good":
                    pa.handle_posture_audio(False)
                    st.success("✅ Good posture detected. Keep it up!")
            else:
                st.error("Gagal memproses video.")
            cap.release()
            os.remove(temp_file)

    # Opsi kamera (tetap dipertahankan)
    run = st.button('Start Camera')
    frame_window = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame, good_frames, bad_frames = detect_posture(frame, pose, fps, good_frames, bad_frames)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.success("Camera stopped.")

# Jalankan aplikasi berdasarkan cara eksekusi
if __name__ == "__main__":
    if "streamlit" in sys.modules or "streamlit" in sys.argv[0]:
        run_streamlit_app()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)