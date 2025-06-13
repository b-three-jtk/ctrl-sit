import cv2
import mediapipe as mp
import numpy as np
import math as m
import posture_alert as pa
import os
import sys
import requests
from main import detect_posture, pose, fps

def main():
    import streamlit as st
    global good_frames, bad_frames

    good_frames = 0
    bad_frames = 0

    st.set_page_config(page_title="Ctrl+Sit - Posture Monitor", layout="centered")
    st.title("🪑 Ctrl+Sit - Real-time Posture Detection")
    st.markdown("Mulai kamera dan pastikan postur dudukmu tetap baik.")

    uploaded_file = st.file_uploader("Unggah gambar atau video", type=["jpg", "png", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            response = requests.post(
                "http://localhost:8000/detect-posture/",
                files={"file": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                result = response.json()
                good_frames = result["good_frames"]
                bad_frames = result["bad_frames"]
                posture = "Good" if good_frames > bad_frames else "Bad"

                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Detected Posture: {posture}", use_column_width=True)

                if posture == "Bad":
                    pa.handle_posture_audio(True)
                    st.warning("⚠️ Postur buruk terdeteksi, silakan sesuaikan posisi duduk Anda.")
                else:
                    pa.handle_posture_audio(False)
                    st.success("✅ Postur baik terdeteksi. Pertahankan!")
            else:
                st.error("❌ Gagal memanggil API FastAPI.")

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
                    st.warning("⚠️ Postur buruk terdeteksi, silakan sesuaikan posisi duduk Anda.")
                elif posture == "Good":
                    pa.handle_posture_audio(False)
                    st.success("✅ Postur baik terdeteksi. Pertahankan!")
            else:
                st.error("Gagal memproses video.")
            cap.release()
            os.remove(temp_file)

    run = st.button('Mulai Deteksi Postur (Live Camera)')
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
        st.success("Kamera dihentikan.")

if __name__ == "__main__":
    main()
