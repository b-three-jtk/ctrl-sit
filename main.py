import streamlit as st
import cv2
import mediapipe as mp
import time
import math as m

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

good_frames = 0
bad_frames = 0
send_alert = False
fps = 30  # asumsi sementara

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
        l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]

        shouder_x = int((l_shoulder.x + r_shoulder.x) / 2 * width)
        shouder_y = int((l_shoulder.y + r_shoulder.y) / 2 * height)
        ear_x = int(l_ear.x * width)
        ear_y = int(l_ear.y * height)
        hip_x = int(l_hip.x * width)
        hip_y = int(l_hip.y * height)

        cv2.circle(image_bgr, (shouder_x, shouder_y), 7, (0, 255, 0), -1)
        cv2.circle(image_bgr, (ear_x, ear_y), 7, (0, 255, 0), -1)
        cv2.circle(image_bgr, (hip_x, hip_y), 7, (0, 255, 0), -1)

        neck_angle = m.degrees(m.atan2(ear_y - shouder_y, ear_x - shouder_x))
        torso_angle = m.degrees(m.atan2(hip_y - shouder_y, hip_x - shouder_x))

        neck_angle = abs(neck_angle - 90)
        torso_angle = abs(torso_angle - 90)

        if neck_angle > 40 or torso_angle > 10:
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

        if bad_time > 20:
            send_alert = True

    return image_bgr, good_frames, bad_frames


st.set_page_config(page_title="Ctrl+Sit - Posture Monitor", layout="centered")
st.title("🪑 Ctrl+Sit - Real-time Posture Detection")
st.markdown("Mulai kamera dan pastikan postur dudukmu tetap baik.")

run = st.checkbox('Start Camera')
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

