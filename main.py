from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import math as m
import posture_alert as pa
import numpy as np

app = FastAPI(title="Ctrl+Sit Posture Detection API")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

good_frames = 0
bad_frames = 0
fps = 30
alert_sent = False

def calculate_neck_angle(x1, y1, x2, y2, x3, y3):
    """
    Hitung sudut antara vektor (x1,y1) → (x2,y2) dan (x2,y2) → (x3,y3).
    Menggunakan rumus cosinus sudut.
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

def calculate_torso_angle(x1, y1, x2, y2):
    """
    Hitung sudut antara vektor (x1,y1) → (x2,y2) terhadap garis vertikal ke bawah (0, 1).
    """
    dx = x2 - x1
    dy = y2 - y1

    vector_magnitude = m.sqrt(dx**2 + dy**2)
    if vector_magnitude == 0:
        return 0

    cos_theta = dy / vector_magnitude
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle = m.degrees(m.acos(cos_theta))
    return abs(angle)

def detect_posture(image, pose, fps, good_frames, bad_frames):
    global alert_sent
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

        neck_angle = calculate_neck_angle(ear_x, ear_y, shoulder_x, shoulder_y, shoulder_x, shoulder_y - 100)
        torso_angle = calculate_torso_angle(shoulder_x, shoulder_y, hip_x, hip_y)

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
        cv2.putText(image_bgr, f"Good Posture Time: {int(good_time)} sec", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image_bgr, f"Bad Posture Time: {int(bad_time)} sec", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        try:
            if bad_time > 10:
                pa.handle_posture_audio(True)
                good_frames = 0
                bad_frames = 0
            elif good_time > 30:
                pa.handle_posture_audio(False)
                good_frames = 0
                bad_frames = 0
        except Exception as e:
            print(f"Error playing audio: {e}")
    return image_bgr, good_frames, bad_frames

@app.post("/detect-posture/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    global good_frames, bad_frames
    result_image, good_frames, bad_frames = detect_posture(image, pose, fps, good_frames, bad_frames)

    _, buffer = cv2.imencode('.jpg', result_image)
    return JSONResponse(content={"status": "success", "good_frames": good_frames, "bad_frames": bad_frames})
