from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import time
import math as m
import MediaPipe as mp

app = FastAPI(title="Ctrl+Sit - Posture Detection API")

@app.get("/")
def root():
    return {"message": "Welcome to Ctrl+Sit API. Upload a video to analyze posture."}

