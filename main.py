from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Ctrl+Sit - Posture Detection API")

@app.get("/")
def root():
    return {"message": "Welcome to Ctrl+Sit API. Upload a video to analyze posture."}

