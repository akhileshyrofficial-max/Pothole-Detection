# server/app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

app = FastAPI()

# Load YOLOv8 model (use a trained model, or yolov8n.pt for testing)
model = YOLO("yolov8n.pt")  # replace with your pothole detection model (e.g., pothole.pt)

@app.get("/")
def root():
    return {"message": "Pothole Detection API is running!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model.predict(image)

    # Parse results (bounding boxes, confidence, class)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist()
            })

    return JSONResponse(content={"detections": detections})
