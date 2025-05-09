from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ultralytics import YOLO
import cv2

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO("yolov8n.pt")  # Make sure you place the model in this folder

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)[0]
        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO("yolov8n.pt")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    print("🖼️ Frame received for detection")
    contents = await file.read()

    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)

    print(f"📏 Frame shape: {image.shape}")

    results = model(image)[0]
    boxes = results.boxes

    detections = []
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy.tolist()[0]
            label = results.names[int(box.cls)]
            conf = float(box.conf)
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": xyxy
            })

    print(f"✅ Detected {len(detections)} objects")
    return JSONResponse({"detections": detections})
