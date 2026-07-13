import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

app = FastAPI()

# CORS इनेबल करना ताकि आपकी Django ऐप या GitHub Pages से रिक्वेस्ट आ सके
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO मॉडल को लोड करना (सर्वर स्टार्ट होते ही एक बार लोड होगा)
try:
    model = YOLO("yolo11n-seg.pt")
except Exception as e:
    print(f"YOLO Load Error: {e}")

@app.get("/")
def home():
    return {"status": "FastAPI is running perfectly on Render!"}

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # 1. अपलोड की गई इमेज को रीड करना
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Invalid Image Data"}

        # 2. 🚀 यहाँ अपना YOLO + MediaPipe का कैलकुलेशन वाला पुराना लॉजिक डालो
        # (जैसे: results = model(img) और MediaPipe से लैंडमार्क्स निकालना)
        
        # 3. फाइनल रिजल्ट को JSON फॉर्मेट में वापस भेजना
        # (इसे अपने असली कैलकुलेशन वेरिएबल्स से बदल लेना)
        result_json = {
            "status": "success",
            "measurements": {
                "length": 14.8,
                "width": 2.4,
                "unit": "cm"
            }
        }
        return result_json

    except Exception as e:
        return {"status": "error", "message": str(e)}
