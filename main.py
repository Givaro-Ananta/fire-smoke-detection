"""
Fire & Smoke Detection API — FastAPI Backend
Serves YOLOv8 model inference for fire & smoke detection.
"""

import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ========================
# APP SETUP
# ========================
app = FastAPI(
    title="Fire & Smoke Detection API",
    description="Upload an image to detect fire and smoke using YOLOv8",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# MODEL LOADING
# ========================
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "runs", "detect", "fire_smoke_detector", "weights", "best.pt")
)

model = None

# Color map for classes (BGR for OpenCV)
CLASS_COLORS = {
    "fire": (0, 69, 255),      # Orange-Red
    "smoke": (180, 180, 180),   # Gray
}
DEFAULT_COLOR = (0, 255, 0)     # Green


@app.on_event("startup")
async def load_model():
    """Load the YOLOv8 model on startup."""
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at: {MODEL_PATH}")
        print("  Please train the model first using train.py")
        print("  Or set MODEL_PATH environment variable")
        # Load a pretrained YOLOv8n as fallback for testing
        model = YOLO("yolov8n.pt")
        print("  Loaded yolov8n.pt as fallback")


# ========================
# ENDPOINTS
# ========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/detect")
async def detect_fire_smoke(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    """
    Detect fire and smoke in uploaded image.
    Returns detection results and annotated image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run inference
        results = model.predict(
            source=img,
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

        # Process results
        detections = []
        annotated_img = img.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get detection info
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]

                x1, y1, x2, y2 = map(int, xyxy)

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    }
                })

                # Draw bounding box on image
                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                thickness = 3
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

                # Draw label background
                label = f"{cls_name} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                cv2.rectangle(
                    annotated_img,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 5, y1),
                    color, -1
                )
                cv2.putText(
                    annotated_img, label,
                    (x1 + 2, y1 - 5),
                    font, font_scale, (255, 255, 255), font_thickness
                )

        # Convert annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Build summary
        class_summary = {}
        for det in detections:
            cls = det["class"]
            class_summary[cls] = class_summary.get(cls, 0) + 1

        return JSONResponse({
            "success": True,
            "total_detections": len(detections),
            "detections": detections,
            "class_summary": class_summary,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "model_path": MODEL_PATH,
        "task": "detect",
        "names": model.names if hasattr(model, "names") else {}
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
