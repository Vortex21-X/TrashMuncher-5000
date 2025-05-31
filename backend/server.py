import os
import asyncio
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from PIL import Image
import io
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Live Trash Finder API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URL)
db = client.trash_finder

# Global variables for model
model = None
device = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

async def load_trash_model():
    """Load the pre-trained trash detection model"""
    global model, device
    try:
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        try:
            # Try to download a trash-specific model first
            # This is a TACO dataset trained model available on Hugging Face
            from huggingface_hub import hf_hub_download
            
            # Try to get a trash-specific YOLO model
            try:
                model_path = hf_hub_download(
                    repo_id="keremberke/yolov8n-trash-detection",
                    filename="yolov8n-trash-detection.pt"
                )
                model = YOLO(model_path)
                logger.info("Trash-specific YOLO model loaded successfully from HuggingFace")
            except Exception as e:
                logger.info(f"Trash-specific model not available: {e}, using base model with custom classes")
                # Fall back to base model
                model = YOLO('yolov8n.pt')
                logger.info("Base YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Final fallback to base YOLO model
            model = YOLO('yolov8n.pt')
            logger.info("Loaded fallback YOLO model")
            
        # Move model to device
        model.to(device)
        logger.info("Model loaded and ready for inference")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load detection model")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    await load_trash_model()
    logger.info("Application started successfully")

def preprocess_frame(frame_data):
    """Preprocess the frame for model inference"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return frame
    except Exception as e:
        logger.error(f"Error preprocessing frame: {e}")
        return None

def detect_trash(frame):
    """Detect trash in the frame using YOLO model"""
    global model
    
    if model is None:
        return []
    
    try:
        # Run inference
        results = model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = model.names[class_id]
                    
                    # Define potential trash classes
                    # If using a trash-specific model, these will be trash categories
                    # If using base COCO model, these are items that could be trash
                    trash_classes = [
                        'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                        'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                        'cell phone', 'book', 'scissors', 'teddy bear',
                        'toothbrush', 'hair drier', 'backpack', 'handbag',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard', 'tennis racket', 'wine glass'
                    ]
                    
                    # Check if detected class could be trash
                    is_potential_trash = (
                        confidence > 0.25 and  # Lower threshold for demo
                        (class_name.lower() in [tc.lower() for tc in trash_classes] or
                         'trash' in class_name.lower() or
                         'litter' in class_name.lower() or
                         'waste' in class_name.lower() or
                         'garbage' in class_name.lower() or
                         'cigarette' in class_name.lower() or
                         'can' in class_name.lower() or
                         'bag' in class_name.lower())
                    )
                    
                    if is_potential_trash:
                        # Determine trash type based on detection
                        if 'bottle' in class_name.lower():
                            trash_type = "Plastic Bottle"
                        elif 'cup' in class_name.lower():
                            trash_type = "Disposable Cup"
                        elif 'can' in class_name.lower():
                            trash_type = "Aluminum Can"
                        elif 'bag' in class_name.lower():
                            trash_type = "Plastic Bag"
                        elif 'cigarette' in class_name.lower():
                            trash_type = "Cigarette Butt"
                        elif 'trash' in class_name.lower() or 'garbage' in class_name.lower():
                            trash_type = class_name.title()
                        elif any(food in class_name.lower() for food in ['banana', 'apple', 'orange', 'sandwich', 'pizza']):
                            trash_type = f"Food Waste: {class_name.title()}"
                        else:
                            trash_type = f"Potential Litter: {class_name.title()}"
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_name': trash_type,
                            'class_id': class_id,
                            'original_class': class_name
                        })
        
        return detections
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return []

@app.websocket("/api/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time trash detection"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_info = json.loads(data)
            
            # Preprocess frame
            frame = preprocess_frame(frame_info['frame'])
            
            if frame is not None:
                # Detect trash
                detections = detect_trash(frame)
                
                # Send results back to client
                response = {
                    'detections': detections,
                    'timestamp': frame_info.get('timestamp'),
                    'frame_count': frame_info.get('frame_count', 0)
                }
                
                await manager.send_personal_message(json.dumps(response), websocket)
            else:
                # Send error response
                error_response = {
                    'error': 'Failed to process frame',
                    'detections': []
                }
                await manager.send_personal_message(json.dumps(error_response), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.get("/api/model-info")
async def model_info():
    """Get information about the loaded model"""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "YOLOv8",
        "device": device,
        "classes": list(model.names.values()) if hasattr(model, 'names') else [],
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
