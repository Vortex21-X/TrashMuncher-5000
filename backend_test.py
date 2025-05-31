
import requests
import sys
import json
import websocket
import base64
import time
import os
from PIL import Image
import io
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrashFinderAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        # Get the backend URL from environment or use default
        self.base_url = os.environ.get('REACT_APP_BACKEND_URL', base_url)
        self.tests_run = 0
        self.tests_passed = 0
        self.ws_url = self.base_url.replace('http', 'ws') + '/api/ws/detect'
        self.ws = None
        self.ws_responses = []
        self.ws_connected = False
        self.ws_error = None

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        logger.info(f"Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                logger.info(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                logger.error(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    logger.error(f"Response: {response.text}")
                    return False, response.json() if response.text else {}
                except:
                    return False, {}

        except Exception as e:
            logger.error(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test the health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        if success:
            logger.info(f"Health check response: {response}")
            if response.get('status') == 'healthy' and response.get('model_loaded') is True:
                logger.info("✅ Health check confirms model is loaded")
            else:
                logger.warning("⚠️ Model may not be loaded properly")
        return success

    def test_model_info(self):
        """Test the model info endpoint"""
        success, response = self.run_test(
            "Model Info",
            "GET",
            "api/model-info",
            200
        )
        if success:
            logger.info(f"Model info: {response}")
            if 'model_type' in response and 'classes' in response:
                logger.info(f"✅ Model type: {response['model_type']}, Classes: {len(response['classes'])}")
            else:
                logger.warning("⚠️ Model info incomplete")
        return success

    def on_ws_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            self.ws_responses.append(data)
            detections = data.get('detections', [])
            logger.info(f"Received detection results: {len(detections)} items detected")
            for detection in detections[:3]:  # Show first 3 detections
                logger.info(f"  - {detection.get('class_name')} ({detection.get('confidence', 0)*100:.1f}%)")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def on_ws_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.ws_error = error
        self.ws_connected = False

    def on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connected = False

    def on_ws_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connected successfully")
        self.ws_connected = True
        self.tests_passed += 1

    def test_websocket_connection(self):
        """Test WebSocket connection"""
        self.tests_run += 1
        logger.info("Testing WebSocket connection...")
        
        try:
            # Connect to WebSocket
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_ws_open,
                on_message=self.on_ws_message,
                on_error=self.on_ws_error,
                on_close=self.on_ws_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.ws_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if self.ws_connected:
                logger.info("✅ WebSocket connected successfully")
                return True
            else:
                logger.error(f"❌ WebSocket connection failed: {self.ws_error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {str(e)}")
            return False

    def test_frame_detection(self, test_image_path=None):
        """Test sending a frame for detection"""
        if not self.ws_connected:
            logger.error("❌ Cannot test frame detection: WebSocket not connected")
            return False
            
        self.tests_run += 1
        logger.info("Testing frame detection...")
        
        try:
            # Create a simple test image if none provided
            if test_image_path:
                with open(test_image_path, 'rb') as img_file:
                    img = Image.open(img_file)
            else:
                # Create a blank image
                img = Image.new('RGB', (640, 480), color='white')
                
            # Convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send frame for detection
            frame_info = {
                'frame': f"data:image/jpeg;base64,{img_str}",
                'timestamp': int(time.time() * 1000),
                'frame_count': 1
            }
            
            self.ws.send(json.dumps(frame_info))
            
            # Wait for response
            timeout = 10
            start_time = time.time()
            while not self.ws_responses and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if self.ws_responses:
                logger.info("✅ Received detection response")
                self.tests_passed += 1
                return True
            else:
                logger.error("❌ No detection response received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Frame detection test failed: {str(e)}")
            return False
        
    def close_websocket(self):
        """Close the WebSocket connection"""
        if self.ws:
            self.ws.close()
            logger.info("WebSocket connection closed")

def main():
    logger.info("Starting Live Trash Finder API Tests")
    
    # Get backend URL from environment if available
    backend_url = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
    logger.info(f"Using backend URL: {backend_url}")
    
    # Setup tester
    tester = TrashFinderAPITester(backend_url)
    
    # Run tests
    health_ok = tester.test_health_check()
    model_ok = tester.test_model_info()
    
    if health_ok and model_ok:
        ws_ok = tester.test_websocket_connection()
        if ws_ok:
            tester.test_frame_detection()
            # Close WebSocket connection
            tester.close_websocket()
    
    # Print results
    logger.info(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
