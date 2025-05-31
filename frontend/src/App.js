import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [detections, setDetections] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  // FPS calculation
  const fpsIntervalRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());

  // Get camera stream
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setError(null);
      }
    } catch (err) {
      setError('Failed to access camera: ' + err.message);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // Setup WebSocket connection
  const connectWebSocket = useCallback(() => {
    const wsUrl = BACKEND_URL.replace('http', 'ws') + '/api/ws/detect';
    
    try {
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.detections) {
            setDetections(data.detections);
            setFrameCount(data.frame_count || 0);
          }
          if (data.error) {
            console.error('Detection error:', data.error);
          }
        } catch (err) {
          console.error('Failed to parse detection data:', err);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        console.log('WebSocket disconnected');
      };
      
    } catch (err) {
      setError('Failed to connect to detection service: ' + err.message);
      setConnectionStatus('error');
    }
  }, []);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Capture and send frame
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send frame to backend for detection
    if (wsRef.current.readyState === WebSocket.OPEN) {
      const frameInfo = {
        frame: frameData,
        timestamp: Date.now(),
        frame_count: frameCount + 1
      };
      
      wsRef.current.send(JSON.stringify(frameInfo));
      setFrameCount(prev => prev + 1);
      
      // Calculate FPS
      const now = Date.now();
      const deltaTime = now - lastFrameTimeRef.current;
      if (deltaTime > 0) {
        setFps(Math.round(1000 / deltaTime));
      }
      lastFrameTimeRef.current = now;
    }
  }, [frameCount]);

  // Start streaming
  const startStreaming = useCallback(async () => {
    await startCamera();
    connectWebSocket();
    setIsStreaming(true);
    
    // Start frame capture loop
    const interval = setInterval(captureFrame, 100); // 10 FPS for processing
    fpsIntervalRef.current = interval;
  }, [startCamera, connectWebSocket, captureFrame]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    stopCamera();
    disconnectWebSocket();
    setIsStreaming(false);
    setDetections([]);
    setFrameCount(0);
    setFps(0);
    
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }
  }, [stopCamera, disconnectWebSocket]);

  // Get model info
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/model-info`);
        if (response.ok) {
          const info = await response.json();
          setModelInfo(info);
        }
      } catch (err) {
        console.error('Failed to fetch model info:', err);
      }
    };
    
    fetchModelInfo();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  // Draw detection overlays
  useEffect(() => {
    if (!videoRef.current || detections.length === 0) return;
    
    const video = videoRef.current;
    const canvas = document.getElementById('overlay-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scale factors
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    // Draw detections
    detections.forEach((detection, index) => {
      const [x1, y1, x2, y2] = detection.bbox;
      
      // Scale coordinates to canvas size
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;
      
      const width = scaledX2 - scaledX1;
      const height = scaledY2 - scaledY1;
      
      // Draw bounding box
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX1, scaledY1, width, height);
      
      // Draw label background
      const label = `${detection.class_name} (${(detection.confidence * 100).toFixed(1)}%)`;
      ctx.font = '14px Arial';
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width + 10;
      const textHeight = 20;
      
      ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.fillRect(scaledX1, scaledY1 - textHeight, textWidth, textHeight);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(label, scaledX1 + 5, scaledY1 - 5);
    });
  }, [detections]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-600 to-orange-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">🗑️ Live Trash Finder</h1>
              <p className="text-red-100 mt-1">Real-time trash detection using AI</p>
            </div>
            <div className="text-right text-sm">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connectionStatus === 'connected' ? 'bg-green-400' : connectionStatus === 'error' ? 'bg-red-400' : 'bg-yellow-400'}`}></div>
                <span className="capitalize">{connectionStatus}</span>
              </div>
              {modelInfo && (
                <div className="text-red-100 mt-1">
                  Model: {modelInfo.model_type} | Device: {modelInfo.device}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Video Stream Section */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg overflow-hidden shadow-xl">
              <div className="p-4 bg-gray-700 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Live Camera Feed</h2>
                <div className="flex items-center space-x-4">
                  <div className="text-sm">
                    FPS: <span className="font-mono text-green-400">{fps}</span>
                  </div>
                  <div className="text-sm">
                    Frames: <span className="font-mono text-blue-400">{frameCount}</span>
                  </div>
                  <button
                    onClick={isStreaming ? stopStreaming : startStreaming}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      isStreaming 
                        ? 'bg-red-600 hover:bg-red-700 text-white' 
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {isStreaming ? 'Stop Detection' : 'Start Detection'}
                  </button>
                </div>
              </div>
              
              <div className="relative">
                <video
                  ref={videoRef}
                  className="w-full h-auto bg-black"
                  playsInline
                  muted
                />
                <canvas
                  id="overlay-canvas"
                  className="absolute top-0 left-0 pointer-events-none"
                  style={{ width: '100%', height: '100%' }}
                />
                <canvas
                  ref={canvasRef}
                  className="hidden"
                />
              </div>
            </div>
          </div>

          {/* Detection Results Section */}
          <div className="space-y-6">
            {/* Current Detections */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <span className="w-3 h-3 bg-red-500 rounded-full mr-2"></span>
                Live Detections ({detections.length})
              </h3>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {detections.length === 0 ? (
                  <div className="text-gray-400 text-center py-4">
                    {isStreaming ? 'No trash detected' : 'Start detection to see results'}
                  </div>
                ) : (
                  detections.map((detection, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">
                          {detection.class_name}
                        </span>
                        <span className="text-xs bg-red-600 px-2 py-1 rounded">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xs text-gray-400">
                        Position: ({Math.round(detection.bbox[0])}, {Math.round(detection.bbox[1])}) - 
                        ({Math.round(detection.bbox[2])}, {Math.round(detection.bbox[3])})
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h3 className="text-lg font-semibold mb-4">Session Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">{frameCount}</div>
                  <div className="text-sm text-gray-400">Frames Processed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">{detections.length}</div>
                  <div className="text-sm text-gray-400">Items Detected</div>
                </div>
              </div>
            </div>

            {/* Instructions */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h3 className="text-lg font-semibold mb-4">How to Use</h3>
              <ul className="text-sm text-gray-300 space-y-2">
                <li>• Click "Start Detection" to begin</li>
                <li>• Point camera at various objects</li>
                <li>• Red boxes show detected trash</li>
                <li>• Confidence scores indicate certainty</li>
                <li>• Works with bottles, cans, cigarettes, etc.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-6 bg-red-600 border border-red-700 rounded-lg p-4">
            <h4 className="font-semibold mb-2">Error</h4>
            <p className="text-sm">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
