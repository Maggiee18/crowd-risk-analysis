import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video } from 'lucide-react';

export default function WebcamCapture({ onFrame, isProcessing, fps = 5 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');

  // List available cameras
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(devs => {
      const cams = devs.filter(d => d.kind === 'videoinput');
      setDevices(cams);
      if (cams.length > 0 && !selectedDevice) {
        setSelectedDevice(cams[0].deviceId);
      }
    }).catch(() => {});
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 15 },
          ...(selectedDevice ? { deviceId: { exact: selectedDevice } } : {})
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);

        // Start capturing frames at the specified FPS
        intervalRef.current = setInterval(() => {
          captureFrame();
        }, 1000 / fps);
      }
    } catch (err) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a webcam.');
      } else {
        setError(`Camera error: ${err.message}`);
      }
    }
  }, [selectedDevice, fps]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);

    // Convert to base64 JPEG
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    const base64 = dataUrl.split(',')[1];

    if (onFrame && base64) {
      onFrame(base64);
    }
  }, [onFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return (
    <div className="space-y-3">
      {/* Camera selector */}
      {devices.length > 1 && (
        <select
          value={selectedDevice}
          onChange={(e) => setSelectedDevice(e.target.value)}
          disabled={isStreaming}
          className="input-field text-xs"
        >
          {devices.map((d, i) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Camera ${i + 1}`}
            </option>
          ))}
        </select>
      )}

      {/* Camera controls */}
      <div className="flex gap-2">
        {!isStreaming ? (
          <button onClick={startCamera} className="btn-primary flex-1">
            <Camera size={16} /> Start Webcam
          </button>
        ) : (
          <button onClick={stopCamera} className="btn-primary flex-1 !bg-red-600 hover:!bg-red-500 !shadow-red-600/25">
            <CameraOff size={16} /> Stop Webcam
          </button>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* Status */}
      {isStreaming && (
        <div className="flex items-center gap-2 text-xs text-emerald-400">
          <span className="pulse-dot online" />
          Streaming at {fps} FPS
          {isProcessing && <span className="text-amber-400 ml-2">• Processing…</span>}
        </div>
      )}

      {/* Hidden video and canvas elements */}
      <video
        ref={videoRef}
        style={{ display: 'none' }}
        playsInline
        muted
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}
