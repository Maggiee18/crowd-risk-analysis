import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Camera, CameraOff, Shield, ShieldOff, Video, Upload,
  FileVideo, X, Loader2, Play, Pause, SkipForward
} from 'lucide-react';

/**
 * VideoFeed — unified input component
 *
 * Three modes:
 *   1. Live camera (asks for permission)
 *   2. Uploaded video (extracts frames client-side → sends to backend)
 *   3. Simulation / backend stream (just displays base64 frames)
 */

export default function VideoFeed({
  frameData,
  isConnected,
  privacyEnabled,
  onTogglePrivacy,
  onFrameReady,          // (base64, source: 'camera'|'upload') => void
  isProcessing,
  inputSource,           // 'none' | 'camera' | 'upload' | 'simulation'
  onSourceChange,        // (source) => void
}) {
  // ── Camera state ──────────────────────────────────────────────
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const captureInterval = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [cameraPermission, setCameraPermission] = useState('prompt'); // prompt | granted | denied

  // ── Video upload state ────────────────────────────────────────
  const fileInputRef = useRef(null);
  const uploadVideoRef = useRef(null);
  const uploadCanvasRef = useRef(null);
  const uploadInterval = useRef(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadPlaying, setUploadPlaying] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadDuration, setUploadDuration] = useState(0);
  const [uploadFps, setUploadFps] = useState(3); // frames sent per second

  // ── Check camera permission on mount ──────────────────────────
  useEffect(() => {
    if (navigator.permissions?.query) {
      navigator.permissions.query({ name: 'camera' }).then(result => {
        setCameraPermission(result.state);
        result.onchange = () => setCameraPermission(result.state);
      }).catch(() => {});
    }
    // Enumerate devices
    navigator.mediaDevices?.enumerateDevices().then(devs => {
      const cams = devs.filter(d => d.kind === 'videoinput');
      setCameraDevices(cams);
      if (cams.length > 0 && !selectedDevice) setSelectedDevice(cams[0].deviceId);
    }).catch(() => {});
  }, []);

  // ── Camera logic ──────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 15 },
          ...(selectedDevice ? { deviceId: { exact: selectedDevice } } : {}),
        },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      setCameraPermission('granted');

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
        onSourceChange?.('camera');

        // Capture frames at ~3 FPS and send upstream
        captureInterval.current = setInterval(() => {
          if (!videoRef.current || !canvasRef.current) return;
          const v = videoRef.current;
          if (v.readyState !== v.HAVE_ENOUGH_DATA) return;

          const c = canvasRef.current;
          c.width = 640;
          c.height = 480;
          c.getContext('2d').drawImage(v, 0, 0, 640, 480);
          const b64 = c.toDataURL('image/jpeg', 0.7).split(',')[1];
          if (b64) onFrameReady?.(b64, 'camera');
        }, 333); // ~3 FPS
      }
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setCameraPermission('denied');
        setCameraError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError') {
        setCameraError('No camera found. Please connect a webcam.');
      } else {
        setCameraError(`Camera error: ${err.message}`);
      }
    }
  }, [selectedDevice, onFrameReady, onSourceChange]);

  const stopCamera = useCallback(() => {
    if (captureInterval.current) {
      clearInterval(captureInterval.current);
      captureInterval.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    onSourceChange?.('none');
  }, [onSourceChange]);

  // Cleanup on unmount
  useEffect(() => () => { stopCamera(); stopUploadPlayback(); }, []);

  // ── Video upload logic ────────────────────────────────────────
  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop camera if active
    if (cameraActive) stopCamera();
    setUploadFile(file);
    setUploadPlaying(false);
    setUploadProgress(0);
    onSourceChange?.('upload');
  }, [cameraActive, stopCamera, onSourceChange]);

  const startUploadPlayback = useCallback(() => {
    const vid = uploadVideoRef.current;
    if (!vid || !uploadFile) return;

    vid.src = URL.createObjectURL(uploadFile);
    vid.load();

    vid.onloadedmetadata = () => {
      setUploadDuration(vid.duration);
      vid.play();
      setUploadPlaying(true);

      uploadInterval.current = setInterval(() => {
        if (!uploadVideoRef.current || !uploadCanvasRef.current) return;
        const v = uploadVideoRef.current;
        if (v.paused || v.ended) {
          if (v.ended) {
            setUploadPlaying(false);
            clearInterval(uploadInterval.current);
          }
          return;
        }

        setUploadProgress((v.currentTime / v.duration) * 100);

        const c = uploadCanvasRef.current;
        c.width = 640;
        c.height = 480;
        c.getContext('2d').drawImage(v, 0, 0, 640, 480);
        const b64 = c.toDataURL('image/jpeg', 0.7).split(',')[1];
        if (b64) onFrameReady?.(b64, 'upload');
      }, 1000 / uploadFps);
    };
  }, [uploadFile, uploadFps, onFrameReady]);

  const stopUploadPlayback = useCallback(() => {
    if (uploadInterval.current) {
      clearInterval(uploadInterval.current);
      uploadInterval.current = null;
    }
    if (uploadVideoRef.current) {
      uploadVideoRef.current.pause();
    }
    setUploadPlaying(false);
  }, []);

  const toggleUploadPlayback = useCallback(() => {
    if (uploadPlaying) {
      stopUploadPlayback();
    } else {
      if (uploadVideoRef.current && uploadVideoRef.current.src) {
        uploadVideoRef.current.play();
        setUploadPlaying(true);
        uploadInterval.current = setInterval(() => {
          const v = uploadVideoRef.current;
          if (!v) return;
          if (v.paused || v.ended) {
            if (v.ended) {
              setUploadPlaying(false);
              clearInterval(uploadInterval.current);
            }
            return;
          }
          setUploadProgress((v.currentTime / v.duration) * 100);
          const c = uploadCanvasRef.current;
          if (!c) return;
          c.width = 640;
          c.height = 480;
          c.getContext('2d').drawImage(v, 0, 0, 640, 480);
          const b64 = c.toDataURL('image/jpeg', 0.7).split(',')[1];
          if (b64) onFrameReady?.(b64, 'upload');
        }, 1000 / uploadFps);
      } else {
        startUploadPlayback();
      }
    }
  }, [uploadPlaying, startUploadPlayback, stopUploadPlayback, uploadFps, onFrameReady]);

  const clearUpload = useCallback(() => {
    stopUploadPlayback();
    setUploadFile(null);
    setUploadProgress(0);
    setUploadDuration(0);
    if (uploadVideoRef.current) {
      uploadVideoRef.current.src = '';
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
    onSourceChange?.('none');
  }, [stopUploadPlayback, onSourceChange]);

  // ── Source badge text ──────────────────────────────────────────
  const sourceBadge = inputSource === 'camera'
    ? { label: '📷 LIVE CAMERA', color: 'text-emerald-400' }
    : inputSource === 'upload'
    ? { label: '🎬 VIDEO FILE', color: 'text-cyan-400' }
    : inputSource === 'simulation'
    ? { label: '🤖 SIMULATION', color: 'text-purple-400' }
    : null;

  return (
    <div className="glass-card" id="video-feed-panel">
      {/* ── Header ───────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Camera size={18} className="text-brand-400" />
          <h3 className="text-sm font-semibold text-white">Video Input</h3>
          <span className={`pulse-dot ${isConnected ? 'online' : 'offline'}`} />
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Camera button */}
          {!cameraActive ? (
            <button
              onClick={startCamera}
              className="btn-ghost text-xs px-3 py-1.5 gap-1"
              title="Start webcam"
            >
              <Camera size={14} /> Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="btn-ghost text-xs px-3 py-1.5 gap-1 !text-red-400"
              title="Stop webcam"
            >
              <CameraOff size={14} /> Stop
            </button>
          )}

          {/* Upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn-ghost text-xs px-3 py-1.5 gap-1"
            title="Upload video file"
          >
            <Upload size={14} /> Upload
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Privacy toggle */}
          <button
            onClick={onTogglePrivacy}
            className={`btn-ghost text-xs px-3 py-1.5 gap-1 ${privacyEnabled ? 'text-emerald-400' : 'text-slate-500'}`}
            title={privacyEnabled ? 'Privacy ON' : 'Privacy OFF'}
          >
            {privacyEnabled ? <Shield size={14} /> : <ShieldOff size={14} />}
            {privacyEnabled ? 'Privacy' : 'Off'}
          </button>
        </div>
      </div>

      {/* ── Camera permission banner ─────────────────────────── */}
      {cameraPermission === 'denied' && !cameraActive && (
        <div className="mb-3 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300 flex items-start gap-2">
          <CameraOff size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Camera access blocked</p>
            <p className="text-red-400/80 mt-0.5">
              Please enable camera permissions in your browser settings, then reload the page.
            </p>
          </div>
        </div>
      )}

      {/* ── Camera error ─────────────────────────────────────── */}
      {cameraError && (
        <div className="mb-3 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300">
          {cameraError}
        </div>
      )}

      {/* ── Camera device selector ───────────────────────────── */}
      {cameraDevices.length > 1 && !cameraActive && (
        <select
          value={selectedDevice}
          onChange={e => setSelectedDevice(e.target.value)}
          className="input-field text-xs mb-3"
        >
          {cameraDevices.map((d, i) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Camera ${i + 1}`}
            </option>
          ))}
        </select>
      )}

      {/* ── Upload controls bar ──────────────────────────────── */}
      {uploadFile && (
        <div className="mb-3 p-3 rounded-xl bg-surface-800/60 border border-white/[0.06] space-y-2">
          <div className="flex items-center gap-2">
            <FileVideo size={14} className="text-cyan-400 shrink-0" />
            <span className="text-xs text-white truncate flex-1">{uploadFile.name}</span>
            <button onClick={clearUpload} className="text-slate-500 hover:text-red-400 transition-colors p-0.5">
              <X size={14} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleUploadPlayback}
              className="btn-primary !py-1.5 !px-3 text-xs"
            >
              {uploadPlaying ? <Pause size={12} /> : <Play size={12} />}
              {uploadPlaying ? 'Pause' : 'Analyze'}
            </button>
            <div className="flex-1 relative h-1.5 bg-surface-700 rounded-full overflow-hidden">
              <div
                className="absolute left-0 top-0 h-full bg-gradient-to-r from-cyan-500 to-brand-500 transition-all duration-300 rounded-full"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 tabular-nums w-10 text-right">
              {uploadProgress.toFixed(0)}%
            </span>
          </div>

          {/* FPS control */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500">Analysis speed:</span>
            {[1, 3, 5, 10].map(f => (
              <button
                key={f}
                onClick={() => setUploadFps(f)}
                className={`text-[10px] px-2 py-0.5 rounded-md transition-all ${
                  uploadFps === f
                    ? 'bg-brand-500/20 text-brand-400 border border-brand-500/30'
                    : 'bg-surface-800/40 text-slate-500 border border-white/[0.04] hover:text-white'
                }`}
              >
                {f} fps
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Video display area ───────────────────────────────── */}
      <div className="video-feed relative">
        {frameData ? (
          <img
            src={`data:image/jpeg;base64,${frameData}`}
            alt="AI analysis feed"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
            <div className="w-20 h-20 rounded-2xl bg-surface-800/60 flex items-center justify-center mb-4">
              <Camera size={32} className="opacity-30" />
            </div>
            <p className="text-sm font-medium">No video source active</p>
            <p className="text-xs mt-1 text-slate-600 text-center px-8">
              Click <strong>Camera</strong> to use your webcam, or <strong>Upload</strong> to analyze a video file
            </p>
          </div>
        )}

        {/* Source badge */}
        {sourceBadge && frameData && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-sm text-[11px] font-medium">
            <span className="pulse-dot online" />
            <span className={sourceBadge.color}>{sourceBadge.label}</span>
          </div>
        )}

        {/* Processing indicator */}
        {isProcessing && frameData && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-sm text-[10px] text-amber-400">
            <Loader2 size={10} className="animate-spin" /> AI Processing…
          </div>
        )}
      </div>

      {/* Camera active status */}
      {cameraActive && (
        <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400">
          <span className="pulse-dot online" />
          Streaming at 3 FPS — all panels updating live
          {isProcessing && <span className="text-amber-400 ml-2">• Processing…</span>}
        </div>
      )}

      {/* Hidden elements */}
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <video ref={uploadVideoRef} style={{ display: 'none' }} playsInline muted />
      <canvas ref={uploadCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}
