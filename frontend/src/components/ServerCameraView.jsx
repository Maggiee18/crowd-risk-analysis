import React from 'react';
import { Camera } from 'lucide-react';

// Live view of a camera the backend reads itself. The frames arrive over the
// WebSocket already analysed (boxes, count, risk and privacy blur drawn in).
export default function ServerCameraView({ camera, frameData, peopleCount, riskLevel, isConnected }) {
  const risk = (riskLevel || 'low').toUpperCase();
  const riskColor = { LOW: 'bg-emerald-500', MEDIUM: 'bg-amber-500', HIGH: 'bg-red-600' }[risk] || 'bg-emerald-500';
  return (
    <div className="glass-card !p-3 video-feed">
      <div className="flex items-center gap-2 mb-2">
        <Camera size={16} className="t-muted" />
        <h3 className="text-sm font-semibold t-heading">{camera ? camera.name : 'Camera'}</h3>
        <span className="text-[10px] t-muted">analysed on the server</span>
        <span className={`ml-auto text-[10px] font-bold text-white px-2 py-0.5 rounded-lg ${riskColor}`}>{risk}</span>
      </div>
      <div className="relative rounded-xl overflow-hidden bg-black" style={{ aspectRatio: '4 / 3' }}>
        {frameData ? (
          <img src={`data:image/jpeg;base64,${frameData}`} alt="Camera view" className="w-full h-full object-contain" />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-xs text-white/60 px-6 text-center">
            {!isConnected ? 'Connecting to the server…'
              : camera && camera.status === 'offline' ? `Camera offline: ${camera.error || 'no video'}`
                : 'Waiting for the first analysed frame…'}
          </div>
        )}
        <div className="absolute bottom-2 left-2 text-xs font-semibold text-white bg-black/60 px-2 py-1 rounded-lg">
          {peopleCount} people
        </div>
      </div>
    </div>
  );
}
