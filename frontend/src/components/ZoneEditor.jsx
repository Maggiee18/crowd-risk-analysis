import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Map, Plus, Trash2, Save, Eye, EyeOff } from 'lucide-react';

/**
 * Feature #6: Geofence / Virtual Zone Editor
 * Interactive canvas for drawing zones over the video feed.
 */
export default function ZoneEditor({ frameData, frameWidth = 640, frameHeight = 480, onZonesUpdate }) {
  const canvasRef = useRef(null);
  const [zones, setZones] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [currentPoints, setCurrentPoints] = useState([]);
  const [selectedZone, setSelectedZone] = useState(null);
  const [showZones, setShowZones] = useState(true);
  const [zoneType, setZoneType] = useState('monitor'); // monitor | restricted | entry | exit

  const zoneColors = {
    monitor: { fill: 'rgba(99, 102, 241, 0.15)', stroke: '#6366f1', label: 'Monitor' },
    restricted: { fill: 'rgba(239, 68, 68, 0.15)', stroke: '#ef4444', label: 'Restricted' },
    entry: { fill: 'rgba(16, 185, 129, 0.15)', stroke: '#10b981', label: 'Entry' },
    exit: { fill: 'rgba(245, 158, 11, 0.15)', stroke: '#f59e0b', label: 'Exit' },
  };

  // Draw zones on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;

    const drawElements = () => {
      if (!showZones) return;

      const scaleX = cw / frameWidth;
      const scaleY = ch / frameHeight;

      // Draw completed zones
      zones.forEach((zone, i) => {
        const colors = zoneColors[zone.type] || zoneColors.monitor;

        ctx.fillStyle = colors.fill;
        ctx.strokeStyle = selectedZone === i ? '#ffffff' : colors.stroke;
        ctx.lineWidth = selectedZone === i ? 2.5 : 1.5;

        ctx.beginPath();
        zone.points.forEach(([x, y], pi) => {
          const px = x * scaleX;
          const py = y * scaleY;
          if (pi === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Label
        const cx = zone.points.reduce((s, p) => s + p[0], 0) / (zone.points.length || 1) * scaleX;
        const cy = zone.points.reduce((s, p) => s + p[1], 0) / (zone.points.length || 1) * scaleY;
        ctx.fillStyle = colors.stroke;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${zone.name} (${colors.label})`, cx, cy);
      });

      // Draw current drawing points
      if (currentPoints.length > 0) {
        const colors = zoneColors[zoneType];
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);

        ctx.beginPath();
        currentPoints.forEach(([x, y], i) => {
          const px = x * scaleX;
          const py = y * scaleY;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw points
        currentPoints.forEach(([x, y]) => {
          const px = x * scaleX;
          const py = y * scaleY;
          ctx.fillStyle = colors.stroke;
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    };

    if (frameData) {
      const img = new window.Image();
      const imgSrc = frameData.startsWith('data:') ? frameData : `data:image/jpeg;base64,${frameData}`;
      img.onload = () => {
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(img, 0, 0, cw, ch);
        // Draw tint over image
        ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
        ctx.fillRect(0, 0, cw, ch);
        drawElements();
      };
      img.src = imgSrc;
    } else {
      ctx.clearRect(0, 0, cw, ch);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(0, 0, cw, ch);
      drawElements();
    }
  }, [zones, currentPoints, showZones, selectedZone, zoneType, frameWidth, frameHeight, frameData]);

  // Handle canvas click
  const handleCanvasClick = useCallback((e) => {
    if (!drawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = frameWidth / canvas.width;
    const scaleY = frameHeight / canvas.height;

    const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width) * scaleX);
    const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height) * scaleY);

    setCurrentPoints(prev => [...prev, [x, y]]);
  }, [drawing, frameWidth, frameHeight]);

  // Complete zone
  const completeZone = useCallback(() => {
    if (currentPoints.length < 3) return;

    const newZone = {
      id: Date.now(),
      name: `Zone ${zones.length + 1}`,
      type: zoneType,
      points: currentPoints,
    };

    const updated = [...zones, newZone];
    setZones(updated);
    setCurrentPoints([]);
    setDrawing(false);
    onZonesUpdate?.(updated);
  }, [currentPoints, zones, zoneType, onZonesUpdate]);

  const deleteZone = useCallback((i) => {
    const updated = zones.filter((_, idx) => idx !== i);
    setZones(updated);
    setSelectedZone(null);
    onZonesUpdate?.(updated);
  }, [zones, onZonesUpdate]);

  return (
    <div className="glass-card" id="zone-editor">
      <div className="flex items-center gap-2 mb-4">
        <Map size={16} className="text-indigo-400" />
        <h3 className="text-sm font-semibold t-heading">Zone Editor</h3>
        <div className="ml-auto flex items-center gap-1.5">
          <button
            onClick={() => setShowZones(!showZones)}
            className="p-1.5 rounded-lg transition-colors"
            style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}
            title={showZones ? 'Hide zones' : 'Show zones'}
          >
            {showZones ? <Eye size={12} className="t-muted" /> : <EyeOff size={12} className="t-muted" />}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        {/* Zone type selector */}
        {Object.entries(zoneColors).map(([key, val]) => (
          <button
            key={key}
            onClick={() => setZoneType(key)}
            className={`text-[10px] px-2 py-1 rounded-lg border transition-all ${
              zoneType === key
                ? `border-[${val.stroke}]/30 bg-[${val.stroke}]/10`
                : ''
            }`}
            style={{
              borderColor: zoneType === key ? val.stroke + '40' : 'var(--panel-border)',
              background: zoneType === key ? val.stroke + '15' : 'var(--panel-bg)',
              color: zoneType === key ? val.stroke : 'var(--color-text-muted)',
            }}
          >
            {val.label}
          </button>
        ))}

        <div className="flex-1" />

        {!drawing ? (
          <button
            onClick={() => { setDrawing(true); setCurrentPoints([]); }}
            className="btn-primary !py-1.5 !px-3 text-[10px]"
          >
            <Plus size={10} /> Draw Zone
          </button>
        ) : (
          <div className="flex gap-1">
            <button
              onClick={completeZone}
              disabled={currentPoints.length < 3}
              className="btn-primary !py-1.5 !px-3 text-[10px]"
            >
              <Save size={10} /> Complete ({currentPoints.length} pts)
            </button>
            <button
              onClick={() => { setDrawing(false); setCurrentPoints([]); }}
              className="btn-ghost !py-1.5 !px-2 text-[10px]"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* Canvas */}
      <div
        className="rounded-xl overflow-hidden cursor-crosshair"
        style={{ border: drawing ? '2px dashed var(--color-brand)' : '1px solid var(--card-border)' }}
      >
        <canvas
          ref={canvasRef}
          width={480}
          height={320}
          onClick={handleCanvasClick}
          className="w-full h-auto"
          style={{ background: 'var(--color-surface)' }}
        />
      </div>

      {drawing && (
        <p className="text-[10px] text-brand-400 mt-2 animate-pulse">
          Click to add points • Minimum 3 points to complete a zone
        </p>
      )}

      {/* Zone list */}
      {zones.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {zones.map((zone, i) => (
            <div
              key={zone.id}
              onClick={() => setSelectedZone(selectedZone === i ? null : i)}
              className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-all text-xs ${
                selectedZone === i ? 'ring-1 ring-brand-400' : ''
              }`}
              style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}
            >
              <div
                className="w-3 h-3 rounded-full shrink-0"
                style={{ background: zoneColors[zone.type]?.stroke }}
              />
              <span className="font-medium t-heading flex-1">{zone.name}</span>
              <span className="text-[10px] t-muted">{zone.points.length} pts</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteZone(i); }}
                className="p-1 rounded hover:bg-red-500/20 hover:text-red-400 transition-colors t-muted"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
