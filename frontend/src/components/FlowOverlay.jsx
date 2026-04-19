import React, { useRef, useEffect, useMemo } from 'react';
import { ArrowUpRight, Wind } from 'lucide-react';

/**
 * Feature #3: Crowd Flow Direction Arrows
 * Renders directional arrows showing dominant crowd movement using canvas.
 */
export default function FlowOverlay({ features = {}, detections = [], frameWidth = 640, frameHeight = 480 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  const flowAngle = features.dominant_direction || 0; // degrees
  const flowMag = features.avg_magnitude || 0;
  const uniformity = features.direction_uniformity || 0;

  // Build flow vectors from detections
  const flowVectors = useMemo(() => {
    if (!detections || detections.length === 0) return [];

    return detections.slice(0, 30).map(det => {
      const [cx, cy] = det.center || [0, 0];
      const angle = (det.direction || flowAngle) * (Math.PI / 180);
      const speed = det.speed || flowMag;
      return { cx, cy, angle, speed };
    });
  }, [detections, flowAngle, flowMag]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const render = () => {
      const cw = canvas.width;
      const ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);

      const scaleX = cw / frameWidth;
      const scaleY = ch / frameHeight;

      const now = Date.now();

      if (flowVectors.length > 0) {
        // Draw individual flow arrows
        flowVectors.forEach((v, i) => {
          const x = v.cx * scaleX;
          const y = v.cy * scaleY;
          const len = Math.min(30, Math.max(8, v.speed * 3));
          const pulse = 0.6 + 0.4 * Math.sin(now / 500 + i);

          const ex = x + Math.cos(v.angle) * len;
          const ey = y + Math.sin(v.angle) * len;

          // Arrow line
          ctx.strokeStyle = `rgba(59, 130, 246, ${pulse * 0.6})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(ex, ey);
          ctx.stroke();

          // Arrowhead
          const headLen = 6;
          const aAngle = Math.atan2(ey - y, ex - x);
          ctx.fillStyle = `rgba(59, 130, 246, ${pulse * 0.8})`;
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(
            ex - headLen * Math.cos(aAngle - Math.PI / 6),
            ey - headLen * Math.sin(aAngle - Math.PI / 6)
          );
          ctx.lineTo(
            ex - headLen * Math.cos(aAngle + Math.PI / 6),
            ey - headLen * Math.sin(aAngle + Math.PI / 6)
          );
          ctx.closePath();
          ctx.fill();
        });
      }

      // Draw dominant flow direction (large central arrow)
      if (flowMag > 0.5) {
        const cx = cw / 2;
        const cy = ch / 2;
        const angle = flowAngle * (Math.PI / 180);
        const len = Math.min(80, flowMag * 8);
        const pulse = 0.5 + 0.5 * Math.sin(now / 800);

        const sx = cx - Math.cos(angle) * len * 0.5;
        const sy = cy - Math.sin(angle) * len * 0.5;
        const ex = cx + Math.cos(angle) * len * 0.5;
        const ey = cy + Math.sin(angle) * len * 0.5;

        // Glow
        ctx.shadowColor = 'rgba(59, 130, 246, 0.5)';
        ctx.shadowBlur = 12;

        ctx.strokeStyle = `rgba(255, 255, 255, ${pulse * 0.4})`;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.setLineDash([]);

        // Big arrowhead
        const headLen = 12;
        const aAngle = Math.atan2(ey - sy, ex - sx);
        ctx.fillStyle = `rgba(255, 255, 255, ${pulse * 0.6})`;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(
          ex - headLen * Math.cos(aAngle - Math.PI / 5),
          ey - headLen * Math.sin(aAngle - Math.PI / 5)
        );
        ctx.lineTo(
          ex - headLen * Math.cos(aAngle + Math.PI / 5),
          ey - headLen * Math.sin(aAngle + Math.PI / 5)
        );
        ctx.closePath();
        ctx.fill();

        ctx.shadowBlur = 0;
      }

      animRef.current = requestAnimationFrame(render);
    };

    render();
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [flowVectors, flowAngle, flowMag, frameWidth, frameHeight]);

  return (
    <div className="glass-card" id="flow-overlay-panel">
      <div className="flex items-center gap-2 mb-3">
        <Wind size={16} className="text-blue-400" />
        <h3 className="text-sm font-semibold t-heading">Crowd Flow</h3>
        <div className="ml-auto flex items-center gap-2 text-[10px] t-muted">
          <span>Direction: {flowAngle.toFixed(0)}°</span>
          <span>•</span>
          <span>Speed: {flowMag.toFixed(1)}</span>
          <span>•</span>
          <span>Uniformity: {(uniformity * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="relative rounded-xl overflow-hidden" style={{ background: 'var(--color-surface)', border: '1px solid var(--card-border)' }}>
        <canvas
          ref={canvasRef}
          width={400}
          height={260}
          className="w-full h-auto"
        />
        {/* Legend overlay */}
        <div className="absolute bottom-2 left-2 flex items-center gap-2 px-2 py-1 rounded-lg bg-black/50 backdrop-blur-sm text-[9px]">
          <ArrowUpRight size={10} className="text-blue-400" />
          <span className="text-blue-300">Individual flow</span>
          <span className="text-white/40">|</span>
          <span className="text-white/80">Dominant direction</span>
        </div>
      </div>
    </div>
  );
}
