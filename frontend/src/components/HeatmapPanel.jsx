import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Flame, Layers, RotateCcw, Maximize2 } from 'lucide-react';

/**
 * Interactive Crowd Density Heatmap
 *
 * Renders a canvas-based density heatmap from the 2D grid array sent by the
 * backend (`heatmap` field in frame data — a 2D array of uint8 values 0-255).
 * Also supports an accumulated mode that averages historical snapshots.
 */

// ── Color palette (JET-inspired) ─────────────────────────────────────
function jetColor(value) {
  // value: 0 → 1
  const t = Math.max(0, Math.min(1, value));
  let r, g, b;

  if (t < 0.125) {
    r = 0; g = 0; b = 0.5 + t * 4;
  } else if (t < 0.375) {
    r = 0; g = (t - 0.125) * 4; b = 1;
  } else if (t < 0.625) {
    r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4;
  } else if (t < 0.875) {
    r = 1; g = 1 - (t - 0.625) * 4; b = 0;
  } else {
    r = 1 - (t - 0.875) * 2; g = 0; b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ── Inferno-inspired palette (more premium) ─────────────────────────
function infernoColor(value) {
  const t = Math.max(0, Math.min(1, value));
  // Simplified inferno stops
  const stops = [
    [0, 0, 4],       // 0.0  — near-black
    [40, 11, 84],     // 0.2
    [101, 21, 110],   // 0.35
    [159, 42, 99],    // 0.5
    [212, 72, 66],    // 0.65
    [245, 125, 21],   // 0.8
    [252, 210, 29],   // 0.95
    [252, 255, 164],  // 1.0  — bright yellow
  ];

  const idx = t * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, stops.length - 1);
  const frac = idx - lo;

  return [
    Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac),
    Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac),
    Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac),
  ];
}

const PALETTES = {
  inferno: infernoColor,
  jet: jetColor,
};

// ── Gaussian blur helper (simple 3×3 kernel) ─────────────────────────
function blur2D(grid, rows, cols) {
  const out = new Float32Array(rows * cols);
  const k = [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      let ki = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rr = Math.max(0, Math.min(rows - 1, r + dr));
          const cc = Math.max(0, Math.min(cols - 1, c + dc));
          sum += grid[rr * cols + cc] * k[ki];
          ki++;
        }
      }
      out[r * cols + c] = sum;
    }
  }
  return out;
}

export default function HeatmapPanel({
  heatmapData = null,       // 2D array from backend
  detections = [],          // Optional: raw detections for point-based mode
  frameWidth = 640,
  frameHeight = 480,
}) {
  const canvasRef = useRef(null);
  const [palette, setPalette] = useState('inferno');
  const [showLabels, setShowLabels] = useState(true);
  const [opacity, setOpacity] = useState(0.85);
  const [history, setHistory] = useState([]);
  const [mode, setMode] = useState('live'); // 'live' | 'accumulated'
  const animFrameRef = useRef(null);

  // Accumulate heatmap history
  useEffect(() => {
    if (heatmapData && Array.isArray(heatmapData) && heatmapData.length > 0) {
      setHistory(prev => [...prev.slice(-59), heatmapData]);
    }
  }, [heatmapData]);

  // Compute the grid to render
  const grid = useMemo(() => {
    if (mode === 'accumulated' && history.length > 0) {
      // Average all history snapshots
      const rows = history[0].length;
      const cols = history[0][0]?.length || 0;
      if (!cols) return null;

      const avg = Array.from({ length: rows }, () => new Array(cols).fill(0));
      for (const snap of history) {
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < (snap[r]?.length || 0); c++) {
            avg[r][c] += (snap[r][c] || 0);
          }
        }
      }
      const n = history.length;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          avg[r][c] /= n;
        }
      }
      return avg;
    }
    return heatmapData;
  }, [heatmapData, history, mode]);

  // Synthetic heatmap from detections (fallback if no grid data)
  const syntheticGrid = useMemo(() => {
    if (grid) return null; // Prefer backend grid
    if (!detections || detections.length === 0) return null;

    const gridSize = 20;
    const rows = Math.ceil(frameHeight / gridSize);
    const cols = Math.ceil(frameWidth / gridSize);
    const flat = new Float32Array(rows * cols);

    for (const det of detections) {
      const [cx, cy] = det.center || [0, 0];
      const gc = Math.min(Math.floor(cx / gridSize), cols - 1);
      const gr = Math.min(Math.floor(cy / gridSize), rows - 1);
      const area = det.area || 1000;
      const weight = Math.log(area + 1) / 10;
      // Add contribution with spread
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rr = gr + dr;
          const cc = gc + dc;
          if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
            const dist = Math.sqrt(dr * dr + dc * dc);
            flat[rr * cols + cc] += weight * Math.exp(-dist);
          }
        }
      }
    }

    // Blur
    const blurred = blur2D(flat, rows, cols);

    // Convert to 2D
    const result = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        row.push(blurred[r * cols + c]);
      }
      result.push(row);
    }
    return result;
  }, [grid, detections, frameWidth, frameHeight]);

  const activeGrid = grid || syntheticGrid;

  // ── Canvas rendering ─────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const render = () => {
      const cw = canvas.width;
      const ch = canvas.height;

      // Clear
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, cw, ch);

      if (!activeGrid || activeGrid.length === 0) {
        // Empty state
        ctx.fillStyle = '#334155';
        ctx.font = '13px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for heatmap data…', cw / 2, ch / 2);
        return;
      }

      const rows = activeGrid.length;
      const cols = activeGrid[0]?.length || 0;
      if (!cols) return;

      // Find max value
      let maxVal = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = activeGrid[r]?.[c] || 0;
          if (v > maxVal) maxVal = v;
        }
      }
      if (maxVal === 0) maxVal = 1;

      const cellW = cw / cols;
      const cellH = ch / rows;
      const colorFn = PALETTES[palette] || infernoColor;

      // Draw cells with interpolation
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const rawVal = activeGrid[r]?.[c] || 0;
          const norm = rawVal / maxVal;

          const [cr, cg, cb] = colorFn(norm);
          const alpha = norm * opacity;

          ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${Math.max(0.08, alpha)})`;
          ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);

          // Grid lines (very subtle)
          ctx.strokeStyle = 'rgba(255,255,255,0.03)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);

          // Value labels
          if (showLabels && rawVal > maxVal * 0.15 && cellW > 18 && cellH > 14) {
            ctx.fillStyle = norm > 0.6 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)';
            ctx.font = `${Math.min(10, cellW * 0.35)}px JetBrains Mono, monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(
              rawVal > 100 ? Math.round(rawVal).toString() : rawVal.toFixed(0),
              c * cellW + cellW / 2,
              r * cellH + cellH / 2
            );
          }
        }
      }

      // Hotspot markers (top 3 cells)
      const cells = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          cells.push({ r, c, val: activeGrid[r]?.[c] || 0 });
        }
      }
      cells.sort((a, b) => b.val - a.val);
      const topCells = cells.filter(x => x.val > maxVal * 0.5).slice(0, 3);

      const now = Date.now();
      topCells.forEach((cell, i) => {
        const cx = cell.c * cellW + cellW / 2;
        const cy = cell.r * cellH + cellH / 2;
        const pulseRadius = 4 + Math.sin(now / 400 + i) * 2;

        // Pulsing glow
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulseRadius * 3);
        grad.addColorStop(0, 'rgba(255, 100, 50, 0.6)');
        grad.addColorStop(1, 'rgba(255, 100, 50, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, pulseRadius * 3, 0, Math.PI * 2);
        ctx.fill();

        // Center dot
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();
      });

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [activeGrid, palette, showLabels, opacity]);

  // Stats
  const stats = useMemo(() => {
    if (!activeGrid || !activeGrid.length) return null;
    const rows = activeGrid.length;
    const cols = activeGrid[0]?.length || 0;
    let total = 0, max = 0, hotspots = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = activeGrid[r]?.[c] || 0;
        total += v;
        if (v > max) max = v;
        if (v > 50) hotspots++;
      }
    }
    return {
      cells: rows * cols,
      total: total.toFixed(0),
      peak: max.toFixed(0),
      hotspots,
      snapshots: history.length,
    };
  }, [activeGrid, history.length]);

  return (
    <div className="glass-card" id="heatmap-panel">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500/20 to-red-500/20 flex items-center justify-center">
          <Flame size={16} className="text-orange-400" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">Density Heatmap</h3>
          <p className="text-[10px] text-slate-500">Real-time crowd density distribution</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          {/* Mode toggle */}
          <button
            onClick={() => setMode(mode === 'live' ? 'accumulated' : 'live')}
            className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-semibold transition-all ${
              mode === 'accumulated'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/20'
                : 'bg-surface-800/60 text-slate-400 border border-white/[0.06] hover:text-white'
            }`}
            title={mode === 'live' ? 'Switch to accumulated' : 'Switch to live'}
          >
            <Layers size={10} />
            {mode === 'live' ? 'Live' : 'Accumulated'}
          </button>

          {/* Palette toggle */}
          <button
            onClick={() => setPalette(p => p === 'inferno' ? 'jet' : 'inferno')}
            className="px-2 py-1 rounded-lg text-[10px] font-semibold bg-surface-800/60 text-slate-400 border border-white/[0.06] hover:text-white transition-all"
            title="Toggle color palette"
          >
            {palette === 'inferno' ? '🔥 Inferno' : '🌈 Jet'}
          </button>

          {/* Reset accumulated */}
          {mode === 'accumulated' && (
            <button
              onClick={() => setHistory([])}
              className="p-1 rounded-lg bg-surface-800/60 text-slate-400 border border-white/[0.06] hover:text-white transition-all"
              title="Reset accumulated data"
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="relative rounded-xl overflow-hidden border border-white/[0.06] bg-surface-900">
        <canvas
          ref={canvasRef}
          width={480}
          height={320}
          className="w-full h-auto"
          style={{ imageRendering: 'auto' }}
        />

        {/* Opacity slider overlay */}
        <div className="absolute bottom-2 right-2 flex items-center gap-2 px-2 py-1 rounded-lg bg-black/60 backdrop-blur-sm">
          <span className="text-[9px] text-slate-400">Opacity</span>
          <input
            type="range"
            min="0.2"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-16 h-1 accent-orange-500"
          />
        </div>

        {/* Labels toggle */}
        <div className="absolute top-2 right-2">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`px-2 py-0.5 rounded text-[9px] font-medium backdrop-blur-sm transition-all ${
              showLabels
                ? 'bg-white/10 text-white'
                : 'bg-black/40 text-slate-500'
            }`}
          >
            {showLabels ? 'Values ON' : 'Values OFF'}
          </button>
        </div>
      </div>

      {/* Color scale legend */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-[9px] text-slate-500">Low</span>
        <div
          className="flex-1 h-2 rounded-full overflow-hidden"
          style={{
            background: palette === 'inferno'
              ? 'linear-gradient(90deg, #000004, #280b54, #65156e, #9f2a63, #d44842, #f57d15, #fcd11d, #fcffa4)'
              : 'linear-gradient(90deg, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000)',
          }}
        />
        <span className="text-[9px] text-slate-500">High</span>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="mt-3 pt-3 border-t border-white/[0.04] grid grid-cols-4 gap-2">
          <div className="text-center">
            <div className="text-xs font-bold text-white">{stats.peak}</div>
            <div className="text-[9px] text-slate-500">Peak</div>
          </div>
          <div className="text-center">
            <div className="text-xs font-bold text-orange-400">{stats.hotspots}</div>
            <div className="text-[9px] text-slate-500">Hotspots</div>
          </div>
          <div className="text-center">
            <div className="text-xs font-bold text-cyan-400">{stats.cells}</div>
            <div className="text-[9px] text-slate-500">Grid Cells</div>
          </div>
          <div className="text-center">
            <div className="text-xs font-bold text-purple-400">{stats.snapshots}</div>
            <div className="text-[9px] text-slate-500">Snapshots</div>
          </div>
        </div>
      )}
    </div>
  );
}
