import React, { useEffect, useState, useCallback } from 'react';
import { Camera, Plus, Trash2, Power, ArrowUpRight, Monitor, Play, AlertTriangle } from 'lucide-react';
import * as api from '../services/api';

const RISK_STYLE = {
  low: 'bg-emerald-500/15 text-emerald-400',
  medium: 'bg-amber-500/15 text-amber-400',
  high: 'bg-red-500/15 text-red-400',
};
const STATUS_STYLE = {
  live: 'text-emerald-400',
  connecting: 'text-amber-400',
  idle: 't-muted',
  off: 't-muted',
  offline: 'text-red-400',
};

const panel = { background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' };

function KindIcon({ kind }) {
  if (kind === 'browser') return <Monitor size={14} />;
  if (kind === 'simulation') return <Play size={14} />;
  return <Camera size={14} />;
}

function statusText(c) {
  if (c.kind === 'browser') return c.status === 'live' ? 'Receiving video' : 'Start a webcam or video on Live Feed';
  if (c.kind === 'simulation') return c.status === 'live' ? 'Running' : 'Stopped';
  if (c.enabled === false) return 'Turned off';
  if (c.status === 'offline') return c.error || 'Offline';
  if (c.status === 'connecting') return 'Connecting…';
  return `Live · ${c.source_fps || 0} fps from camera`;
}

function CameraCard({ cam, selected, isAdmin, onOpen, onToggle, onDelete }) {
  return (
    <div className={`glass-card !p-3 space-y-2 ${selected ? 'ring-2 ring-brand-500' : ''}`}>
      <div className="flex items-center gap-2">
        <span className="t-muted"><KindIcon kind={cam.kind} /></span>
        <span className="text-sm font-semibold t-heading truncate">{cam.name}</span>
        {selected && <span className="text-[10px] font-semibold text-brand-400">Showing</span>}
        <span className={`ml-auto text-[10px] font-semibold px-2 py-0.5 rounded-lg uppercase ${RISK_STYLE[cam.risk_level] || RISK_STYLE.low}`}>
          {cam.risk_level}
        </span>
      </div>

      <div className="relative rounded-lg overflow-hidden bg-black/40" style={{ aspectRatio: '4 / 3' }}>
        {cam.thumbnail ? (
          <img src={`data:image/jpeg;base64,${cam.thumbnail}`} alt={cam.name} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-[11px] t-muted px-4 text-center">
            {statusText(cam)}
          </div>
        )}
        {cam.alert_count > 0 && (
          <div className={`absolute top-2 left-2 flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-lg ${cam.critical ? 'bg-red-600 text-white animate-pulse' : 'bg-amber-500 text-black'}`}>
            <AlertTriangle size={10} /> {cam.alert_count} alert{cam.alert_count > 1 ? 's' : ''}
          </div>
        )}
        {cam.panic && (
          <div className="absolute top-2 right-2 text-[10px] font-bold px-2 py-0.5 rounded-lg bg-red-600 text-white">PANIC</div>
        )}
      </div>

      <div className="flex items-end gap-3">
        <div>
          <div className="text-xl font-bold t-heading leading-none">{cam.people_count}</div>
          <div className="text-[10px] t-muted">people</div>
        </div>
        <div className="text-[10px] t-muted leading-tight">
          <div className={STATUS_STYLE[cam.enabled === false ? 'off' : cam.status] || 't-muted'}>{statusText(cam)}</div>
          <div>{cam.zones} zone{cam.zones === 1 ? '' : 's'}</div>
        </div>
        <div className="ml-auto flex gap-1">
          {isAdmin && cam.kind === 'server' && (
            <>
              <button className="btn-ghost !p-1.5" title={cam.enabled === false ? 'Turn on' : 'Turn off'} onClick={() => onToggle(cam)}>
                <Power size={13} className={cam.enabled === false ? 't-muted' : 'text-emerald-400'} />
              </button>
              <button className="btn-ghost !p-1.5" title="Remove camera" onClick={() => onDelete(cam)}>
                <Trash2 size={13} className="text-red-400" />
              </button>
            </>
          )}
          <button className="btn-primary !py-1.5 !px-3 text-[11px]" onClick={() => onOpen(cam)}>
            Open <ArrowUpRight size={12} />
          </button>
        </div>
      </div>
      {isAdmin && cam.source && <div className="text-[10px] t-muted truncate" title={cam.source}>{cam.source}</div>}
    </div>
  );
}

export default function CamerasPage({ selectedCamera, onOpenCamera, isAdmin }) {
  const [cams, setCams] = useState([]);
  const [form, setForm] = useState({ name: '', source: '' });
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await api.listCameras(true);
      setCams(res.data.cameras || []);
    } catch (e) {
      setMsg({ ok: false, text: api.errorMessage(e, 'Could not load cameras') });
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [load]);

  const add = async (e) => {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    try {
      const res = await api.addCamera(form.name, form.source);
      setForm({ name: '', source: '' });
      setMsg({ ok: true, text: `Added "${res.data.name}". It shows up below once the first frame is analysed.` });
      load();
    } catch (err) {
      setMsg({ ok: false, text: api.errorMessage(err, 'Could not add the camera') });
    } finally {
      setBusy(false);
    }
  };

  const toggle = async (cam) => {
    try {
      await api.updateCamera(cam.id, { enabled: cam.enabled === false });
      load();
    } catch (err) {
      setMsg({ ok: false, text: api.errorMessage(err, 'Could not change the camera') });
    }
  };

  const remove = async (cam) => {
    if (!window.confirm(`Remove camera "${cam.name}"?`)) return;
    try {
      await api.deleteCamera(cam.id);
      if (selectedCamera === cam.id) onOpenCamera({ id: 'main' }, false);
      load();
    } catch (err) {
      setMsg({ ok: false, text: api.errorMessage(err, 'Could not remove the camera') });
    }
  };

  const live = cams.filter((c) => c.status === 'live').length;
  const people = cams.reduce((s, c) => s + (c.status === 'live' ? c.people_count : 0), 0);
  const alerting = cams.filter((c) => c.alert_count > 0).length;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="glass-card flex flex-wrap items-center gap-6">
        <div>
          <div className="text-2xl font-bold t-heading">{people}</div>
          <div className="text-[11px] t-muted">people across live cameras</div>
        </div>
        <div>
          <div className="text-2xl font-bold t-heading">{live}<span className="text-sm t-muted"> / {cams.length}</span></div>
          <div className="text-[11px] t-muted">cameras live</div>
        </div>
        <div>
          <div className={`text-2xl font-bold ${alerting ? 'text-red-400' : 't-heading'}`}>{alerting}</div>
          <div className="text-[11px] t-muted">cameras with alerts</div>
        </div>
        <p className="text-[11px] t-muted ml-auto max-w-sm">
          Each camera has its own counts, zones, alerts and forecasts. Open one to see it on every page;
          the camera picker at the top switches too.
        </p>
      </div>

      {msg && (
        <div className={`text-xs p-2 rounded-lg ${msg.ok ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
          {msg.text}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {cams.map((c) => (
          <CameraCard key={c.id} cam={c} selected={c.id === selectedCamera} isAdmin={isAdmin}
                      onOpen={(cam) => onOpenCamera(cam, true)} onToggle={toggle} onDelete={remove} />
        ))}
      </div>

      {isAdmin && (
        <form onSubmit={add} className="glass-card space-y-3">
          <h3 className="text-sm font-semibold t-heading flex items-center gap-2"><Plus size={16} /> Add a camera</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 items-end">
            <label className="block space-y-1">
              <span className="text-[11px] t-muted">Name</span>
              <input className="input-field w-full" value={form.name} placeholder="Main gate"
                     onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </label>
            <label className="block space-y-1 sm:col-span-2">
              <span className="text-[11px] t-muted">Source</span>
              <input className="input-field w-full" value={form.source} autoComplete="off"
                     placeholder="0   or   rtsp://user:pass@192.168.1.20/stream   or   C:\videos\gate.mp4"
                     onChange={(e) => setForm({ ...form, source: e.target.value })} />
            </label>
          </div>
          <div className="p-3 rounded-xl text-[11px] t-muted space-y-1" style={panel}>
            <div><b className="t-body">Webcam on this computer:</b> <code>0</code> for the first one, <code>1</code> for the next. Close other apps using it first.</div>
            <div><b className="t-body">Phone as a camera:</b> install an IP camera app (for example "IP Webcam" on Android), start it, and use the address it shows plus <code>/video</code>, like <code>http://192.168.1.5:8080/video</code>. The phone and computer must be on the same WiFi.</div>
            <div><b className="t-body">CCTV / IP camera:</b> its RTSP address, like <code>rtsp://user:password@192.168.1.20:554/stream1</code>.</div>
            <div><b className="t-body">Video file:</b> the full path on the computer running the backend, like <code>C:\Users\HP\Videos\mall.mp4</code>. It plays on a loop.</div>
          </div>
          <button className="btn-primary" disabled={busy || !form.name || !form.source}><Plus size={14} /> Add camera</button>
        </form>
      )}
    </div>
  );
}
