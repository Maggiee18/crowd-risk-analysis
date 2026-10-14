import React, { useEffect, useState } from 'react';
import { Bell, Send, Search, Save } from 'lucide-react';
import * as api from '../services/api';

const panel = { background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' };

function Field({ label, children }) {
  return (
    <label className="block space-y-1">
      <span className="text-[11px] t-muted">{label}</span>
      {children}
    </label>
  );
}

export default function NotificationSettings() {
  const [cfg, setCfg] = useState(null);
  const [form, setForm] = useState({});
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      const res = await api.getNotifySettings();
      setCfg(res.data);
      setForm(res.data);
    } catch (e) {
      setMsg({ ok: false, text: api.errorMessage(e, 'Could not load notification settings') });
    }
  };

  useEffect(() => { load(); }, []);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value });

  const run = async (fn, okText) => {
    setBusy(true);
    setMsg(null);
    try {
      const out = await fn();
      if (out) { setCfg(out); setForm(out); }
      if (okText) setMsg({ ok: true, text: okText });
    } catch (e) {
      setMsg({ ok: false, text: api.errorMessage(e, 'Request failed') });
    } finally {
      setBusy(false);
    }
  };

  const save = () => run(async () => {
    const keys = ['enabled', 'min_severity', 'cooldown_seconds', 'site_name', 'telegram_token', 'telegram_chat_id',
      'smtp_host', 'smtp_port', 'smtp_user', 'smtp_password', 'smtp_from', 'smtp_to', 'smtp_ssl'];
    const settings = Object.fromEntries(keys.map((k) => [k, form[k]]));
    return (await api.updateNotifySettings(settings)).data;
  }, 'Saved');

  const findChat = () => run(async () => {
    await api.updateNotifySettings({ telegram_token: form.telegram_token });
    return (await api.findTelegramChat()).data;
  }, 'Found your chat. Now click "Send test".');

  const test = () => run(async () => {
    const res = await api.sendTestNotification();
    const parts = Object.entries(res.data.result).map(([ch, r]) => `${ch}: ${r}`);
    setMsg({ ok: parts.every((p) => p.endsWith('sent')), text: parts.join(' · ') });
    return (await api.getNotifySettings()).data;
  });

  if (!cfg) {
    return (
      <div className="glass-card">
        <h3 className="text-sm font-semibold t-heading flex items-center gap-2"><Bell size={16} /> Notifications</h3>
        {msg && <p className="text-xs text-red-400 mt-2">{msg.text}</p>}
      </div>
    );
  }

  return (
    <div className="glass-card space-y-4">
      <h3 className="text-sm font-semibold t-heading flex items-center gap-2">
        <Bell size={16} /> Notifications
        <span className={`ml-auto text-[10px] font-semibold ${cfg.telegram_ready || cfg.email_ready ? 'text-emerald-400' : 't-muted'}`}>
          {[cfg.telegram_ready && 'Telegram', cfg.email_ready && 'Email'].filter(Boolean).join(' + ') || 'Not set up'}
        </span>
      </h3>
      <p className="text-xs t-muted">
        Get alerts on your phone or by email with a snapshot of the camera. Each kind of alert is sent when it
        starts, then at most once per cooldown while it continues.
      </p>

      {msg && (
        <div className={`text-xs p-2 rounded-lg ${msg.ok ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
          {msg.text}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <label className="flex items-center gap-2 text-xs t-body">
          <input type="checkbox" checked={!!form.enabled} onChange={set('enabled')} /> Send notifications
        </label>
        <Field label="Send alerts that are at least">
          <select className="input-field w-full" value={form.min_severity} onChange={set('min_severity')}>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
          </select>
        </Field>
        <Field label="Cooldown (seconds)">
          <input className="input-field w-full" type="number" min="10" value={form.cooldown_seconds} onChange={set('cooldown_seconds')} />
        </Field>
      </div>

      <div className="p-3 rounded-xl space-y-3" style={panel}>
        <p className="text-xs font-semibold t-heading">Telegram</p>
        <ol className="text-[11px] t-muted list-decimal ml-4 space-y-0.5">
          <li>In Telegram, open <b>@BotFather</b>, send <code>/newbot</code> and copy the token it gives you.</li>
          <li>Paste it below, then open your new bot and send it any message (like "hi").</li>
          <li>Click <b>Find my chat ID</b>, then <b>Send test</b>.</li>
        </ol>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 items-end">
          <Field label="Bot token">
            <input className="input-field w-full" value={form.telegram_token || ''} onChange={set('telegram_token')}
                   placeholder="123456:ABC..." autoComplete="off" />
          </Field>
          <Field label="Chat ID">
            <input className="input-field w-full" value={form.telegram_chat_id || ''} onChange={set('telegram_chat_id')} />
          </Field>
          <button className="btn-ghost justify-center" onClick={findChat} disabled={busy || !form.telegram_token}>
            <Search size={14} /> Find my chat ID
          </button>
        </div>
      </div>

      <div className="p-3 rounded-xl space-y-3" style={panel}>
        <p className="text-xs font-semibold t-heading">Email</p>
        <p className="text-[11px] t-muted">
          For Gmail use host <code>smtp.gmail.com</code>, port 587, your address as user, and an
          app password (Google Account → Security → App passwords), not your normal password.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <Field label="SMTP host"><input className="input-field w-full" value={form.smtp_host || ''} onChange={set('smtp_host')} /></Field>
          <Field label="Port"><input className="input-field w-full" type="number" value={form.smtp_port || 587} onChange={set('smtp_port')} /></Field>
          <Field label="Send to"><input className="input-field w-full" value={form.smtp_to || ''} onChange={set('smtp_to')} placeholder="you@example.com" /></Field>
          <Field label="User"><input className="input-field w-full" value={form.smtp_user || ''} onChange={set('smtp_user')} autoComplete="off" /></Field>
          <Field label="Password"><input className="input-field w-full" type="password" value={form.smtp_password || ''} onChange={set('smtp_password')} autoComplete="new-password" /></Field>
          <Field label="From (optional)"><input className="input-field w-full" value={form.smtp_from || ''} onChange={set('smtp_from')} /></Field>
        </div>
      </div>

      <div className="flex gap-2">
        <button className="btn-primary" onClick={save} disabled={busy}><Save size={14} /> Save</button>
        <button className="btn-ghost" onClick={test} disabled={busy || !(cfg.telegram_ready || cfg.email_ready)}>
          <Send size={14} /> Send test
        </button>
      </div>

      {cfg.recent && cfg.recent.length > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] t-muted">Recently sent</p>
          {cfg.recent.slice().reverse().slice(0, 5).map((r, i) => (
            <div key={i} className="text-[11px] t-body flex gap-2">
              <span className="t-muted">{new Date(r.time * 1000).toLocaleTimeString()}</span>
              <span>{(r.alerts || []).join(', ')}</span>
              <span className="ml-auto t-muted">
                {['telegram', 'email'].filter((c) => r[c]).map((c) => `${c}: ${r[c]}`).join(' · ')}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
