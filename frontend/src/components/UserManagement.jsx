import React, { useEffect, useState } from 'react';
import { UserPlus, Trash2, KeyRound, Users } from 'lucide-react';
import * as api from '../services/api';

const panel = { background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' };

export default function UserManagement({ user, onPasswordChanged }) {
  const isAdmin = user && user.role === 'admin';
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ username: '', password: '', role: 'viewer' });
  const [pw, setPw] = useState({ old: '', next: '' });
  const [msg, setMsg] = useState(null);

  const show = (text, ok = true) => setMsg({ text, ok });

  const refresh = async () => {
    if (!isAdmin) return;
    try {
      const res = await api.listUsers();
      setUsers(res.data.users);
    } catch (e) {
      show(api.errorMessage(e, 'Could not load users'), false);
    }
  };

  useEffect(() => { refresh(); }, [isAdmin]); // eslint-disable-line react-hooks/exhaustive-deps

  const addUser = async (e) => {
    e.preventDefault();
    try {
      const res = await api.createUser(form.username, form.password, form.role);
      setUsers(res.data.users);
      show(`Created ${form.username} (${form.role})`);
      setForm({ username: '', password: '', role: 'viewer' });
    } catch (err) {
      show(api.errorMessage(err, 'Could not create user'), false);
    }
  };

  const removeUser = async (name) => {
    if (!window.confirm(`Delete account "${name}"?`)) return;
    try {
      const res = await api.deleteUser(name);
      setUsers(res.data.users);
      show(`Deleted ${name}`);
    } catch (err) {
      show(api.errorMessage(err, 'Could not delete user'), false);
    }
  };

  const changePassword = async (e) => {
    e.preventDefault();
    try {
      await api.changePassword(pw.old, pw.next);
      setPw({ old: '', next: '' });
      show('Password changed. Please sign in again.');
      setTimeout(onPasswordChanged, 1200);
    } catch (err) {
      show(api.errorMessage(err, 'Could not change password'), false);
    }
  };

  return (
    <div className="glass-card space-y-4">
      <h3 className="text-sm font-semibold t-heading flex items-center gap-2">
        <Users size={16} /> Accounts
      </h3>
      <p className="text-xs t-muted">
        Signed in as <b>{user ? user.username : '?'}</b> ({user ? user.role : '?'}).
        {isAdmin ? ' Admins control the system and manage accounts.' : ' Viewers can watch but not change settings.'}
      </p>

      {msg && (
        <div className={`text-xs p-2 rounded-lg ${msg.ok ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
          {msg.text}
        </div>
      )}

      {isAdmin && (
        <>
          <div className="space-y-2">
            {users.map((u) => (
              <div key={u.username} className="flex items-center justify-between p-2 rounded-xl" style={panel}>
                <span className="text-sm t-body">{u.username}</span>
                <span className="flex items-center gap-3">
                  <span className="text-xs t-muted capitalize">{u.role}</span>
                  {u.username !== user.username && (
                    <button onClick={() => removeUser(u.username)} className="text-red-400 hover:text-red-300" title="Delete">
                      <Trash2 size={14} />
                    </button>
                  )}
                </span>
              </div>
            ))}
          </div>

          <form onSubmit={addUser} className="grid grid-cols-1 sm:grid-cols-4 gap-2">
            <input className="input-field" placeholder="Username" value={form.username}
                   onChange={(e) => setForm({ ...form, username: e.target.value })} />
            <input className="input-field" type="password" placeholder="Password (8+)" value={form.password}
                   onChange={(e) => setForm({ ...form, password: e.target.value })} autoComplete="new-password" />
            <select className="input-field" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
              <option value="viewer">Viewer</option>
              <option value="admin">Admin</option>
            </select>
            <button type="submit" className="btn-primary justify-center" disabled={!form.username || !form.password}>
              <UserPlus size={14} /> Add
            </button>
          </form>
        </>
      )}

      <form onSubmit={changePassword} className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2"
            style={{ borderTop: '1px solid var(--card-border)' }}>
        <input className="input-field" type="password" placeholder="Current password" value={pw.old}
               onChange={(e) => setPw({ ...pw, old: e.target.value })} autoComplete="current-password" />
        <input className="input-field" type="password" placeholder="New password (8+)" value={pw.next}
               onChange={(e) => setPw({ ...pw, next: e.target.value })} autoComplete="new-password" />
        <button type="submit" className="btn-ghost justify-center whitespace-nowrap" disabled={!pw.old || !pw.next}>
          <KeyRound size={14} /> Change password
        </button>
      </form>
    </div>
  );
}
