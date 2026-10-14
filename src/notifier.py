"""
Phone and email notifications for critical alerts.

Channels (either or both):
  Telegram  a bot sends the alert text plus a snapshot of the camera frame
  Email     the same, over SMTP (Gmail works with an app password)

Behaviour:
- only alerts at or above `min_severity` are sent (default: critical)
- each kind of alert (type + zone) is sent when it starts, then at most once
  every `cooldown_seconds` while it keeps going (default 300 s), so a crowd
  that stays over capacity does not flood your phone
- sending runs in a background thread; a slow or failing network never
  slows the camera pipeline, and errors are logged, never raised

Settings come from data/notify.json (written from the dashboard, git-ignored)
and can be overridden by environment variables:
  CROWDGUARD_TELEGRAM_TOKEN, CROWDGUARD_TELEGRAM_CHAT_ID,
  CROWDGUARD_SMTP_HOST, CROWDGUARD_SMTP_PORT, CROWDGUARD_SMTP_USER,
  CROWDGUARD_SMTP_PASSWORD, CROWDGUARD_SMTP_FROM, CROWDGUARD_SMTP_TO
"""
import json
import logging
import os
import queue
import smtplib
import ssl
import threading
import time
import urllib.parse
import urllib.request
import uuid
from email.message import EmailMessage
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}
SECRET_FIELDS = ("telegram_token", "smtp_password")
ENV = {
    "telegram_token": "CROWDGUARD_TELEGRAM_TOKEN",
    "telegram_chat_id": "CROWDGUARD_TELEGRAM_CHAT_ID",
    "smtp_host": "CROWDGUARD_SMTP_HOST",
    "smtp_port": "CROWDGUARD_SMTP_PORT",
    "smtp_user": "CROWDGUARD_SMTP_USER",
    "smtp_password": "CROWDGUARD_SMTP_PASSWORD",
    "smtp_from": "CROWDGUARD_SMTP_FROM",
    "smtp_to": "CROWDGUARD_SMTP_TO",
}
DEFAULTS = {
    "enabled": True,
    "min_severity": "critical",
    "cooldown_seconds": 300,
    "telegram_token": "", "telegram_chat_id": "",
    "smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_password": "",
    "smtp_from": "", "smtp_to": "", "smtp_ssl": False,
    "site_name": "CrowdGuard",
}
TELEGRAM_API = os.environ.get("CROWDGUARD_TELEGRAM_API", "https://api.telegram.org")


def _alert_key(alert: Dict) -> str:
    data = alert.get("data") or {}
    return f"{alert.get('type')}:{data.get('zone', '')}"


class Notifier:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._last_sent: Dict[str, float] = {}
        self._queue: "queue.Queue" = queue.Queue(maxsize=20)
        self.history: List[Dict] = []   # recent send results, for the dashboard
        self.config = self._load()
        threading.Thread(target=self._worker, daemon=True, name="notifier").start()

    # ---- settings -----------------------------------------------------
    def _load(self) -> Dict:
        cfg = dict(DEFAULTS)
        try:
            with open(self.config_path, encoding="utf-8") as f:
                cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass
        for key, env in ENV.items():
            if os.environ.get(env):
                cfg[key] = os.environ[env]
        return cfg

    def update(self, changes: Dict) -> Dict:
        with self._lock:
            cfg = dict(self.config)
            for k, v in changes.items():
                if k not in DEFAULTS:
                    continue
                if k in SECRET_FIELDS and (v is None or v == "" or "•" in str(v)):
                    continue  # unchanged secret (the dashboard only sees it masked)
                cfg[k] = v
            if cfg["min_severity"] not in SEVERITY_RANK:
                raise ValueError("min_severity must be info, warning or critical")
            cfg["cooldown_seconds"] = max(10, int(cfg["cooldown_seconds"]))
            cfg["smtp_port"] = int(cfg["smtp_port"] or 587)
            self.config = cfg
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            fd = os.open(self.config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        return self.public_config()

    def public_config(self) -> Dict:
        """Settings safe to show in the dashboard (secrets masked)."""
        cfg = dict(self.config)
        for k in SECRET_FIELDS:
            cfg[k] = ("•" * 8 + str(cfg[k])[-4:]) if cfg.get(k) else ""
        cfg["telegram_ready"] = self.telegram_ready
        cfg["email_ready"] = self.email_ready
        cfg["recent"] = self.history[-10:]
        return cfg

    @property
    def telegram_ready(self) -> bool:
        return bool(self.config.get("telegram_token") and self.config.get("telegram_chat_id"))

    @property
    def email_ready(self) -> bool:
        c = self.config
        return bool(c.get("smtp_host") and c.get("smtp_to") and (c.get("smtp_from") or c.get("smtp_user")))

    # ---- deciding what to send ----------------------------------------
    def wants(self, alerts: List[Dict], now: Optional[float] = None) -> List[Dict]:
        """Alerts that should go out now (severity filter + cooldown)."""
        c = self.config
        if not c.get("enabled") or not (self.telegram_ready or self.email_ready):
            return []
        now = now or time.time()
        floor = SEVERITY_RANK.get(c.get("min_severity", "critical"), 2)
        out, seen = [], set()
        with self._lock:
            for a in alerts:
                if SEVERITY_RANK.get(a.get("severity"), 0) < floor:
                    continue
                key = _alert_key(a)
                if key in seen or now - self._last_sent.get(key, 0) < c["cooldown_seconds"]:
                    continue
                seen.add(key)
                self._last_sent[key] = now
                out.append(a)
        return out

    def notify(self, alerts: List[Dict], snapshot_jpeg: Optional[bytes] = None, context: str = ""):
        """Queue one message for these alerts (call wants() first)."""
        if not alerts:
            return
        try:
            self._queue.put_nowait((alerts, snapshot_jpeg, context))
        except queue.Full:
            logger.warning("Notification queue full, dropping a message")

    def send_test(self) -> Dict:
        """Send a test message right away (blocking) and report per channel."""
        alert = {"type": "test", "severity": "critical",
                 "message": "Test notification from CrowdGuard. If you can read this, alerts will reach you."}
        return self._send([alert], None, "Test")

    # ---- sending --------------------------------------------------------
    def _worker(self):
        while True:
            alerts, jpeg, context = self._queue.get()
            try:
                self._send(alerts, jpeg, context)
            except Exception as e:  # never let the thread die
                logger.error(f"Notification failed: {e}")

    def _text(self, alerts: List[Dict], context: str) -> str:
        site = self.config.get("site_name") or "CrowdGuard"
        lines = [f"🚨 {site}: {len(alerts)} alert{'s' if len(alerts) > 1 else ''}"
                 + (f" ({context})" if context else "")]
        for a in alerts:
            lines.append(f"• [{a.get('severity', '').upper()}] {a.get('message', a.get('type'))}")
        lines.append(time.strftime("%d %b %Y, %H:%M:%S"))
        return "\n".join(lines)

    def _send(self, alerts: List[Dict], jpeg: Optional[bytes], context: str) -> Dict:
        text = self._text(alerts, context)
        result = {}
        if self.telegram_ready:
            result["telegram"] = self._try(self._send_telegram, text, jpeg)
        if self.email_ready:
            result["email"] = self._try(self._send_email, text, jpeg, alerts)
        self.history.append({"time": int(time.time()), "alerts": [a.get("type") for a in alerts], **result})
        self.history = self.history[-50:]
        return result

    @staticmethod
    def _try(fn, *args) -> str:
        try:
            fn(*args)
            return "sent"
        except Exception as e:
            logger.error(f"{fn.__name__} failed: {e}")
            return f"failed: {e}"

    def _send_telegram(self, text: str, jpeg: Optional[bytes]):
        token = self.config["telegram_token"]
        chat = str(self.config["telegram_chat_id"])
        if jpeg:
            boundary = uuid.uuid4().hex
            parts = []
            for name, value in (("chat_id", chat), ("caption", text[:1000])):
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="snapshot.jpg"\r\n'
                         f'Content-Type: image/jpeg\r\n\r\n'.encode() + jpeg + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode())
            req = urllib.request.Request(f"{TELEGRAM_API}/bot{token}/sendPhoto", data=b"".join(parts),
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        else:
            body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
            req = urllib.request.Request(f"{TELEGRAM_API}/bot{token}/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=15) as r:
            reply = json.load(r)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("description", "Telegram refused the message"))

    def find_telegram_chat_id(self) -> Optional[str]:
        """After the user messages the bot once, read their chat id from it."""
        token = self.config.get("telegram_token")
        if not token:
            raise RuntimeError("Enter the bot token first")
        with urllib.request.urlopen(f"{TELEGRAM_API}/bot{token}/getUpdates", timeout=15) as r:
            reply = json.load(r)
        for upd in reversed(reply.get("result", [])):
            msg = upd.get("message") or upd.get("channel_post") or {}
            if msg.get("chat", {}).get("id") is not None:
                return str(msg["chat"]["id"])
        return None

    def _send_email(self, text: str, jpeg: Optional[bytes], alerts: List[Dict]):
        c = self.config
        msg = EmailMessage()
        top = max(alerts, key=lambda a: SEVERITY_RANK.get(a.get("severity"), 0))
        msg["Subject"] = f"[{c.get('site_name') or 'CrowdGuard'}] {top.get('severity', '').upper()}: {top.get('message', '')}"[:200]
        msg["From"] = c.get("smtp_from") or c.get("smtp_user")
        msg["To"] = c["smtp_to"]
        msg.set_content(text)
        if jpeg:
            msg.add_attachment(jpeg, maintype="image", subtype="jpeg", filename="snapshot.jpg")
        port = int(c.get("smtp_port") or 587)
        if c.get("smtp_ssl") or port == 465:
            server = smtplib.SMTP_SSL(c["smtp_host"], port, timeout=20, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(c["smtp_host"], port, timeout=20)
            server.ehlo()
            if server.has_extn("starttls"):
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
        try:
            if c.get("smtp_user") and c.get("smtp_password"):
                server.login(c["smtp_user"], c["smtp_password"])
            server.send_message(msg)
        finally:
            server.quit()
