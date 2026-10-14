"""
User accounts and login tokens for the CrowdGuard API.

Standard library only (no extra packages):
- passwords are stored as salted scrypt hashes, never as plain text
- login tokens are signed with HMAC-SHA256 and expire (default 12 hours)
- two roles: 'admin' (can control the system and manage users) and
  'viewer' (can watch the dashboard and stream a camera)
- repeated wrong passwords from the same address are slowed down

Accounts live in data/users.json and the signing key in data/secret.key
(both git-ignored). On the very first start, an 'admin' account is created
with the password from the CROWDGUARD_ADMIN_PASSWORD environment variable,
or a random one that is printed once in the server log.

Manage accounts from a terminal with:  python api/manage_users.py --help
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Dict, List, Optional

ROLES = ("admin", "viewer")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN_HOURS = float(os.environ.get("CROWDGUARD_TOKEN_HOURS", "12"))
MIN_PASSWORD_LEN = 8

# scrypt cost: ~50 ms per check on a laptop, which is what makes guessing slow
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


class AuthError(Exception):
    """Raised for bad credentials, bad tokens or invalid account changes."""


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        check = hashlib.scrypt(password.encode(), salt=_unb64(salt), **_SCRYPT)
        return hmac.compare_digest(check, _unb64(digest))
    except Exception:
        return False


class UserStore:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = os.path.abspath(data_dir)
        self.users_path = os.path.join(self.data_dir, "users.json")
        self.secret_path = os.path.join(self.data_dir, "secret.key")
        self._lock = threading.Lock()
        self._failures: Dict[str, List[float]] = {}
        os.makedirs(self.data_dir, exist_ok=True)
        self._secret = self._load_secret()
        self._users = self._load_users()

    # ---- storage ------------------------------------------------------
    def _load_secret(self) -> bytes:
        env = os.environ.get("CROWDGUARD_SECRET")
        if env:
            return env.encode()
        if os.path.exists(self.secret_path):
            with open(self.secret_path, "rb") as f:
                return f.read()
        key = secrets.token_bytes(32)
        self._write_private(self.secret_path, key)
        return key

    def _load_users(self) -> Dict[str, Dict]:
        if os.path.exists(self.users_path):
            with open(self.users_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        tmp = self.users_path + ".tmp"
        self._write_private(tmp, json.dumps(self._users, indent=2).encode())
        os.replace(tmp, self.users_path)

    @staticmethod
    def _write_private(path: str, data: bytes):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    def ensure_admin(self) -> Optional[str]:
        """
        Create the first admin account if there are no accounts yet.
        Returns the generated password (to show once), or None.
        """
        with self._lock:
            if self._users:
                return None
            password = os.environ.get("CROWDGUARD_ADMIN_PASSWORD")
            generated = None
            if not password:
                password = generated = secrets.token_urlsafe(9)
            self._users["admin"] = {"password": hash_password(password), "role": "admin",
                                    "created": int(time.time()), "token_version": 0}
            self._save()
            return generated

    # ---- accounts -----------------------------------------------------
    def list_users(self) -> List[Dict]:
        return [{"username": u, "role": d["role"], "created": d.get("created")}
                for u, d in sorted(self._users.items())]

    def add_user(self, username: str, password: str, role: str = "viewer"):
        username = username.strip().lower()
        if not username or not username.replace("_", "").replace(".", "").isalnum():
            raise AuthError("Username may only contain letters, numbers, '.' and '_'")
        if role not in ROLES:
            raise AuthError(f"Role must be one of {ROLES}")
        if len(password) < MIN_PASSWORD_LEN:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            if username in self._users:
                raise AuthError("That username already exists")
            self._users[username] = {"password": hash_password(password), "role": role,
                                     "created": int(time.time()), "token_version": 0}
            self._save()

    def delete_user(self, username: str):
        with self._lock:
            if username not in self._users:
                raise AuthError("No such user")
            admins = [u for u, d in self._users.items() if d["role"] == "admin"]
            if self._users[username]["role"] == "admin" and len(admins) == 1:
                raise AuthError("Cannot delete the last admin")
            del self._users[username]
            self._save()

    def set_password(self, username: str, new_password: str):
        if len(new_password) < MIN_PASSWORD_LEN:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters")
        with self._lock:
            if username not in self._users:
                raise AuthError("No such user")
            u = self._users[username]
            u["password"] = hash_password(new_password)
            u["token_version"] = u.get("token_version", 0) + 1   # logs out old sessions
            self._save()

    # ---- login and tokens ---------------------------------------------
    def _throttled(self, client: str) -> bool:
        now = time.time()
        recent = [t for t in self._failures.get(client, []) if now - t < 60]
        self._failures[client] = recent
        return len(recent) >= 5

    def login(self, username: str, password: str, client: str = "?") -> Dict:
        if self._throttled(client):
            raise AuthError("Too many failed attempts. Wait a minute and try again.")
        username = (username or "").strip().lower()
        user = self._users.get(username)
        # verify even for unknown users so timing doesn't reveal which exist
        ok = verify_password(password or "", user["password"] if user else hash_password("x"))
        if not (user and ok):
            self._failures.setdefault(client, []).append(time.time())
            raise AuthError("Wrong username or password")
        self._failures.pop(client, None)
        exp = int(time.time() + TOKEN_HOURS * 3600)
        return {"token": self.make_token(username, user["role"], exp),
                "user": {"username": username, "role": user["role"]},
                "expires_at": exp}

    def make_token(self, username: str, role: str, exp: int) -> str:
        ver = self._users.get(username, {}).get("token_version", 0)
        body = _b64(json.dumps({"sub": username, "role": role, "exp": exp, "v": ver},
                               separators=(",", ":")).encode())
        sig = _b64(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())
        return f"{body}.{sig}"

    def verify_token(self, token: str) -> Dict:
        """Returns {'username', 'role'} or raises AuthError."""
        try:
            body, sig = (token or "").split(".")
        except ValueError:
            raise AuthError("Not logged in")
        good = _b64(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise AuthError("Invalid login token")
        data = json.loads(_unb64(body))
        if data.get("exp", 0) < time.time():
            raise AuthError("Login expired, please log in again")
        user = self._users.get(data.get("sub"))
        if not user or user.get("token_version", 0) != data.get("v"):
            raise AuthError("Login no longer valid, please log in again")
        return {"username": data["sub"], "role": user["role"]}
