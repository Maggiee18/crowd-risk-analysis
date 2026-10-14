"""
Account and token checks.  Run:  python tests/test_auth.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import auth  # noqa: E402
from auth import AuthError, UserStore, hash_password, verify_password  # noqa: E402


def store():
    return UserStore(tempfile.mkdtemp())


def raises(fn, *a):
    try:
        fn(*a)
    except AuthError:
        return True
    return False


def test_passwords_are_hashed():
    h = hash_password("correct horse")
    assert "correct horse" not in h and h.startswith("scrypt$")
    assert verify_password("correct horse", h) and not verify_password("wrong", h)


def test_first_start_creates_admin_with_generated_password():
    s = store()
    os.environ.pop("CROWDGUARD_ADMIN_PASSWORD", None)
    pw = s.ensure_admin()
    assert pw and s.login("admin", pw)["user"]["role"] == "admin"
    assert s.ensure_admin() is None              # only once


def test_tokens_roles_and_tampering():
    s = store()
    s.add_user("guard1", "watchtower1", "viewer")
    tok = s.login("guard1", "watchtower1")["token"]
    assert s.verify_token(tok) == {"username": "guard1", "role": "viewer"}
    assert raises(s.verify_token, tok[:-2] + "xx")          # bad signature
    body, sig = tok.split(".")
    fake = auth._b64(b'{"sub":"guard1","role":"admin","exp":9999999999,"v":0}')
    assert raises(s.verify_token, f"{fake}.{sig}")          # can't promote yourself


def test_expired_token_rejected():
    s = store()
    s.add_user("guard1", "watchtower1")
    assert raises(s.verify_token, s.make_token("guard1", "viewer", int(time.time()) - 1))


def test_password_change_logs_out_old_sessions():
    s = store()
    s.add_user("guard1", "watchtower1")
    old = s.login("guard1", "watchtower1")["token"]
    s.set_password("guard1", "newwatch123")
    assert raises(s.verify_token, old)
    assert raises(s.login, "guard1", "watchtower1")
    assert s.login("guard1", "newwatch123")


def test_rules():
    s = store()
    os.environ["CROWDGUARD_ADMIN_PASSWORD"] = "adminpass1"
    s.ensure_admin()
    assert raises(s.add_user, "x", "short")                 # too short
    assert raises(s.add_user, "bad name!", "longenough1")   # bad characters
    assert raises(s.delete_user, "admin")                   # last admin
    os.environ.pop("CROWDGUARD_ADMIN_PASSWORD")


def test_brute_force_is_throttled():
    s = store()
    s.add_user("guard1", "watchtower1")
    for _ in range(5):
        assert raises(s.login, "guard1", "nope", "1.2.3.4")
    try:
        s.login("guard1", "watchtower1", "1.2.3.4")
        assert False, "should be throttled"
    except AuthError as e:
        assert "Too many" in str(e)
    assert s.login("guard1", "watchtower1", "5.6.7.8")      # other addresses unaffected


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
