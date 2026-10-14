"""
Notification rules (no network needed).  Run:  python tests/test_notifier.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for k in list(os.environ):
    if k.startswith("CROWDGUARD_TELEGRAM") or k.startswith("CROWDGUARD_SMTP"):
        del os.environ[k]
from notifier import Notifier  # noqa: E402


def make(**settings):
    n = Notifier(os.path.join(tempfile.mkdtemp(), "notify.json"))
    n.update({"telegram_token": "123:abc", "telegram_chat_id": "42", **settings})
    return n


def alert(t, sev="critical", zone=None):
    return {"type": t, "severity": sev, "message": t, "data": {"zone": zone} if zone else {}}


def test_nothing_sent_until_set_up():
    n = Notifier(os.path.join(tempfile.mkdtemp(), "notify.json"))
    assert n.wants([alert("panic")]) == []


def test_severity_filter():
    n = make()
    assert [a["type"] for a in n.wants([alert("panic"), alert("filling", "warning")], now=1000)] == ["panic"]
    n.update({"min_severity": "warning"})
    assert [a["type"] for a in n.wants([alert("filling2", "warning")], now=1000)] == ["filling2"]


def test_cooldown_per_kind_and_zone():
    n = make(cooldown_seconds=300)
    first = n.wants([alert("zone_capacity_exceeded", zone="Hall"), alert("zone_capacity_exceeded", zone="Exit")], now=1000)
    assert len(first) == 2                                                 # different zones
    assert n.wants([alert("zone_capacity_exceeded", zone="Hall")], now=1100) == []   # still cooling down
    assert len(n.wants([alert("zone_capacity_exceeded", zone="Hall")], now=1301)) == 1
    assert len(n.wants([alert("panic"), alert("panic")], now=1000)) == 1   # duplicates in one frame


def test_secrets_masked_and_kept():
    n = make(smtp_password="supersecret")
    pub = n.public_config()
    assert "abc" not in pub["telegram_token"][:-4] and pub["telegram_token"].startswith("•")
    assert "supersecret" not in json.dumps(pub)
    n.update({"telegram_token": pub["telegram_token"], "smtp_password": ""})   # dashboard resubmits masked values
    assert n.config["telegram_token"] == "123:abc" and n.config["smtp_password"] == "supersecret"


def test_disabled():
    n = make(enabled=False)
    assert n.wants([alert("panic")]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
