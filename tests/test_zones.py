"""
Zone counting and alert checks.  Run:  python tests/test_zones.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from zones import ZoneManager, zone_alerts  # noqa: E402


def det(x, y_feet):
    """A 20x60 person box whose feet are at (x, y_feet)."""
    return {"bbox": [x - 10, y_feet - 60, x + 10, y_feet], "center": [x, y_feet - 30]}


def manager(zones):
    m = ZoneManager(os.path.join(tempfile.mkdtemp(), "zones.json"))
    m.set_zones(zones)
    return m


SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_people_counted_by_feet():
    m = manager([{"name": "A", "type": "monitor", "points": SQUARE, "capacity": 5}])
    # feet inside the square even though the box centre is above it
    stats = m.analyze([det(50, 90), det(50, 150)])
    assert stats[0]["people_count"] == 1


def test_counts_scaled_to_corrected_total():
    m = manager([{"name": "A", "type": "monitor", "points": SQUARE, "capacity": 5}])
    stats = m.analyze([det(50, 50), det(500, 400)], total_count=4)   # 2 seen, 4 estimated
    assert stats[0]["people_count"] == 2


def test_alerts():
    m = manager([
        {"name": "Hall", "type": "monitor", "points": SQUARE, "capacity": 2},
        {"name": "Exit A", "type": "exit", "points": [[200, 0], [300, 0], [300, 100], [200, 100]], "capacity": 1},
        {"name": "Staff", "type": "restricted", "points": [[400, 0], [500, 0], [500, 100], [400, 100]]},
    ])
    people = [det(10, 50), det(20, 50), det(30, 50),        # Hall 3/2
              det(250, 50), det(260, 50),                   # Exit 2/1
              det(450, 50)]                                 # Staff 1
    types = {a["data"]["zone"]: a["type"] for a in zone_alerts(m.analyze(people), 0)}
    assert types == {"Hall": "zone_capacity_exceeded", "Exit A": "zone_capacity_exceeded",
                     "Staff": "zone_restricted"}
    assert any("congested" in a["message"] for a in zone_alerts(m.analyze(people), 0))


def test_warning_at_80_percent_and_quiet_when_empty():
    m = manager([{"name": "Hall", "type": "monitor", "points": SQUARE, "capacity": 5}])
    assert [a["type"] for a in zone_alerts(m.analyze([det(i * 10 + 5, 50) for i in range(4)]), 0)] == ["zone_capacity_threshold"]
    assert zone_alerts(m.analyze([]), 0) == []


def test_validation_and_persistence():
    path = os.path.join(tempfile.mkdtemp(), "zones.json")
    m = ZoneManager(path)
    for bad in ([{"points": [[0, 0], [1, 1]]}], [{"type": "lava", "points": SQUARE}], "nope"):
        try:
            m.set_zones(bad)
            assert False, bad
        except ValueError:
            pass
    m.set_zones([{"name": "Gate", "type": "entry", "points": [[-50, 0], [900, 0], [900, 900]]}])
    again = ZoneManager(path).zones
    assert again[0]["capacity"] == 10                      # default for entry
    assert max(p[0] for p in again[0]["points"]) == 640     # clipped to the frame


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
