"""Send 5 mock sensor readings to the backend so the dashboard updates right away.

Usage:
    python scripts/send_test.py low
    python scripts/send_test.py medium
    python scripts/send_test.py high
    python scripts/send_test.py high --url http://192.168.1.5:8000/items

Only uses the standard library, so it runs with any Python.
"""
import argparse
import json
import random
import time
import urllib.request

DEFAULT_URL = "http://localhost:8000/items"

# (temp, humidity, co2) at the first and last of the 5 readings.
# The readings ramp between the two, so "low" is flat and the others trend upward.
SCENARIOS = {
    "low":    {"start": (32.8, 55, 650),  "end": (32.8, 55, 650)},
    "medium": {"start": (33.0, 56, 700),  "end": (33.6, 58, 800)},
    "high":   {"start": (36.0, 75, 1600), "end": (44.0, 90, 3000)},
}


def build_readings(name):
    s, e = SCENARIOS[name]["start"], SCENARIOS[name]["end"]
    readings = []
    for i in range(5):
        f = i / 4
        readings.append({
            "temperature": round(s[0] + (e[0] - s[0]) * f + random.uniform(-0.05, 0.05), 2),
            "humidity":    round(s[1] + (e[1] - s[1]) * f + random.uniform(-0.3, 0.3), 2),
            "co2_ppm":     round(s[2] + (e[2] - s[2]) * f + random.uniform(-10, 10), 2),
        })
    return readings


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()

    for i, reading in enumerate(build_readings(args.scenario), 1):
        status = post(args.url, reading)
        print(f"[{i}/5] {reading} -> HTTP {status}")
        time.sleep(0.3)
    print("Done. The dashboard should update now.")


if __name__ == "__main__":
    main()
