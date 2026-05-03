"""
Test: Multi-User Data Isolation
Logs in as Kolitha and Binara separately, then fetches and displays:
  - Profile details
  - Game session history
  - Game analytics / stats

Usage:
    python test_user_data_isolation.py
"""

import requests
import json

BASE_URL = "http://localhost:8080"

USERS = [
    {"email": "kolitha@gmail.com", "password": "kolitha123", "label": "Kolitha"},
    {"email": "binara@gmail.com",  "password": "binara123",  "label": "Binara"},
]

SEP  = "=" * 70
SEP2 = "-" * 50

def pretty(data):
    return json.dumps(data, indent=2, default=str)

def login(email: str, password: str) -> dict | None:
    resp = requests.post(
        f"{BASE_URL}/api/user/login",
        json={"email": email, "password": password},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"  ✗ Login failed [{resp.status_code}]: {resp.text}")
    return None

def get_profile(token: str) -> dict | None:
    resp = requests.get(
        f"{BASE_URL}/api/user/profile",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"  ✗ Profile fetch failed [{resp.status_code}]: {resp.text}")
    return None

def get_game_stats(user_id: str) -> dict | None:
    resp = requests.get(
        f"{BASE_URL}/game/stats/{user_id}",
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"  ✗ Game stats fetch failed [{resp.status_code}]: {resp.text}")
    return None

def get_game_history(user_id: str, limit: int = 5) -> dict | None:
    resp = requests.get(
        f"{BASE_URL}/game/history/{user_id}",
        params={"limit": limit},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"  ✗ Game history fetch failed [{resp.status_code}]: {resp.text}")
    return None

def print_profile(profile: dict):
    p = profile.get("user", profile)   # endpoint may wrap in {"user": {...}}
    fields = ["user_id", "full_name", "email", "phone_number", "age", "gender", "address",
              "emergency_contact_name", "emergency_contact_number", "caregiver_id",
              "account_status", "created_at"]
    for f in fields:
        val = p.get(f, "—")
        print(f"    {f:<32}: {val}")

def print_stats(stats: dict):
    print(f"    Total Sessions    : {stats.get('totalSessions', 0)}")
    print(f"    Avg SAC           : {stats.get('avgSAC', 0)}")
    print(f"    Avg IES           : {stats.get('avgIES', 0)}")
    print(f"    Current Risk Level: {stats.get('currentRiskLevel', 'N/A')}")
    print(f"    Recent Risk Score : {stats.get('recentRiskScore', 0)}")
    print(f"    Last Session Date : {stats.get('lastSessionDate', 'N/A')}")

def print_history(history: dict):
    sessions = history.get("sessions", [])
    total    = history.get("totalSessions", 0)
    print(f"    Total sessions in DB : {total}")
    if not sessions:
        print("    No sessions found.")
        return
    print(f"    Showing last {len(sessions)} session(s):")
    for i, s in enumerate(sessions, 1):
        print(f"\n    [{i}] Session ID  : {s.get('sessionId', '—')}")
        print(f"        Timestamp   : {s.get('timestamp', '—')}")
        print(f"        Game Type   : {s.get('gameType', '—')}")
        print(f"        Level       : {s.get('level', '—')}")
        print(f"        SAC         : {s.get('sac', '—')}")
        print(f"        IES         : {s.get('ies', '—')}")
        print(f"        Risk Level  : {s.get('riskLevel', '—')}")
        print(f"        Risk Score  : {s.get('riskScore', '—')}")

def run_test():
    for user_cfg in USERS:
        label = user_cfg["label"]
        print(f"\n{SEP}")
        print(f"  USER: {label}  ({user_cfg['email']})")
        print(SEP)

        # ── 1. Login ──────────────────────────────────────────────────────────
        print(f"\n  [1] Logging in as {label}...")
        login_data = login(user_cfg["email"], user_cfg["password"])
        if not login_data:
            print(f"  ✗ Skipping {label} — login failed.\n")
            continue

        token   = login_data.get("access_token") or login_data.get("token", "")
        user_id = (login_data.get("user", {}) or {}).get("user_id") or login_data.get("user_id", "")
        print(f"  ✓ Logged in  |  user_id = {user_id}")

        # ── 2. Profile ────────────────────────────────────────────────────────
        print(f"\n{SEP2}")
        print(f"  [2] Profile Details for {label}")
        print(SEP2)
        profile = get_profile(token)
        if profile:
            print_profile(profile)
        else:
            print("  No profile data returned.")

        # ── 3. Game Stats ─────────────────────────────────────────────────────
        print(f"\n{SEP2}")
        print(f"  [3] Game Analytics / Stats for {label}")
        print(SEP2)
        if user_id:
            stats = get_game_stats(user_id)
            if stats:
                print_stats(stats)
            else:
                print("  No stats returned.")
        else:
            print("  ✗ Cannot fetch stats — user_id missing from login response.")

        # ── 4. Game History ───────────────────────────────────────────────────
        print(f"\n{SEP2}")
        print(f"  [4] Game Session History for {label} (last 5)")
        print(SEP2)
        if user_id:
            history = get_game_history(user_id)
            if history:
                print_history(history)
            else:
                print("  No session history returned.")
        else:
            print("  ✗ Cannot fetch history — user_id missing from login response.")

    print(f"\n{SEP}")
    print("  Test complete.")
    print(SEP)

if __name__ == "__main__":
    run_test()
