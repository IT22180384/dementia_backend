"""
Seed behavioral data for user USER-IMALSHA-50-0B71 into MongoDB.
Inserts reminders + reminder_interactions so the behaviour analysis
endpoint returns real counts (confirmed, ignored, confused, delayed).

Run from the dementia_backend folder:
    python scripts/seed_imalsha_behavior.py
"""

import sys
import os
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import random

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://susadisandanima_db_user:Xd1guytlFb69YHZk@cluster0.zogggao.mongodb.net/"
)
DB_NAME = os.getenv("MONGODB_DB_NAME", "dementia_care_db")
USER_ID = "USER-IMALSHA-50-0B71"

# How many interactions of each type to create
COUNTS = {
    "confirmed": 8,
    "ignored":   4,
    "confused":  5,
    "delayed":   3,
}

CATEGORIES = ["medication", "meal", "appointment", "hygiene", "safety"]
PRIORITIES  = ["low", "medium", "high", "critical"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk(interaction_type: str) -> float:
    return {
        "confirmed": random.uniform(0.05, 0.20),
        "delayed":   random.uniform(0.20, 0.40),
        "ignored":   random.uniform(0.50, 0.70),
        "confused":  random.uniform(0.65, 0.90),
    }[interaction_type]


def _rt(interaction_type: str) -> float:
    return {
        "confirmed": random.uniform(5,  20),
        "delayed":   random.uniform(40, 90),
        "ignored":   random.uniform(0,   5),
        "confused":  random.uniform(30,120),
    }[interaction_type]


def _reminder_status(interaction_type: str) -> str:
    return {
        "confirmed": "completed",
        "delayed":   "completed",
        "ignored":   "missed",
        "confused":  "missed",
    }[interaction_type]


def _now_minus(days: int, hour: int = 9) -> datetime:
    base = datetime.utcnow() - timedelta(days=days)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def seed():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    reminders_col     = db["reminders"]
    interactions_col  = db["reminder_interactions"]

    # Spread interactions across the last 30 days
    total = sum(COUNTS.values())
    interaction_list = []
    for itype, count in COUNTS.items():
        interaction_list.extend([itype] * count)
    random.shuffle(interaction_list)

    reminders_inserted = []
    interactions_inserted = []

    for idx, itype in enumerate(interaction_list):
        days_ago = random.randint(1, 29)
        sched_time = _now_minus(days_ago, hour=random.choice([8, 12, 18, 21]))

        # ── Reminder document ──────────────────────────────────────────────
        reminder_id = str(ObjectId())
        reminder_doc = {
            "_id":                       reminder_id,
            "user_id":                   USER_ID,
            "title":                     f"Reminder {idx + 1}",
            "description":               f"Auto-seeded {itype} reminder",
            "scheduled_time":            sched_time,
            "priority":                  random.choice(PRIORITIES),
            "category":                  random.choice(CATEGORIES),
            "repeat_pattern":            None,
            "repeat_interval_minutes":   None,
            "caregiver_ids":             ["CF-VINDIO-2356"],
            "adaptive_scheduling_enabled": True,
            "escalation_enabled":        True,
            "escalation_threshold_minutes": 10,
            "status":                    _reminder_status(itype),
            "notify_caregiver_on_miss":  True,
            "created_at":                sched_time - timedelta(days=1),
            "updated_at":                sched_time,
            "completed_at":              sched_time if itype in ("confirmed", "delayed") else None,
        }
        reminders_col.insert_one(reminder_doc)
        reminders_inserted.append(reminder_id)

        # ── Interaction document ───────────────────────────────────────────
        interaction_id = str(ObjectId())
        rt = _rt(itype)
        risk = _risk(itype)
        interaction_doc = {
            "_id":                    interaction_id,
            "reminder_id":            reminder_id,
            "user_id":                USER_ID,
            "reminder_category":      reminder_doc["category"],
            "interaction_type":       itype,
            "interaction_time":       sched_time + timedelta(seconds=rt),
            "user_response_text":     None,
            "user_response_audio_path": None,
            "cognitive_risk_score":   round(risk, 4),
            "confusion_detected":     itype == "confused",
            "memory_issue_detected":  itype in ("confused", "ignored"),
            "uncertainty_detected":   itype in ("confused", "delayed"),
            "features":               {
                "hesitation_pauses":      random.uniform(0, 10),
                "semantic_incoherence":   risk,
                "low_confidence_answers": random.uniform(0, 1),
            },
            "recommended_action":     "escalate_to_caregiver" if itype == "confused" else "monitor",
            "caregiver_alert_triggered": itype == "confused",
            "response_time_seconds":  round(rt, 2),
            "created_at":             sched_time,
        }
        interactions_col.insert_one(interaction_doc)
        interactions_inserted.append(interaction_id)

    client.close()

    print(f"\nSeeded {len(reminders_inserted)} reminders and {len(interactions_inserted)} interactions for {USER_ID}")
    print(f"  confirmed : {COUNTS['confirmed']}")
    print(f"  ignored   : {COUNTS['ignored']}")
    print(f"  confused  : {COUNTS['confused']}")
    print(f"  delayed   : {COUNTS['delayed']}")
    print("\nRefresh the Behaviour Analysis tab — counts should now be non-zero.\n")


if __name__ == "__main__":
    seed()
