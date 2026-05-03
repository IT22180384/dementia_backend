"""
Demo data seeder for adaptive scheduling demonstration.
Inserts realistic reminders + interactions for USER-IMALSHA-50-0B71
spread across the past 14 days so the behaviour analysis dashboard
and adaptive scheduler have enough data to display meaningful output.

Run:
    python scripts/seed_demo_data.py
"""

import asyncio
import uuid
from datetime import datetime, timedelta
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGODB_DB_NAME", "dementia_care_db")
USER_ID   = "USER-IMALSHA-50-0B71"

# ---------------------------------------------------------------------------
# Reminder templates — realistic for an elderly care patient
# ---------------------------------------------------------------------------
REMINDER_TEMPLATES = [
    {"title": "Take Morning Medication",     "category": "medication",    "priority": "high",   "hour": 8,  "minute": 0},
    {"title": "Blood Pressure Medication",   "category": "medication",    "priority": "high",   "hour": 8,  "minute": 30},
    {"title": "Breakfast",                   "category": "meal",          "priority": "medium", "hour": 8,  "minute": 0},
    {"title": "Morning Walk",                "category": "exercise",      "priority": "low",    "hour": 9,  "minute": 0},
    {"title": "Take Vitamin Supplements",    "category": "medication",    "priority": "medium", "hour": 10, "minute": 0},
    {"title": "Lunch",                       "category": "meal",          "priority": "medium", "hour": 12, "minute": 30},
    {"title": "Afternoon Medication",        "category": "medication",    "priority": "high",   "hour": 14, "minute": 0},
    {"title": "Physio Exercises",            "category": "exercise",      "priority": "medium", "hour": 15, "minute": 0},
    {"title": "Doctor Appointment",          "category": "appointment",   "priority": "critical","hour": 10, "minute": 0},
    {"title": "Evening Yoga Session",        "category": "exercise",      "priority": "medium", "hour": 18, "minute": 0},
    {"title": "Dinner",                      "category": "meal",          "priority": "medium", "hour": 19, "minute": 0},
    {"title": "Night Medication",            "category": "medication",    "priority": "high",   "hour": 21, "minute": 0},
    {"title": "Drink Water",                 "category": "hygiene",       "priority": "low",    "hour": 11, "minute": 0},
    {"title": "Personal Hygiene",            "category": "hygiene",       "priority": "medium", "hour": 7,  "minute": 30},
    {"title": "Caregiver Check-in Call",     "category": "appointment",   "priority": "high",   "hour": 16, "minute": 0},
]

# ---------------------------------------------------------------------------
# Realistic user responses per interaction type (for Pitt model analysis)
# ---------------------------------------------------------------------------
CONFIRMED_RESPONSES = [
    "Yes I took it",
    "Done",
    "I did it just now",
    "Finished",
    "All done, taken",
    "Yes, completed",
    "I already took my medicine",
    "Done, no problem",
    "Acknowledged, I finished",
    "Yes I am done",
]

CONFUSED_RESPONSES = [
    "What medicine? I don't remember taking anything",
    "Um... did I already do this? I'm not sure",
    "I forget, did I take it?",
    "What was this reminder for? I can't recall",
    "I don't remember... what was I supposed to do?",
    "Hmm wait, I'm not sure if I took it or not",
    "I think I did... maybe? I'm not certain",
    "Can you remind me again? I forgot",
    "Wait what? I don't know what I was doing",
    "I can't remember, did I already finish this?",
]

DELAYED_RESPONSES = [
    "Snoozed for 3 minutes",
    "Not now, a bit later",
    "Give me 5 more minutes",
    "I'll do it soon",
    "Later please",
]

IGNORED_RESPONSES = [
    "",  # no response — timeout
    "...",
]

# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------

def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_reminder(template: dict, day_offset: int) -> dict:
    """Build a reminder document for `day_offset` days ago."""
    now = datetime.now()
    base = now - timedelta(days=day_offset)
    sched = base.replace(hour=template["hour"], minute=template["minute"],
                         second=0, microsecond=0)
    rid = make_id("reminder")
    return {
        "_id": rid,
        "user_id": USER_ID,
        "title": template["title"],
        "description": f"Daily reminder: {template['title'].lower()}",
        "scheduled_time": sched,
        "priority": template["priority"],
        "category": template["category"],
        "status": "completed",
        "repeat_pattern": "daily",
        "repeat_interval_minutes": None,
        "caregiver_ids": [],
        "adaptive_scheduling_enabled": True,
        "escalation_enabled": True,
        "escalation_threshold_minutes": 10,
        "notify_caregiver_on_miss": True,
        "created_at": sched - timedelta(days=1),
        "updated_at": sched,
        "completed_at": sched,
        "alarm_triggered_at": sched,
    }


def make_interaction(reminder: dict, itype: str, response_text: str,
                     response_delay_seconds: float) -> dict:
    """Build a reminder_interactions document."""
    confused   = itype == "confused"
    memory_iss = confused and "remember" in response_text.lower()

    # Realistic cognitive risk: confused=0.55-0.85, ignored=0.4-0.65,
    # delayed=0.15-0.4, confirmed=0.05-0.25
    if itype == "confirmed":
        risk = round(random.uniform(0.05, 0.25), 4)
        rec  = "mark_completed"
        alert = False
    elif itype == "delayed":
        risk = round(random.uniform(0.15, 0.40), 4)
        rec  = "snooze_reminder"
        alert = False
    elif itype == "confused":
        risk = round(random.uniform(0.55, 0.85), 4)
        rec  = "escalate_to_caregiver" if risk > 0.70 else "repeat_reminder"
        alert = risk > 0.70
    else:  # ignored
        risk = round(random.uniform(0.40, 0.65), 4)
        rec  = "repeat_reminder"
        alert = False

    sched = reminder["scheduled_time"]
    interaction_time = sched + timedelta(seconds=response_delay_seconds)

    return {
        "_id": make_id("int"),
        "reminder_id": reminder["_id"],
        "user_id": USER_ID,
        "reminder_category": reminder["category"],
        "interaction_type": itype,
        "interaction_time": interaction_time,
        "user_response_text": response_text,
        "cognitive_risk_score": risk,
        "confusion_detected": confused,
        "memory_issue_detected": memory_iss,
        "response_time_seconds": response_delay_seconds if itype != "ignored" else None,
        "recommended_action": rec,
        "caregiver_alert_triggered": alert,
        "response_score": round(1.0 - risk, 4),
    }


def pick_interaction(day: int, template: dict):
    """
    Choose interaction type based on day and category.
    Early days (1-3): mostly confirmed, few delayed.
    Mid days (4-8): mix including some confusion.
    Recent days (9-14): more confusion on medication reminders.
    """
    cat = template["category"]
    if day <= 3:
        weights = {"confirmed": 0.75, "delayed": 0.20, "confused": 0.03, "ignored": 0.02}
    elif day <= 7:
        weights = {"confirmed": 0.55, "delayed": 0.25, "confused": 0.12, "ignored": 0.08}
    else:  # days 8-14 — more issues to show trend
        if cat == "medication":
            weights = {"confirmed": 0.40, "delayed": 0.20, "confused": 0.28, "ignored": 0.12}
        else:
            weights = {"confirmed": 0.50, "delayed": 0.25, "confused": 0.15, "ignored": 0.10}

    types = list(weights.keys())
    probs = list(weights.values())
    return random.choices(types, weights=probs, k=1)[0]


def pick_response(itype: str) -> str:
    if itype == "confirmed":
        return random.choice(CONFIRMED_RESPONSES)
    elif itype == "confused":
        return random.choice(CONFUSED_RESPONSES)
    elif itype == "delayed":
        return random.choice(DELAYED_RESPONSES)
    else:
        return random.choice(IGNORED_RESPONSES)


def pick_delay(itype: str, day: int) -> float:
    """
    Response delay in seconds. Older days: faster response.
    Recent days: slower (shows adaptive scheduling needs to shift times).
    """
    base_delays = {
        "confirmed": (30, 120),
        "delayed":   (180, 600),
        "confused":  (300, 900),
        "ignored":   (600, 1800),
    }
    lo, hi = base_delays[itype]
    # Recent days: user responds later (triggers time-shift logic)
    if day <= 5:
        return round(random.uniform(lo, lo + (hi - lo) * 0.3), 1)
    else:
        return round(random.uniform(lo + (hi - lo) * 0.5, hi), 1)


async def seed():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    reminders_col    = db["reminders"]
    interactions_col = db["reminder_interactions"]

    print(f"Connected to MongoDB: {DB_NAME}")
    print(f"Seeding data for user: {USER_ID}\n")

    total_reminders    = 0
    total_interactions = 0

    # Generate 14 days of data, 4-6 reminders per day
    for day_offset in range(14, 0, -1):
        # Pick 4-6 templates for this day
        daily_templates = random.sample(REMINDER_TEMPLATES, k=random.randint(4, 6))

        for tmpl in daily_templates:
            reminder = make_reminder(tmpl, day_offset)

            # Skip if already exists (idempotent re-runs)
            existing = await reminders_col.find_one(
                {"user_id": USER_ID, "title": tmpl["title"],
                 "scheduled_time": {"$gte": reminder["scheduled_time"] - timedelta(minutes=1),
                                    "$lte": reminder["scheduled_time"] + timedelta(minutes=1)}}
            )
            if existing:
                continue

            await reminders_col.insert_one(reminder)
            total_reminders += 1

            itype    = pick_interaction(day_offset, tmpl)
            response = pick_response(itype)
            delay    = pick_delay(itype, day_offset)

            interaction = make_interaction(reminder, itype, response, delay)
            await interactions_col.insert_one(interaction)
            total_interactions += 1

            status_icon = {"confirmed": "[OK]", "delayed": "[DELAY]",
                           "confused": "[CONF]", "ignored": "[MISS]"}[itype]
            print(f"  Day -{day_offset:2d} | {tmpl['title'][:30]:<30} | "
                  f"{itype:<9} {status_icon} | risk={interaction['cognitive_risk_score']:.2f}")

    print(f"\nDone! Inserted {total_reminders} reminders + "
          f"{total_interactions} interactions for {USER_ID}")
    print("\nBehaviour dashboard should now show:")
    print("  - Cognitive risk trend (7d / 30d / 90d)")
    print("  - Confirmed / Confused / Delayed / Ignored counts")
    print("  - Daily pattern by hour")
    print("  - Adaptive scheduling will activate (7+ interactions per category)")
    client.close()


if __name__ == "__main__":
    random.seed(42)  # reproducible output
    asyncio.run(seed())
