# src/features/mmse_screening/mmse_db_service.py

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME")

if not MONGO_URL:
    raise RuntimeError("MONGODB_URI not set in environment variables")

_client = AsyncIOMotorClient(MONGO_URL)
_db = _client[DATABASE_NAME]


class MMSEDatabaseService:
    def __init__(self):
        self.users = _db["users"]
        self.assessments = _db["assessments"]
        self.caregivers = _db["caregivers"]

    # ----------------------------
    # User
    # ----------------------------
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self.users.find_one({"user_id": user_id})

    # ----------------------------
    # Assessment lifecycle
    # ----------------------------
    async def create_assessment(self, user_id: str) -> str:
        doc = {
            "user_id": user_id,
            "assessment_type": "MMSE",
            "assessment_date": datetime.utcnow(),
            "questions": [],
            "total_score": 0,
            "ml_summary": {},
            "status": "in_progress",
        }

        result = await self.assessments.insert_one(doc)
        return str(result.inserted_id)

    async def add_question(
        self,
        assessment_id: str,
        user_id: str,
        question_doc: Dict[str, Any],
    ):
        await self.assessments.update_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id},
            {"$push": {"questions": question_doc}},
        )

    async def get_assessment(
        self, assessment_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:

        if not ObjectId.is_valid(assessment_id):
            return None

        return await self.assessments.find_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id}
        )

    async def finalize_assessment(
        self,
        assessment_id: str,
        total_score: float,
        avg_ml_prob: float,
        ml_label: str,
    ):
        await self.assessments.update_one(
            {"_id": ObjectId(assessment_id)},
            {
                "$set": {
                    "total_score": total_score,
                    "ml_summary": {
                        "avg_probability": avg_ml_prob,
                        "ml_risk_label": ml_label,
                    },
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                }
            },
        )

    # ----------------------------
    # Get assessments by user
    # ----------------------------
    async def get_user_assessments(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all MMSE assessments for a given user.
        """

        cursor = self.assessments.find(
            {"user_id": user_id, "assessment_type": "MMSE"}
        ).sort("assessment_date", -1)

        results = []

        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)

        return results

    # ----------------------------
    # Get caregiver patients with assessments
    # ----------------------------
    async def get_patients_with_assessments(
        self, caregiver_id: str
    ) -> List[Dict[str, Any]]:

        caregiver = await self.caregivers.find_one({"caregiver_id": caregiver_id})

        if not caregiver:
            return []

        patient_ids = caregiver.get("patient_ids", [])

        patients_data = []

        for pid in patient_ids:

            user = await self.users.find_one({"user_id": pid})

            if not user:
                continue

            assessments_cursor = self.assessments.find(
                {"user_id": pid, "assessment_type": "MMSE"}
            ).sort("assessment_date", -1)

            assessments = []

            async for doc in assessments_cursor:
                doc["_id"] = str(doc["_id"])
                assessments.append(doc)

            patients_data.append(
                {
                    "user_id": user["user_id"],
                    "full_name": user.get("full_name"),
                    "email": user.get("email"),
                    "age": user.get("age"),
                    "gender": user.get("gender"),
                    "assessments": assessments,
                }
            )

        return patients_data


# Singleton instance
db_service = MMSEDatabaseService()