import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB_NAME")]
    uid = "USER-IMALSHA-50-0B71"
    count = await db["reminder_interactions"].count_documents({"user_id": uid})
    print("Interactions in DB for", uid, ":", count)
    sample = await db["reminder_interactions"].find_one({"user_id": uid})
    if sample:
        print("Sample type:", sample["interaction_type"])
        print("Sample risk:", sample["cognitive_risk_score"])
        print("Sample time:", sample["interaction_time"])
    client.close()

asyncio.run(check())
