import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB_NAME")]

    print("=== USERS (patients) ===")
    async for u in db["users"].find({}, {"user_id":1,"username":1,"email":1,"caregiver_id":1}):
        print(" ", u)

    print("\n=== CAREGIVERS ===")
    async for c in db["caregivers"].find({}, {"caregiver_id":1,"username":1,"patient_ids":1}):
        print(" ", c)

    print("\n=== Interaction user_ids (distinct) ===")
    ids = await db["reminder_interactions"].distinct("user_id")
    for i in ids:
        print(" ", i)

    client.close()

asyncio.run(check())
