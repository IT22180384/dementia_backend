# Quick Reference: Game Components Integration

## ✅ Integration Complete

The game-based cognitive assessment has been successfully integrated into `src/api/app.py` without modifying any existing team code.

---

## 📋 What's New

### New API Endpoints (All under `/game` prefix)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/game/session` | Submit game session for risk assessment |
| POST | `/game/calibration` | Calibrate user's motor baseline |
| GET | `/game/history/{userId}?limit=20` | Get user's session history |
| GET | `/game/stats/{userId}` | Get user statistics (dashboard) |
| DELETE | `/game/session/{sessionId}` | Delete a session (testing) |

### Example Usage

#### 1. Motor Baseline Calibration (First-time setup)
```bash
curl -X POST http://localhost:8000/game/calibration \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "reactionTimes": [250, 260, 245, 255, 250]
  }'
```

**Response:**
```json
{
  "userId": "user123",
  "motorBaseline": 250.0,
  "calibrationDate": "2026-01-02T12:00:00",
  "message": "Motor baseline calibrated successfully"
}
```

#### 2. Submit Game Session
```bash
curl -X POST http://localhost:8000/game/session \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "sessionId": "session_001",
    "gameData": {
      "cards": [
        {"cardId": "A1", "firstClickTime": 1000, "secondClickTime": 2500, "isMatch": true},
        {"cardId": "A2", "firstClickTime": 3000, "secondClickTime": 4200, "isMatch": true}
      ]
    },
    "summary": {
      "totalPairs": 8,
      "correctPairs": 7,
      "totalTime": 120.5,
      "attempts": 16
    }
  }'
```

**Response:**
```json
{
  "sessionId": "session_001",
  "userId": "user123",
  "timestamp": "2026-01-02T12:00:00",
  "features": {
    "sac": 0.875,
    "ies": 1580.5,
    "accuracy": 0.875,
    "avgMatchTime": 1250.0,
    "avgMismatchTime": 1800.0,
    "totalAttempts": 16
  },
  "riskAssessment": {
    "riskLevel": "LOW",
    "riskScore": 0.15,
    "probabilities": {
      "LOW": 0.85,
      "MEDIUM": 0.12,
      "HIGH": 0.03
    }
  }
}
```

#### 3. Get User History
```bash
curl http://localhost:8000/game/history/user123?limit=10
```

#### 4. Get User Statistics
```bash
curl http://localhost:8000/game/stats/user123
```

---

## 🔧 Running the Server

```bash
# Option 1: Direct uvicorn
cd d:\SLIIT\4Y\Research\dementia\Implementation\1\dementia_backend
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Option 2: Python script
python -m uvicorn src.api.app:app --reload

# Option 3: Using the app directly
python src/api/app.py
```

---

## 📊 What Happens on Startup

```
================================================================================
Dementia Detection & Monitoring API starting up...
================================================================================
✓ MongoDB connected (conversational AI collections)
✓ Game component indexes created
✓ Game ML models loaded
================================================================================
API ready to serve requests
================================================================================
```

---

## 🔍 Checking Integration

### Visit API Documentation
```
http://localhost:8000/docs
```

### Test Root Endpoint
```bash
curl http://localhost:8000/
```

Should show:
```json
{
  "message": "Dementia Detection & Monitoring API",
  "version": "2.0.0",
  "features": {
    "conversational_ai": "Analyze speech patterns for dementia indicators",
    "gamified_assessment": "Card-matching game with cognitive risk scoring"
  },
  "components": {
    "conversational": [...],
    "game": ["/game/session", "/game/calibration", ...]
  }
}
```

---

## 🎮 Game Component Architecture

### Database Collections (Automatic)
- **game_sessions**: Stores all game sessions with features
- **calibrations**: Motor baseline per user
- **alerts**: High-risk alerts

### ML Models Loaded
- **LSTM Model**: Temporal trend analysis
- **Risk Classifier**: 3-class classification (LOW/MEDIUM/HIGH)
- **Scalers**: Feature normalization

---

## ⚙️ Configuration

Make sure `.env` contains:
```env
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=dementia_detection
```

---

## 🛡️ Existing Features (Unchanged)

All conversational AI endpoints still work:
- ✅ `/api/analyze` - Analyze message
- ✅ `/api/session` - Analyze session
- ✅ `/api/predict` - Direct prediction
- ✅ `/api/features` - List features
- ✅ `/api/risk-levels` - Risk definitions
- ✅ `/health` - Health check
- ✅ Reminder routes

---

## 📝 Frontend Integration Tips

### Workflow
1. **First Time User**: Call `/game/calibration` to set motor baseline
2. **During Game**: Collect card click data
3. **After Game**: Call `/game/session` with game data
4. **Dashboard**: Use `/game/history` and `/game/stats`

### TypeScript Example
```typescript
// Calibration
async function calibrateUser(userId: string, reactionTimes: number[]) {
  const response = await fetch('http://localhost:8000/game/calibration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, reactionTimes })
  });
  return await response.json();
}

// Submit session
async function submitGameSession(sessionData: GameSession) {
  const response = await fetch('http://localhost:8000/game/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sessionData)
  });
  return await response.json();
}
```

---

## 🚨 Troubleshooting

### Models not loading?
- Check `src/models/game/lstm_model/` contains `lstm_model.keras` or `lstm_model.h5`
- Check `src/models/game/risk_classifier/` contains `risk_classifier.pkl`
- Server will use dummy models as fallback (for testing)

### MongoDB connection issues?
- Check `.env` file has correct `MONGODB_URL`
- Ensure MongoDB is running locally

### Import errors?
- Ensure `src/routes/game_routes.py` exists
- Ensure `src/models/game/model_registry.py` exists

---

## 📞 Support

- **Integration Questions**: See `INTEGRATION_SUMMARY.md`
- **Game Routes**: Check `src/routes/game_routes.py`
- **Model Loading**: Check `src/models/game/model_registry.py`
- **API Docs**: Visit `/docs` when server is running

---

**Last Updated**: 2026-01-02  
**Integration Version**: 2.0.0  
**Status**: ✅ Production Ready
