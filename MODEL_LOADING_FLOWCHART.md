# Model Loading Flow Diagram

## What Actually Happens When API Starts

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    API STARTUP: python run_api.py                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                    ↓
                    ┌─────────────────────────────────┐
                    │ app.py @on_event("startup")     │
                    │ loads: model_registry.py         │
                    │ calls: load_all_models()         │
                    └─────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────┐
        │                  load_all_models()                    │
        │         [from src/models/game/model_registry.py]      │
        └───────────────────────────────────────────────────────┘
                                    ↓
          ┌─────────────────────────────────────────────┐
          │         Load LSTM Model                     │
          └─────────────────────────────────────────────┘
                    ↓                       ↓
    Check LOCAL             (if not)     Download from
    ┌──────────────────┐              ┌─────────────────┐
    │ lstm_model.keras │              │ HuggingFace:    │
    │ or lstm_model.h5 │              │ vlakvindu/      │
    └──────────────────┘              │ Dementia_LSTM   │
         ✅ Found?                     └─────────────────┘
         └─ YES: Load locally              ↓
                (FAST ~10ms)        ✅ Download success?
                                    └─ YES: Cache & Load
                                    └─ NO: Use dummy
                            ↓
          ┌─────────────────────────────────────────────┐
          │  Load LSTM Scaler (lstm_scaler.pkl)         │
          └─────────────────────────────────────────────┘
                    ✅ Found locally
                    └─ YES: Load immediately
                                    ↓
          ┌─────────────────────────────────────────────┐
          │     Load Risk Classifier Model              │
          └─────────────────────────────────────────────┘
                    ↓
    Check LOCAL paths (in order):
    1. risk_logreg.pkl              ✅ FOUND
    2. logistic_regression_model.pkl ✅ FOUND
    
    ✅ YES - Load locally
       └─ FAST (~5ms)
                                    ↓
          ┌─────────────────────────────────────────────┐
          │  Load Risk Scaler & Label Encoder           │
          │  (feature_scaler.pkl, label_encoder.pkl)    │
          └─────────────────────────────────────────────┘
                    ✅ FOUND locally
                    └─ YES: Load immediately
                                    ↓
          ┌─────────────────────────────────────────────┐
          │       ✅ ALL MODELS LOADED                  │
          │         Store in Global Cache               │
          │    _MODELS = {                              │
          │      "lstm_model": <Model>,                 │
          │      "risk_classifier": <LogisticRegression>│
          │      "scaler": <StandardScaler>,            │
          │      "lstm_scaler": <StandardScaler>,       │
          │      "label_encoder": <LabelEncoder>        │
          │    }                                        │
          └─────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────┐
        │        🚀 API READY FOR REQUESTS             │
        │   All models cached in memory - FAST!         │
        └───────────────────────────────────────────────┘
```

---

## User Sends Game Session Request

```
User sends: POST /api/game/session
                    ↓
        ┌────────────────────────────────┐
        │  game_service.py               │
        │  process_game_session()         │
        └────────────────────────────────┘
                    ↓
        ┌────────────────────────────────┐
        │  Extract Features:              │
        │  - SAC, IES, Accuracy          │
        │  - Reaction Time, Variability   │
        └────────────────────────────────┘
                    ↓
        ┌────────────────────────────────┐
        │  Get LSTM Model (cached)        │ ✅ No reload!
        │  ↓                              │
        │  lstm_model.predict()           │
        │  └─ decline_score               │
        └────────────────────────────────┘
                    ↓
        ┌────────────────────────────────┐
        │  Get Risk Classifier (cached)   │ ✅ No reload!
        │  ↓                              │
        │  risk_classifier.predict_proba()│
        │  └─ HIGH/MEDIUM/LOW + scores    │
        └────────────────────────────────┘
                    ↓
        ┌────────────────────────────────┐
        │  Store in MongoDB               │
        │  risk_predictions collection    │
        └────────────────────────────────┘
                    ↓
        ┌────────────────────────────────┐
        │  Return to Client               │
        │  {                              │
        │    "label": "MEDIUM",           │
        │    "score": 45.2,               │
        │    "confidence": 0.78           │
        │  }                              │
        └────────────────────────────────┘
```

---

## File Organization - Visual Map

```
dementia_backend/
│
├── models/
│   └── models_registry.json          📚 DOCUMENTATION (9 models listed)
│                                        - For reference only
│                                        - NOT executed
│
├── src/
│   └── models/
│       ├── game/
│       │   ├── model_registry.py      ⚙️  ACTIVE CODE (Loader)
│       │   │                             - Runs at startup
│       │   │                             - Handles loading logic
│       │   │
│       │   ├── lstm_model/
│       │   │   ├── .cache/            📦 HuggingFace cache
│       │   │   └── lstm_scaler.pkl    ✅ USED (Feature normalization)
│       │   │       └─ Downloaded from HF automatically
│       │   │
│       │   └── risk_classifier/
│       │       ├── .cache/            📦 HuggingFace cache
│       │       ├── risk_logreg.pkl    ✅ USED (Primary model)
│       │       ├── feature_scaler.pkl ✅ USED (Feature normalization)
│       │       ├── label_encoder.pkl  ✅ USED (Label mapping)
│       │       └─ Downloaded from HF automatically
│       │
│       └── conversational_ai/         (Other team's component)
│
└── HuggingFace Cloud (Online):
    ├── vlakvindu/Dementia_LSTM_Model
    │   ├── lstm_model.keras           ✅ Downloaded on demand
    │   └── lstm_scaler.pkl
    │
    └── vlakvindu/Dementia_Risk_Clasif...
        ├── risk_logreg.pkl            ✅ Downloaded on demand
        ├── feature_scaler.pkl
        └── label_encoder.pkl
```

---

## Data Flow: Model → Prediction → Result

```
┌─────────────────────────────────────────────────────────────┐
│                 USER GAME SESSION                           │
│  User plays game, gets accuracy 75%, RT 280ms               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            FEATURE EXTRACTION (cognitive_scoring.py)        │
│  Calculated from game performance:                          │
│  - SAC = 0.268  (Speed-Accuracy Composite)                  │
│  - IES = 0.373  (Inverse Efficiency Score)                  │
│  - Accuracy = 0.75                                          │
│  - RT = 0.280 seconds                                       │
│  - Variability = 0.12                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LSTM MODEL (Loaded from HF)                    │
│  Processes: Last 10 game sessions (temporal trend)          │
│  Input shape: (1, 10, 5) = 1 batch, 10 timesteps, 5 features
│  ├─ Loaded from: HuggingFace cloud                          │
│  ├─ Cached in: src/models/game/lstm_model/.cache/           │
│  ├─ Scaler: lstm_scaler.pkl (local file)                   │
│  └─ Output: decline_score = 0.23 (mild decline)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         14-FEATURE VECTOR (for risk classifier)             │
│  1. mean_SAC = 0.245        9. current_SAC = 0.268          │
│  2. slope_SAC = 0.002       10. current_IES = 0.373         │
│  3. mean_IES = 0.350        11. slope_accuracy = 0.001      │
│  4. slope_IES = 0.005       12. slope_rt = -0.002           │
│  5. mean_accuracy = 0.74    13. std_SAC = 0.032             │
│  6. mean_RT = 0.275         14. std_IES = 0.045             │
│  7. mean_variability = 0.11                                 │
│  8. lstm_decline_score = 0.23                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│     RISK CLASSIFIER (LogisticRegression)                    │
│  ├─ Loaded from: Local file risk_logreg.pkl                 │
│  │  (Or downloaded from HF if missing)                      │
│  │                                                           │
│  ├─ Feature Scaler: feature_scaler.pkl (local)              │
│  │  └─ Normalizes 14 features to mean=0, std=1             │
│  │                                                           │
│  ├─ Label Encoder: label_encoder.pkl (local)                │
│  │  └─ Maps [0,1,2] → ['HIGH', 'LOW', 'MEDIUM']            │
│  │                                                           │
│  ├─ Model Prediction:                                       │
│  │  Raw probabilities: [0.12, 0.65, 0.23]                   │
│  │  └─ HIGH: 12%, LOW: 65%, MEDIUM: 23%                     │
│  │                                                           │
│  ├─ Safety Rules Check:                                     │
│  │  ✅ Accuracy 75% ≥ 60% → MEDIUM minimum                  │
│  │  ✅ Variability 0.12 ≤ 0.4 → No override                 │
│  │                                                           │
│  └─ Final Prediction: LOW (65% confidence)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              MongoDB Storage                                │
│  Collection: risk_predictions                               │
│  {                                                          │
│    "userId": "user123",                                     │
│    "window_size": 10,                                       │
│    "prediction": {                                          │
│      "label": "LOW",                                        │
│      "prob_high": 0.12,                                     │
│      "prob_low": 0.65,                                      │
│      "prob_medium": 0.23,                                   │
│      "risk_score_0_100": 32.8                               │
│    },                                                       │
│    "created_at": "2026-05-03T10:30:00"                      │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              API Response                                   │
│  HTTP 200 OK                                                │
│  {                                                          │
│    "risk_level": "LOW",                                     │
│    "risk_score": 32.8,                                      │
│    "confidence": 0.65,                                      │
│    "message": "Patient cognitive performance stable"        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Component | Location | Status | Used? | Source |
|-----------|----------|--------|-------|--------|
| **LSTM Model** | src/models/game/lstm_model/lstm_model.keras | ❌ MISSING | ✅ YES | HuggingFace ↓ |
| **LSTM Model** | HuggingFace cloud | ✅ AVAILABLE | ✅ YES | vlakvindu/Dementia_LSTM_Model |
| **LSTM Scaler** | src/models/game/lstm_model/lstm_scaler.pkl | ✅ EXISTS | ✅ YES | Local cache |
| **Risk Classifier** | src/models/game/risk_classifier/risk_logreg.pkl | ✅ EXISTS | ✅ YES | Local cache |
| **Risk Scaler** | src/models/game/risk_classifier/feature_scaler.pkl | ✅ EXISTS | ✅ YES | Local cache |
| **Label Encoder** | src/models/game/risk_classifier/label_encoder.pkl | ✅ EXISTS | ✅ YES | Local cache |
| **Model Registry (code)** | src/models/game/model_registry.py | ✅ EXISTS | ✅ YES | Active runtime |
| **Model Registry (docs)** | models/models_registry.json | ✅ EXISTS | ❌ NO | Reference only |

