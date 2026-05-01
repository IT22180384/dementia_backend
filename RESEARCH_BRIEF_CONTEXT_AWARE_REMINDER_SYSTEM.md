# Context-Aware Reminder System: Research Brief

## 1. System Scope (Three Repos)
- dementia_backend: Core intelligence + APIs + real-time engine.
- RP_ElderyCareApp: Patient-facing cross-platform app (Android/iOS/Desktop/Web via Kotlin Multiplatform).
- RP_DementiaDash: Caregiver dashboard (React) for monitoring, reports, and intervention.

## 2. High-Level Architecture

User Layer
- Patient app receives reminders, captures acknowledgments/text/audio responses.
- Caregiver dashboard shows risk/adherence trends, alerts, and weekly summaries.

Communication Layer
- REST APIs for create/update/query/report actions.
- WebSocket channels for low-latency reminder alarms and caregiver alerts.

Intelligence Layer (backend)
- Reminder parsing (natural language + audio transcription).
- Cognitive response analysis (linguistic features + ML models).
- Adaptive scheduler (behavior-based timing/frequency updates).
- Alerting + escalation logic.

Data Layer
- MongoDB for reminders, interactions, caregiver alerts, weekly analytics artifacts.
- Model artifacts loaded from local files and optionally Hugging Face fallback.

## 3. End-to-End Process (How It Works)
1. Reminder creation
- Structured API, natural language text API, or audio API.
- Audio path: Whisper transcription -> NLP parser (BERT parser if available, regex fallback) -> reminder object.

2. Reminder delivery
- RealTimeReminderEngine checks due reminders and pushes events via WebSocket (/ws/user/{user_id}).
- Alarm cycle supports repeat attempts and escalation policy.

3. User response capture
- Patient confirms/snoozes/misses/responds (text or audio).
- Response is analyzed by PittBasedReminderAnalyzer + EnhancedModelLoader.

4. Cognitive/behavior analysis
- Cognitive risk + confusion/memory uncertainty indicators are computed.
- Interaction type is classified (confirmed/confused/ignored/delayed/etc.).

5. Adaptation and action
- BehaviorTracker logs interaction history and computes optimal hour, worst hours, trend, and frequency multiplier.
- Scheduler adjusts timing/frequency, triggers caregiver notifications if needed.

6. Reporting and dashboards
- Weekly report endpoint aggregates completion, risk trend, category-level stats, and alerts.
- Caregiver dashboard computes weighted health summaries from multiple cognitive streams.

## 4. Core ML/Analytics Metrics Used

Model-level metrics (documented)
- Dementia risk model (Gradient Boosting): Accuracy, Precision, Recall, F1, AUC-ROC, CV F1.
- Confusion detection model: Accuracy, Precision, Recall, F1, AUC-ROC, optimal threshold.
- Caregiver alert model: Accuracy, Precision, Recall, optimal threshold.
- Severity classifier: multi-class severity labels.

Runtime/operational metrics
- cognitive_risk_score (0-1)
- confusion_detected
- caregiver_alert_needed
- completion_rate
- confirmation_rate, confusion_rate
- avg_response_time_seconds
- risk_trend (improving/stable/declining)
- best_response_hours / worst_response_hours
- category_breakdown (total/completed/completion_rate/avg_risk)
- weekly aggregate metrics: avg/peak/lowest risk, unresolved alerts, risk change percentage.

Caregiver dashboard aggregation (RP_DementiaDash)
- Weighted final score model:
  - Chat risk-derived score: 30%
  - MMSE normalized score: 30%
  - Game risk-derived score: 20%
  - Reminder/adherence score: 20%

## 5. Technologies Used

Backend (dementia_backend)
- Python, FastAPI, Uvicorn
- scikit-learn, pandas, numpy, joblib
- imbalanced-learn pipelines (SMOTE + scaler + feature selection + classifier)
- spaCy, transformers, sentence-transformers, torch
- Whisper for speech-to-text
- Motor/PyMongo for MongoDB
- WebSockets for real-time reminder and caregiver channels

Patient app (RP_ElderyCareApp)
- Kotlin Multiplatform + Compose Multiplatform
- Ktor HTTP + Ktor WebSockets
- Kotlinx Serialization + Coroutines
- Multi-target builds (Android/iOS/JVM/WASM)

Caregiver dashboard (RP_DementiaDash)
- React + Vite
- TailwindCSS
- Recharts (visual analytics)
- jsPDF (report generation)

## 6. Context Awareness Mechanisms
- Temporal context: due-now windowing, repeat timeout handling, escalation thresholds.
- Behavioral context: 30-day interaction history (completion, confusion, delays, response latency).
- Linguistic context: hesitation, uncertainty, repetition, lexical diversity, semantic coherence.
- Clinical context: risk-level thresholds mapped to recommendation severity.
- Channel context: fallbacks between REST polling and WebSocket push for reliability.

## 7. Practical Research Contribution Angle
This system is not a static reminder app. It is a closed-loop cognitive-care pipeline:
- Senses user behavior and language,
- Estimates cognitive risk continuously,
- Personalizes reminder timing/frequency,
- Escalates to caregivers under high-risk patterns,
- Generates explainable weekly summaries for care decisions.

## 8. Important Implementation Note (for thesis accuracy)
There are minor documentation-vs-code differences in fusion weights and risk thresholds for final combined score.
- Documentation describes one weighting scheme and bands.
- Current runtime code in enhanced_model_loader.py uses a slightly different weighting and tighter score bands.

For research writing, state clearly whether results come from:
- Documented design specification, or
- Deployed code configuration.

## 9. Suggested Thesis Framing (Short)
"We implemented a multimodal, context-aware adaptive reminder architecture where reminder interactions are treated as cognitive probes. Linguistic risk modeling, behavior trend analysis, and real-time escalation are integrated in one loop to improve adherence and early detection of cognitive decline risk."
