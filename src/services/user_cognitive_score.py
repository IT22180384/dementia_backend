"""
User Cognitive Score Service

Computes a final composite score (0-100%) for each user by combining:
1. ML Model Predictions (dementia risk, confusion, severity)
2. Reminder Adherence (confirmation rate, missed reminders)
3. Behavioral Trends (confusion trend, response times)

Score Components:
  - Cognitive Risk Score (40%): From Pitt-trained dementia risk model
  - Confusion & Severity (20%): From confusion detection + severity classifier
  - Adherence Score (25%): Based on confirmed vs missed/ignored reminders
  - Behavioral Trend (15%): Improving/declining patterns over time

Lower score = healthier. Higher score = more concern.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class UserCognitiveScoreService:
    """
    Computes a final composite dementia-risk percentage for a user
    by combining ML model outputs with reminder adherence data.
    """

    # Weight allocation for composite score
    WEIGHT_COGNITIVE_RISK = 0.40
    WEIGHT_CONFUSION_SEVERITY = 0.20
    WEIGHT_ADHERENCE = 0.25
    WEIGHT_TREND = 0.15

    def __init__(self, reminder_analyzer, behavior_tracker):
        """
        Args:
            reminder_analyzer: PittBasedReminderAnalyzer instance
            behavior_tracker: BehaviorTracker instance
        """
        self.analyzer = reminder_analyzer
        self.tracker = behavior_tracker

    def compute_user_score(
        self,
        user_id: str,
        recent_responses: Optional[List[str]] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute the full composite score for a user.

        Args:
            user_id: User identifier
            recent_responses: Optional list of recent text responses to run
                              through ML models. If None, only adherence data
                              is used for the model component.
            days: Number of days of history to analyze

        Returns:
            Dictionary with overall score and component breakdown.
        """
        # 1. ML model scores from recent responses
        model_scores = self._compute_model_scores(recent_responses or [])

        # 2. Adherence from behavior tracker
        adherence = self._compute_adherence_score(user_id, days)

        # 3. Trend from behavior tracker
        trend = self._compute_trend_score(user_id, days)

        # --- Weighted composite ---
        cognitive_risk = model_scores["avg_dementia_risk"]
        confusion_severity = model_scores["confusion_severity_score"]
        adherence_score = adherence["adherence_risk_score"]
        trend_score = trend["trend_risk_score"]

        composite = (
            cognitive_risk * self.WEIGHT_COGNITIVE_RISK
            + confusion_severity * self.WEIGHT_CONFUSION_SEVERITY
            + adherence_score * self.WEIGHT_ADHERENCE
            + trend_score * self.WEIGHT_TREND
        )

        # Clamp to 0-1
        composite = max(0.0, min(1.0, composite))

        # Risk level classification
        if composite >= 0.7:
            risk_level = "high_risk"
        elif composite >= 0.4:
            risk_level = "moderate_risk"
        elif composite >= 0.2:
            risk_level = "mild_concern"
        else:
            risk_level = "healthy"

        return {
            "user_id": user_id,
            "overall_score_percentage": round(composite * 100, 1),
            "risk_level": risk_level,
            "analysis_period_days": days,
            "computed_at": datetime.now().isoformat(),

            # Component breakdown
            "components": {
                "cognitive_risk": {
                    "score_percentage": round(cognitive_risk * 100, 1),
                    "weight": f"{self.WEIGHT_COGNITIVE_RISK:.0%}",
                    "weighted_contribution": round(cognitive_risk * self.WEIGHT_COGNITIVE_RISK * 100, 1),
                    "details": model_scores["cognitive_detail"],
                },
                "confusion_severity": {
                    "score_percentage": round(confusion_severity * 100, 1),
                    "weight": f"{self.WEIGHT_CONFUSION_SEVERITY:.0%}",
                    "weighted_contribution": round(confusion_severity * self.WEIGHT_CONFUSION_SEVERITY * 100, 1),
                    "details": model_scores["confusion_detail"],
                },
                "adherence": {
                    "score_percentage": round(adherence_score * 100, 1),
                    "weight": f"{self.WEIGHT_ADHERENCE:.0%}",
                    "weighted_contribution": round(adherence_score * self.WEIGHT_ADHERENCE * 100, 1),
                    "details": adherence["details"],
                },
                "behavioral_trend": {
                    "score_percentage": round(trend_score * 100, 1),
                    "weight": f"{self.WEIGHT_TREND:.0%}",
                    "weighted_contribution": round(trend_score * self.WEIGHT_TREND * 100, 1),
                    "details": trend["details"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Component 1: ML Model Scores
    # ------------------------------------------------------------------
    def _compute_model_scores(self, responses: List[str]) -> Dict[str, Any]:
        """
        Run recent responses through all 4 ML models and average the results.
        """
        if not responses:
            return {
                "avg_dementia_risk": 0.0,
                "confusion_severity_score": 0.0,
                "cognitive_detail": "No recent responses available for ML analysis",
                "confusion_detail": "No recent responses available",
            }

        model_loader = self.analyzer.enhanced_models

        dementia_risks = []
        confusion_flags = []
        severity_labels = []
        alert_flags = []

        for text in responses:
            try:
                # Dementia risk
                risk, confidence = model_loader.predict_cognitive_risk(text)
                dementia_risks.append(risk)

                # Confusion detection
                is_confused, conf_confidence = model_loader.predict_confusion_detection(text)
                confusion_flags.append(1.0 if is_confused else 0.0)

                # Severity classification
                severity_label, sev_conf = model_loader.predict_severity(text)
                severity_map = {"normal": 0.0, "mild_concern": 0.5, "high_risk": 1.0}
                severity_labels.append(severity_map.get(severity_label, 0.0))

                # Caregiver alert
                should_alert, alert_risk = model_loader.predict_caregiver_alert(text)
                alert_flags.append(1.0 if should_alert else 0.0)

            except Exception as e:
                logger.warning(f"Model prediction error for response: {e}")

        n = len(dementia_risks)
        if n == 0:
            return {
                "avg_dementia_risk": 0.0,
                "confusion_severity_score": 0.0,
                "cognitive_detail": "All model predictions failed",
                "confusion_detail": "All model predictions failed",
            }

        avg_risk = sum(dementia_risks) / n
        confusion_rate = sum(confusion_flags) / n
        avg_severity = sum(severity_labels) / n
        alert_rate = sum(alert_flags) / n

        # Confusion + severity combined (50/50)
        confusion_severity = (confusion_rate * 0.5) + (avg_severity * 0.5)

        return {
            "avg_dementia_risk": avg_risk,
            "confusion_severity_score": confusion_severity,
            "cognitive_detail": {
                "responses_analyzed": n,
                "avg_dementia_risk_pct": round(avg_risk * 100, 1),
                "caregiver_alert_rate_pct": round(alert_rate * 100, 1),
            },
            "confusion_detail": {
                "confusion_rate_pct": round(confusion_rate * 100, 1),
                "avg_severity": round(avg_severity, 3),
                "severity_breakdown": {
                    "normal": sum(1 for s in severity_labels if s == 0.0),
                    "mild_concern": sum(1 for s in severity_labels if s == 0.5),
                    "high_risk": sum(1 for s in severity_labels if s == 1.0),
                },
            },
        }

    # ------------------------------------------------------------------
    # Component 2: Adherence Score
    # ------------------------------------------------------------------
    def _compute_adherence_score(self, user_id: str, days: int) -> Dict[str, Any]:
        """
        Compute adherence risk from reminder interaction history.

        adherence_risk_score: 0.0 = perfect adherence, 1.0 = no adherence
        """
        pattern = self.tracker.get_user_behavior_pattern(
            user_id=user_id, days=days
        )

        total = pattern.total_reminders
        if total == 0:
            return {
                "adherence_risk_score": 0.0,
                "details": {
                    "message": "No reminder history available",
                    "total_reminders": 0,
                },
            }

        confirmed = pattern.confirmed_count
        missed = pattern.ignored_count
        delayed = pattern.delayed_count
        confused = pattern.confused_count

        # Confirmation rate (higher = better)
        confirmation_rate = confirmed / total

        # Missed rate (higher = worse)
        missed_rate = missed / total

        # Delayed and confused contribute partially
        delayed_rate = delayed / total
        confused_rate = confused / total

        # Adherence risk = inverse of good adherence + penalty for confusion
        # Perfect adherence: all confirmed → risk = 0
        # All missed: risk = 1
        adherence_risk = (
            (1.0 - confirmation_rate) * 0.5    # non-confirmation penalty
            + missed_rate * 0.25               # direct miss penalty
            + confused_rate * 0.15             # confusion penalty
            + delayed_rate * 0.10              # delay penalty
        )
        adherence_risk = max(0.0, min(1.0, adherence_risk))

        return {
            "adherence_risk_score": adherence_risk,
            "details": {
                "total_reminders": total,
                "confirmed": confirmed,
                "missed": missed,
                "delayed": delayed,
                "confused": confused,
                "confirmation_rate_pct": round(confirmation_rate * 100, 1),
                "missed_rate_pct": round(missed_rate * 100, 1),
                "avg_response_time_seconds": pattern.avg_response_time_seconds,
            },
        }

    # ------------------------------------------------------------------
    # Component 3: Behavioral Trend Score
    # ------------------------------------------------------------------
    def _compute_trend_score(self, user_id: str, days: int) -> Dict[str, Any]:
        """
        Compute risk from behavioral trends (improving/declining).
        """
        pattern = self.tracker.get_user_behavior_pattern(
            user_id=user_id, days=days
        )

        confusion_trend = pattern.confusion_trend
        avg_cognitive_risk = pattern.avg_cognitive_risk_score or 0.0
        escalation_needed = pattern.escalation_recommended

        # Map trend to risk multiplier
        trend_map = {
            "improving": 0.2,
            "stable": 0.4,
            "insufficient_data": 0.3,
            "declining": 0.8,
        }
        trend_base = trend_map.get(confusion_trend, 0.4)

        # Blend with average cognitive risk from interactions
        trend_risk = trend_base * 0.6 + avg_cognitive_risk * 0.4

        # Escalation flag adds extra risk
        if escalation_needed:
            trend_risk = min(1.0, trend_risk + 0.15)

        trend_risk = max(0.0, min(1.0, trend_risk))

        return {
            "trend_risk_score": trend_risk,
            "details": {
                "confusion_trend": confusion_trend,
                "avg_cognitive_risk_from_interactions": round(avg_cognitive_risk, 3),
                "escalation_recommended": escalation_needed,
                "optimal_reminder_hour": pattern.optimal_reminder_hour,
                "worst_hours": pattern.worst_response_hours,
            },
        }
