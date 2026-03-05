"""
Enhanced Model Loader for Reminder System

Loads Pitt Corpus-trained models (pure real data, no synthetic):
1. Gradient Boosting - Dementia risk prediction (models/improved/)
2. Confusion Detection - imblearn Pipeline (models/reminder_system/)
3. Caregiver Alert - imblearn Pipeline (models/reminder_system/)
4. Severity Classifier - imblearn Pipeline 3-class (models/reminder_system/)

Reminder system models are full imblearn Pipelines
(SMOTE -> StandardScaler -> SelectKBest -> Classifier).
SMOTE is skipped during predict(); scaler + feature selection run internally.

All models trained on 1,289 Pitt Corpus samples only.
"""

import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    """Loads Pitt Corpus-trained models for cognitive risk assessment."""
    
    # The 17 features used by the dementia risk model (models/improved/)
    ALL_FEATURE_COLUMNS = [
        'word_count', 'sentence_count', 'avg_words_per_sentence', 'avg_word_length',
        'ttr', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'pronoun_ratio',
        'hesitation_count', 'false_starts', 'self_corrections', 'uncertainty_markers',
        'task_completion', 'connective_density', 'repetition_score', 'semantic_density'
    ]
    
    # The 17 features used by reminder system models (models/reminder_system/)
    REMINDER_FEATURE_COLUMNS = [
        'word_count', 'sentence_count', 'avg_words_per_sentence', 'avg_word_length',
        'unique_word_ratio', 'hesitation_count', 'false_starts', 'self_corrections',
        'uncertainty_markers', 'semantic_incoherence', 'word_finding_difficulty',
        'circumlocution', 'tangentiality', 'narrative_coherence', 'response_coherence',
        'task_completion_score', 'language_deterioration_score'
    ]
    
    def __init__(self, models_dir: str = "models/improved"):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory containing the Pitt-trained dementia risk model
        """
        self.models_dir = Path(models_dir)
        self.reminder_models_dir = Path("models/reminder_system")
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.metadata = None
        self.spacy_nlp = None
        
        # Reminder system ML models (imblearn Pipelines)
        self.confusion_model = None
        self.alert_model = None
        self.severity_model = None
        self.reminder_scaler = None
        self.reminder_metadata = None
        self.optimal_thresholds = {}
        
        self._load_models()
        
    def _load_models(self):
        """Load all Pitt-trained models."""
        # Load dementia risk model (models/improved/)
        try:
            model_path = self.models_dir / 'best_model_gradient_boosting.joblib'
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("[SUCCESS] Loaded Pitt-trained Gradient Boosting model (dementia risk)")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            scaler_path = self.models_dir / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            selector_path = self.models_dir / 'feature_selector.joblib'
            if selector_path.exists():
                self.feature_selector = joblib.load(selector_path)
            
            metadata_path = self.models_dir / 'training_results.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
            logger.info(f"[INFO] Dementia risk model: AUC={self.metadata.get('test_auc', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error loading dementia risk model: {e}", exc_info=True)
            raise
        
        # Load reminder system models (models/reminder_system/)
        try:
            confusion_path = self.reminder_models_dir / 'confusion_detection_model.joblib'
            if confusion_path.exists():
                self.confusion_model = joblib.load(confusion_path)
                logger.info("[SUCCESS] Loaded confusion detection model")
            
            alert_path = self.reminder_models_dir / 'caregiver_alert_model.joblib'
            if alert_path.exists():
                self.alert_model = joblib.load(alert_path)
                logger.info("[SUCCESS] Loaded caregiver alert model")
            
            severity_path = self.reminder_models_dir / 'severity_classifier_model.joblib'
            if severity_path.exists():
                self.severity_model = joblib.load(severity_path)
                logger.info("[SUCCESS] Loaded severity classifier model")
            
            scaler_path = self.reminder_models_dir / 'feature_scaler.joblib'
            if scaler_path.exists():
                self.reminder_scaler = joblib.load(scaler_path)
            
            # Load optimal decision thresholds
            thresholds_path = self.reminder_models_dir / 'optimal_thresholds.json'
            if thresholds_path.exists():
                with open(thresholds_path, 'r') as f:
                    self.optimal_thresholds = json.load(f)
                logger.info(f"[INFO] Optimal thresholds loaded: {self.optimal_thresholds}")
            
            metadata_path = self.reminder_models_dir / 'training_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.reminder_metadata = json.load(f)
            
            loaded = []
            if self.confusion_model: loaded.append('confusion_detection')
            if self.alert_model: loaded.append('caregiver_alert')
            if self.severity_model: loaded.append('severity_classifier')
            logger.info(f"[INFO] Reminder system models loaded: {', '.join(loaded)}")
            
        except Exception as e:
            logger.warning(f"Reminder system models not available, using fallback: {e}")
    
    def _get_spacy(self):
        """Lazy-load spaCy model."""
        if self.spacy_nlp is None:
            import spacy
            self.spacy_nlp = spacy.load('en_core_web_sm')
        return self.spacy_nlp
    
    def extract_pitt_features(self, text: str) -> Dict[str, float]:
        """
        Extract the same linguistic features used during Pitt Corpus training.
        
        Args:
            text: User's response text
            
        Returns:
            Dictionary of 17 features matching the training pipeline
        """
        if not text or not text.strip():
            return {col: 0.0 for col in self.ALL_FEATURE_COLUMNS}
        
        nlp = self._get_spacy()
        doc = nlp(text)
        words = text.split()
        sentences = list(doc.sents)
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        
        # POS tag distribution
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        total_tokens = max(1, len(doc))
        
        # Disfluency markers
        text_lower = text.lower()
        hesitation_count = sum(text_lower.count(h) for h in ['um', 'uh', 'er', 'ah', 'hmm'])
        false_starts = text_lower.count('i mean') + text_lower.count('that is')
        self_corrections = sum(text_lower.count(c) for c in ['no ', 'wait', 'actually', 'i mean', 'sorry'])
        uncertainty = sum(text_lower.count(u) for u in ['maybe', 'i think', 'probably', 'i guess'])
        
        # Lexical diversity
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(1, word_count)
        
        # Connective density
        connectives = ['and', 'but', 'then', 'so', 'because', 'while', 'when', 'after', 'before']
        connective_count = sum(text_lower.count(f' {c} ') for c in connectives)
        
        # Repetition score
        word_list = [w.lower() for w in words]
        if len(word_list) >= 2:
            bigrams = list(zip(word_list[:-1], word_list[1:]))
            unique_bigrams = set(bigrams)
            repetition_score = 1 - (len(unique_bigrams) / max(1, len(bigrams)))
        else:
            repetition_score = 0
        
        # Semantic density
        content_words = [t for t in doc if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
        semantic_density = len(content_words) / max(1, total_tokens)
        
        # Task completion (for reminder responses, use a general measure)
        task_completion = min(1.0, word_count / 30)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': word_count / sentence_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'ttr': ttr,
            'noun_ratio': pos_counts.get('NOUN', 0) / total_tokens,
            'verb_ratio': pos_counts.get('VERB', 0) / total_tokens,
            'adj_ratio': pos_counts.get('ADJ', 0) / total_tokens,
            'pronoun_ratio': pos_counts.get('PRON', 0) / total_tokens,
            'hesitation_count': hesitation_count,
            'false_starts': false_starts,
            'self_corrections': self_corrections,
            'uncertainty_markers': uncertainty,
            'task_completion': task_completion,
            'connective_density': connective_count / sentence_count,
            'repetition_score': repetition_score,
            'semantic_density': semantic_density,
        }
    
    def predict_cognitive_risk(self, text: str) -> tuple:
        """
        Predict cognitive risk from text using Pitt-validated linguistic features.
        
        Uses the same dementia-discriminating features identified by the Pitt
        Corpus model (hesitation, lexical diversity, repetition, etc.) but
        applied as density-normalized indicators suitable for short reminder
        responses.
        
        Args:
            text: User's response text
            
        Returns:
            (dementia_probability, confidence) — both 0.0 to 1.0
        """
        if self.model is None:
            raise ValueError("Pitt model not loaded")
        
        # Extract features matching the training pipeline
        features = self.extract_pitt_features(text)
        word_count = features['word_count']
        text_lower = text.lower()
        
        # For responses with enough text (30+ words), use the Pitt model directly
        if word_count >= 30:
            features_df = pd.DataFrame([features])[self.ALL_FEATURE_COLUMNS]
            X = features_df.values
            if self.scaler is not None:
                X = self.scaler.transform(X)
            if self.feature_selector is not None:
                X = self.feature_selector.transform(X)
            probabilities = self.model.predict_proba(X)[0]
            dementia_prob = float(probabilities[1])
            confidence = float(max(probabilities))
            return dementia_prob, confidence
        
        # For shorter reminder responses, use density-normalized linguistic indicators
        # These are the same features the Pitt model identified as important,
        # but normalized per-word to work with short text.
        risk_score = 0.0
        
        # 1. Hesitation density (strong Pitt-validated signal, weight=0.20)
        if word_count > 0:
            hesitation_density = features['hesitation_count'] / word_count
            risk_score += min(1.0, hesitation_density * 5) * 0.20
        
        # 2. Self-correction density (moderate signal, weight=0.10)
        if word_count > 0:
            correction_density = features['self_corrections'] / word_count
            risk_score += min(1.0, correction_density * 5) * 0.10
        
        # 3. Uncertainty density (moderate signal, weight=0.10)
        if word_count > 0:
            uncertainty_density = features['uncertainty_markers'] / word_count
            risk_score += min(1.0, uncertainty_density * 5) * 0.10
        
        # 4. Repetition of words (strong dementia signal, weight=0.20)
        if word_count >= 3:
            words_list = text_lower.split()
            word_freq = {}
            for w in words_list:
                if len(w) > 2:  # skip short function words
                    word_freq[w] = word_freq.get(w, 0) + 1
            max_repeat = max(word_freq.values()) if word_freq else 1
            repetition_ratio = (max_repeat - 1) / max(1, word_count)
            risk_score += min(1.0, repetition_ratio * 10) * 0.20
        
        # 5. Memory-related keywords (contextual signal, weight=0.20)
        memory_phrases = ['don\'t remember', 'do not remember', 'not remember',
                          'forgot', 'can\'t recall', 'cannot recall', 'what was',
                          'i forget', 'don\'t know', 'do not know',
                          'can\'t remember', 'cannot remember', 'not sure what']
        memory_hits = sum(1 for phrase in memory_phrases if phrase in text_lower)
        risk_score += min(1.0, memory_hits * 0.5) * 0.20
        
        # 6. Low lexical diversity (for responses with enough words, weight=0.10)
        if word_count >= 5:
            low_diversity_penalty = max(0, 0.6 - features['ttr'])
            risk_score += min(1.0, low_diversity_penalty * 3) * 0.10
        
        # 7. False starts (weight=0.10)
        if word_count > 0:
            false_start_density = features['false_starts'] / word_count
            risk_score += min(1.0, false_start_density * 5) * 0.10
        
        dementia_prob = max(0.0, min(1.0, risk_score))
        confidence = max(0.3, min(1.0, 0.5 + abs(dementia_prob - 0.5)))
        
        return dementia_prob, confidence
    
    def extract_reminder_features(self, text: str) -> Dict[str, float]:
        """
        Extract the 17 features used by reminder system models
        (confusion, alert, severity).
        
        These differ from the dementia risk features — they include
        semantic incoherence, word-finding difficulty, etc.
        """
        if not text or not text.strip():
            return {col: 0.0 for col in self.REMINDER_FEATURE_COLUMNS}
        
        nlp = self._get_spacy()
        doc = nlp(text)
        words = text.split()
        sentences = list(doc.sents)
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        text_lower = text.lower()
        
        # Basic stats
        unique_words = set(w.lower() for w in words)
        unique_word_ratio = len(unique_words) / max(1, word_count)
        
        # Disfluency markers
        hesitation_count = sum(text_lower.count(h) for h in ['um', 'uh', 'er', 'ah', 'hmm'])
        false_starts = text_lower.count('i mean') + text_lower.count('that is')
        self_corrections = sum(text_lower.count(c) for c in ['no ', 'wait', 'actually', 'i mean', 'sorry'])
        uncertainty = sum(text_lower.count(u) for u in ['maybe', 'i think', 'probably', 'i guess'])
        
        # Semantic incoherence (topic shifts, incomplete thoughts)
        incomplete = sum(1 for s in sentences if len(list(s)) < 3)
        semantic_incoherence = incomplete / max(1, sentence_count)
        
        # Word-finding difficulty (long pauses, circumlocution markers)
        wfd_markers = ['thing', 'stuff', 'you know', 'whatchamacallit', 'thingy', 'whatnot']
        word_finding_difficulty = sum(text_lower.count(m) for m in wfd_markers) / max(1, word_count)
        
        # Circumlocution (talking around a word)
        circ_markers = ['the one that', 'it\'s like', 'kind of like', 'sort of', 'you know when']
        circumlocution = sum(text_lower.count(m) for m in circ_markers) / max(1, sentence_count)
        
        # Tangentiality (going off-topic)
        connectives = ['and', 'but', 'then', 'so', 'because']
        connective_count = sum(text_lower.count(f' {c} ') for c in connectives)
        tangentiality = max(0, (connective_count / max(1, sentence_count)) - 1) / 3
        
        # Narrative coherence
        content_words = [t for t in doc if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
        content_ratio = len(content_words) / max(1, len(doc))
        narrative_coherence = min(1.0, content_ratio + (1 - semantic_incoherence)) / 2
        
        # Response coherence
        avg_sent_len = word_count / sentence_count
        response_coherence = min(1.0, avg_sent_len / 15) * (1 - semantic_incoherence * 0.5)
        
        # Task completion score (for reminder responses)
        task_completion_score = min(1.0, word_count / 20)
        
        # Language deterioration
        deterioration_indicators = (
            hesitation_count / max(1, word_count) +
            self_corrections / max(1, word_count) +
            word_finding_difficulty +
            semantic_incoherence
        )
        language_deterioration_score = min(1.0, deterioration_indicators / 2)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': word_count / sentence_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_word_ratio': unique_word_ratio,
            'hesitation_count': hesitation_count,
            'false_starts': false_starts,
            'self_corrections': self_corrections,
            'uncertainty_markers': uncertainty,
            'semantic_incoherence': semantic_incoherence,
            'word_finding_difficulty': word_finding_difficulty,
            'circumlocution': circumlocution,
            'tangentiality': tangentiality,
            'narrative_coherence': narrative_coherence,
            'response_coherence': response_coherence,
            'task_completion_score': task_completion_score,
            'language_deterioration_score': language_deterioration_score,
        }
    
    def _get_reminder_features_raw(self, text: str) -> np.ndarray:
        """Extract reminder features as raw (unscaled) array.
        
        The imblearn Pipeline models include their own StandardScaler
        and SelectKBest steps, so we pass RAW features — the pipeline
        handles scaling and feature selection internally during predict().
        """
        features = self.extract_reminder_features(text)
        features_df = pd.DataFrame([features])[self.REMINDER_FEATURE_COLUMNS]
        return features_df.values
    
    def predict_confusion_detection(self, text: str) -> tuple:
        """
        Detect confusion using Pitt-trained ML pipeline.
        Uses optimal threshold from training if available.
        
        Returns:
            (is_confused: bool, confidence: float)
        """
        if self.confusion_model is not None:
            X = self._get_reminder_features_raw(text)
            probabilities = self.confusion_model.predict_proba(X)[0]
            positive_prob = float(probabilities[1])
            
            # Use optimal threshold if available, else default 0.5
            threshold = self.optimal_thresholds.get('confusion_detection', 0.5)
            is_confused = positive_prob >= threshold
            confidence = float(max(probabilities))
            return is_confused, confidence
        
        # Fallback to rule-based if ML model not available
        return self._rule_based_confusion(text)
    
    def predict_caregiver_alert(self, text: str) -> tuple:
        """
        Predict caregiver alert using Pitt-trained ML pipeline.
        Uses optimal threshold from training if available.
        
        Returns:
            (should_alert: bool, risk_score: float)
        """
        if self.alert_model is not None:
            X = self._get_reminder_features_raw(text)
            probabilities = self.alert_model.predict_proba(X)[0]
            positive_prob = float(probabilities[1])
            
            threshold = self.optimal_thresholds.get('caregiver_alert', 0.5)
            should_alert = positive_prob >= threshold
            risk_score = positive_prob
            return should_alert, risk_score
        
        # Fallback to rule-based
        dementia_prob, confidence = self.predict_cognitive_risk(text)
        confusion, confusion_score = self.predict_confusion_detection(text)
        combined_risk = (dementia_prob * 0.7) + (confusion_score * 0.3)
        should_alert = combined_risk > 0.6 or dementia_prob > 0.7
        return should_alert, combined_risk
    
    def predict_severity(self, text: str) -> tuple:
        """
        Predict response severity level using Pitt-trained ML pipeline.
        
        Returns:
            (severity_label: str, confidence: float)
            severity_label: 'normal', 'mild_concern', or 'high_risk'
        """
        severity_labels = ['normal', 'mild_concern', 'high_risk']
        
        if self.severity_model is not None:
            X = self._get_reminder_features_raw(text)
            prediction = self.severity_model.predict(X)[0]
            probabilities = self.severity_model.predict_proba(X)[0]
            label = severity_labels[int(prediction)]
            confidence = float(max(probabilities))
            return label, confidence
        
        # Fallback
        dementia_prob, _ = self.predict_cognitive_risk(text)
        if dementia_prob > 0.6:
            return 'high_risk', dementia_prob
        elif dementia_prob > 0.3:
            return 'mild_concern', dementia_prob
        return 'normal', 1 - dementia_prob
    
    def _rule_based_confusion(self, text: str) -> tuple:
        """Fallback rule-based confusion detection."""
        features = self.extract_pitt_features(text)
        text_lower = text.lower()
        confusion_score = 0.0
        
        if features['hesitation_count'] > 2:
            confusion_score += 0.3
        if features['ttr'] < 0.4 and features['word_count'] >= 5:
            confusion_score += 0.2
        if features['uncertainty_markers'] > 1:
            confusion_score += 0.25
        if features['self_corrections'] > 1:
            confusion_score += 0.15
        
        memory_phrases = ['don\'t remember', 'forgot', 'can\'t recall',
                          'don\'t know', 'can\'t remember', 'not sure what']
        if any(phrase in text_lower for phrase in memory_phrases):
            confusion_score += 0.3
        
        is_confused = confusion_score >= 0.4
        return is_confused, min(1.0, confusion_score)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        models_loaded = ['cognitive_risk']
        if self.confusion_model: models_loaded.append('confusion_detection')
        if self.alert_model: models_loaded.append('caregiver_alert')
        if self.severity_model: models_loaded.append('severity_classifier')
        
        info = {
            'models_loaded': models_loaded,
            'scalers_loaded': [],
            'encoders_loaded': [],
            'total_samples': self.metadata.get('total_samples', 'unknown') if self.metadata else 'unknown',
            'training_date': 'Pitt Corpus (pure real data)',
            'synthetic_data_used': False,
            'dementia_risk': {
                'test_f1': self.metadata.get('test_f1', 'unknown') if self.metadata else 'unknown',
                'test_auc': self.metadata.get('test_auc', 'unknown') if self.metadata else 'unknown',
            },
        }
        
        if self.scaler: info['scalers_loaded'].append('dementia_risk_scaler')
        if self.reminder_scaler: info['scalers_loaded'].append('reminder_scaler')
        
        if self.reminder_metadata and 'results' in self.reminder_metadata:
            results = self.reminder_metadata['results']
            for model_name in ['confusion_detection', 'caregiver_alert', 'severity_classifier']:
                if model_name in results:
                    info[model_name] = {
                        'algorithm': results[model_name].get('best_algorithm', 'unknown'),
                        'test_f1': results[model_name].get('test_f1', results[model_name].get('test_f1_weighted', 'unknown')),
                        'test_accuracy': results[model_name].get('test_accuracy', 'unknown'),
                    }
                    if 'test_auc' in results[model_name]:
                        info[model_name]['test_auc'] = results[model_name]['test_auc']
        
        return info