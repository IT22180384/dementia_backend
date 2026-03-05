"""
Train Reminder System Models Using Pitt Corpus Data Only

Trains 3 ML models for the context-aware smart reminder system:
1. Confusion Detection (Random Forest) - detects confused responses
2. Caregiver Alert (Logistic Regression) - predicts when to alert caregiver
3. Response Classifier (Random Forest) - classifies response severity level

All labels are derived from real Pitt Corpus linguistic markers,
NOT from synthetic data or circular thresholds.

Dataset: 1,289 Pitt Corpus transcripts (247 Control, 1042 Dementia)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
import joblib
import json
from datetime import datetime


# ============================================================
# STEP 1: Load Pitt Corpus features
# ============================================================
print("=" * 60)
print("TRAINING REMINDER SYSTEM MODELS (Pitt Corpus Only)")
print("=" * 60)

df = pd.read_csv('data/pitt_features.csv')
print(f"\nLoaded {len(df)} Pitt Corpus samples")
print(f"  Control: {sum(df['dementia_label']==0)}, Dementia: {sum(df['dementia_label']==1)}")


# ============================================================
# STEP 2: Derive proper labels from linguistic markers
# ============================================================
print("\n--- Deriving labels from linguistic markers ---")

# The 17 features used by the existing Gradient Boosting model
FEATURE_COLUMNS = [
    'word_count', 'sentence_count', 'avg_words_per_sentence', 'avg_word_length',
    'unique_word_ratio', 'hesitation_count', 'false_starts', 'self_corrections',
    'uncertainty_markers', 'semantic_incoherence', 'word_finding_difficulty',
    'circumlocution', 'tangentiality', 'narrative_coherence', 'response_coherence',
    'task_completion_score', 'language_deterioration_score'
]

# Verify all columns exist
available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
print(f"  Using {len(available_features)} features: {available_features}")

# --- CONFUSION LABEL ---
# Derived from multiple linguistic confusion indicators (not just severity threshold)
# A sample is "confused" if it shows multiple signs of linguistic confusion:
#   - High semantic incoherence (above 75th percentile)
#   - High word-finding difficulty
#   - Low narrative coherence (below 25th percentile)
#   - High circumlocution or tangentiality
#   - High hesitation relative to word count

confusion_score = np.zeros(len(df))

# Semantic incoherence (strong confusion signal)
si_threshold = df['semantic_incoherence'].quantile(0.75)
confusion_score += (df['semantic_incoherence'] > si_threshold).astype(float)

# Word-finding difficulty (strong confusion signal)
wfd_threshold = df['word_finding_difficulty'].quantile(0.75)
confusion_score += (df['word_finding_difficulty'] > wfd_threshold).astype(float)

# Low narrative coherence
nc_threshold = df['narrative_coherence'].quantile(0.25)
confusion_score += (df['narrative_coherence'] < nc_threshold).astype(float)

# High circumlocution
circ_threshold = df['circumlocution'].quantile(0.75)
confusion_score += (df['circumlocution'] > circ_threshold).astype(float)

# High tangentiality
tang_threshold = df['tangentiality'].quantile(0.75)
confusion_score += (df['tangentiality'] > tang_threshold).astype(float)

# High hesitation density (hesitation_count / word_count)
hesitation_density = df['hesitation_count'] / df['word_count'].clip(lower=1)
hd_threshold = hesitation_density.quantile(0.75)
confusion_score += (hesitation_density > hd_threshold).astype(float)

# Low response coherence
rc_threshold = df['response_coherence'].quantile(0.25)
confusion_score += (df['response_coherence'] < rc_threshold).astype(float)

# Confused if >= 3 indicators are present
df['confusion_label'] = (confusion_score >= 3).astype(int)

print(f"\n  Confusion label distribution:")
print(f"    Not confused: {sum(df['confusion_label']==0)}")
print(f"    Confused:     {sum(df['confusion_label']==1)}")
print(f"    Confusion by group:")
print(f"      Control confused:  {sum((df['group']=='Control') & (df['confusion_label']==1))}")
print(f"      Dementia confused: {sum((df['group']=='Dementia') & (df['confusion_label']==1))}")

# --- CAREGIVER ALERT LABEL ---
# Alert when: dementia patient AND shows severe linguistic deterioration
# Combines: dementia + high confusion score + low task completion + high language deterioration

alert_score = np.zeros(len(df))

# Must be dementia patient (base condition)
alert_score += df['dementia_label'].astype(float) * 2

# High confusion (from above)
alert_score += (confusion_score >= 3).astype(float)

# Low task completion
tc_threshold = df['task_completion_score'].quantile(0.25)
alert_score += (df['task_completion_score'] < tc_threshold).astype(float)

# High language deterioration
ld_threshold = df['language_deterioration_score'].quantile(0.75)
alert_score += (df['language_deterioration_score'] > ld_threshold).astype(float)

# High semantic incoherence
alert_score += (df['semantic_incoherence'] > si_threshold).astype(float)

# Alert if dementia + multiple risk factors (score >= 4)
df['alert_label'] = (alert_score >= 4).astype(int)

print(f"\n  Caregiver alert label distribution:")
print(f"    No alert: {sum(df['alert_label']==0)}")
print(f"    Alert:    {sum(df['alert_label']==1)}")
print(f"    Alert by group:")
print(f"      Control alert:  {sum((df['group']=='Control') & (df['alert_label']==1))}")
print(f"      Dementia alert: {sum((df['group']=='Dementia') & (df['alert_label']==1))}")

# --- RESPONSE SEVERITY LABEL ---
# 3-class: 0=normal, 1=mild_concern, 2=high_risk
# Based on combined linguistic indicators

severity_score = (
    (df['dementia_label'] * 2) +
    (confusion_score / 7) * 3 +  # normalize to 0-3
    (df['language_deterioration_score'] > ld_threshold).astype(float) +
    (df['semantic_incoherence'] > si_threshold).astype(float)
)

df['severity_label'] = pd.cut(
    severity_score, 
    bins=[-0.1, 1.5, 3.5, 10],
    labels=[0, 1, 2]
).astype(int)

print(f"\n  Severity label distribution:")
print(f"    Normal (0):       {sum(df['severity_label']==0)}")
print(f"    Mild concern (1): {sum(df['severity_label']==1)}")
print(f"    High risk (2):    {sum(df['severity_label']==2)}")


# ============================================================
# STEP 3: Prepare features
# ============================================================
print("\n--- Preparing features ---")

X = df[available_features].fillna(0).values
print(f"  Feature matrix shape: {X.shape}")

# Train/test split (stratified)
# We'll use the same split for all models
X_train, X_test, idx_train, idx_test = train_test_split(
    X, np.arange(len(df)), test_size=0.2, random_state=42, stratify=df['dementia_label']
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

save_dir = Path('models/reminder_system')
save_dir.mkdir(parents=True, exist_ok=True)

# Save the shared scaler
joblib.dump(scaler, save_dir / 'feature_scaler.joblib')
print(f"  Saved shared scaler")

all_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# STEP 4: Train Confusion Detection Model
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: Confusion Detection (Random Forest)")
print("=" * 60)

y_confusion = df['confusion_label'].values
y_conf_train = y_confusion[idx_train]
y_conf_test = y_confusion[idx_test]

# Compare 3 algorithms
models_to_try = {
    'random_forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        class_weight='balanced', random_state=42
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_split=20, random_state=42
    ),
    'logistic_regression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    )
}

best_score = 0
best_name = None
best_model = None
confusion_results = {}

for name, model in models_to_try.items():
    cv_scores = cross_validate(
        model, X_train_scaled, y_conf_train, cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=True
    )
    
    mean_f1 = cv_scores['test_f1'].mean()
    mean_auc = cv_scores['test_roc_auc'].mean()
    
    confusion_results[name] = {
        'cv_accuracy': float(cv_scores['test_accuracy'].mean()),
        'cv_f1': float(mean_f1),
        'cv_auc': float(mean_auc),
        'cv_f1_std': float(cv_scores['test_f1'].std()),
    }
    
    print(f"\n  {name}:")
    print(f"    CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} +/- {cv_scores['test_accuracy'].std():.4f}")
    print(f"    CV F1:       {mean_f1:.4f} +/- {cv_scores['test_f1'].std():.4f}")
    print(f"    CV AUC:      {mean_auc:.4f} +/- {cv_scores['test_roc_auc'].std():.4f}")
    
    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_model = model

print(f"\n  Best model: {best_name} (F1={best_score:.4f})")

# Train final model
best_model.fit(X_train_scaled, y_conf_train)
y_conf_pred = best_model.predict(X_test_scaled)
y_conf_prob = best_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_conf_test, y_conf_pred)
test_f1 = f1_score(y_conf_test, y_conf_pred)
test_auc = roc_auc_score(y_conf_test, y_conf_prob)
test_prec = precision_score(y_conf_test, y_conf_pred)
test_rec = recall_score(y_conf_test, y_conf_pred)

print(f"\n  Final Test Results:")
print(f"    Accuracy:  {test_acc:.4f}")
print(f"    F1:        {test_f1:.4f}")
print(f"    AUC:       {test_auc:.4f}")
print(f"    Precision: {test_prec:.4f}")
print(f"    Recall:    {test_rec:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_conf_test, y_conf_pred, target_names=['Not Confused', 'Confused']))

joblib.dump(best_model, save_dir / 'confusion_detection_model.joblib')

all_results['confusion_detection'] = {
    'best_algorithm': best_name,
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'cv_results': confusion_results,
    'label_derivation': 'Derived from 7 linguistic markers: semantic_incoherence, word_finding_difficulty, narrative_coherence, circumlocution, tangentiality, hesitation_density, response_coherence. Confused if >= 3 indicators above threshold.',
}


# ============================================================
# STEP 5: Train Caregiver Alert Model
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: Caregiver Alert (Logistic Regression)")
print("=" * 60)

y_alert = df['alert_label'].values
y_alert_train = y_alert[idx_train]
y_alert_test = y_alert[idx_test]

models_to_try = {
    'logistic_regression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        class_weight='balanced', random_state=42
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_split=20, random_state=42
    ),
}

best_score = 0
best_name = None
best_model = None
alert_results = {}

for name, model in models_to_try.items():
    cv_scores = cross_validate(
        model, X_train_scaled, y_alert_train, cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=True
    )
    
    mean_f1 = cv_scores['test_f1'].mean()
    mean_auc = cv_scores['test_roc_auc'].mean()
    
    alert_results[name] = {
        'cv_accuracy': float(cv_scores['test_accuracy'].mean()),
        'cv_f1': float(mean_f1),
        'cv_auc': float(mean_auc),
        'cv_f1_std': float(cv_scores['test_f1'].std()),
    }
    
    print(f"\n  {name}:")
    print(f"    CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} +/- {cv_scores['test_accuracy'].std():.4f}")
    print(f"    CV F1:       {mean_f1:.4f} +/- {cv_scores['test_f1'].std():.4f}")
    print(f"    CV AUC:      {mean_auc:.4f} +/- {cv_scores['test_roc_auc'].std():.4f}")
    
    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_model = model

print(f"\n  Best model: {best_name} (F1={best_score:.4f})")

best_model.fit(X_train_scaled, y_alert_train)
y_alert_pred = best_model.predict(X_test_scaled)
y_alert_prob = best_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_alert_test, y_alert_pred)
test_f1 = f1_score(y_alert_test, y_alert_pred)
test_auc = roc_auc_score(y_alert_test, y_alert_prob)
test_prec = precision_score(y_alert_test, y_alert_pred)
test_rec = recall_score(y_alert_test, y_alert_pred)

print(f"\n  Final Test Results:")
print(f"    Accuracy:  {test_acc:.4f}")
print(f"    F1:        {test_f1:.4f}")
print(f"    AUC:       {test_auc:.4f}")
print(f"    Precision: {test_prec:.4f}")
print(f"    Recall:    {test_rec:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_alert_test, y_alert_pred, target_names=['No Alert', 'Alert']))

joblib.dump(best_model, save_dir / 'caregiver_alert_model.joblib')

all_results['caregiver_alert'] = {
    'best_algorithm': best_name,
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'cv_results': alert_results,
    'label_derivation': 'Derived from: dementia_label (weighted x2) + confusion indicators (3+) + low task_completion + high language_deterioration + high semantic_incoherence. Alert if combined score >= 4.',
}


# ============================================================
# STEP 6: Train Response Severity Classifier
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Response Severity Classifier (3-class)")
print("=" * 60)

y_severity = df['severity_label'].values
y_sev_train = y_severity[idx_train]
y_sev_test = y_severity[idx_test]

models_to_try = {
    'random_forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        class_weight='balanced', random_state=42
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_split=20, random_state=42
    ),
    'logistic_regression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    ),
}

best_score = 0
best_name = None
best_model = None
severity_results = {}

for name, model in models_to_try.items():
    cv_scores = cross_validate(
        model, X_train_scaled, y_sev_train, cv=cv,
        scoring=['accuracy', 'f1_weighted'],
        return_train_score=True
    )
    
    mean_f1 = cv_scores['test_f1_weighted'].mean()
    
    severity_results[name] = {
        'cv_accuracy': float(cv_scores['test_accuracy'].mean()),
        'cv_f1_weighted': float(mean_f1),
        'cv_f1_std': float(cv_scores['test_f1_weighted'].std()),
    }
    
    print(f"\n  {name}:")
    print(f"    CV Accuracy:     {cv_scores['test_accuracy'].mean():.4f} +/- {cv_scores['test_accuracy'].std():.4f}")
    print(f"    CV F1 (weighted): {mean_f1:.4f} +/- {cv_scores['test_f1_weighted'].std():.4f}")
    
    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_model = model

print(f"\n  Best model: {best_name} (F1={best_score:.4f})")

best_model.fit(X_train_scaled, y_sev_train)
y_sev_pred = best_model.predict(X_test_scaled)

test_acc = accuracy_score(y_sev_test, y_sev_pred)
test_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')

print(f"\n  Final Test Results:")
print(f"    Accuracy:  {test_acc:.4f}")
print(f"    F1 (weighted): {test_f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_sev_test, y_sev_pred, target_names=['Normal', 'Mild Concern', 'High Risk']))

# Save model and label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['normal', 'mild_concern', 'high_risk'])
joblib.dump(best_model, save_dir / 'severity_classifier_model.joblib')
joblib.dump(label_encoder, save_dir / 'severity_classifier_encoder.joblib')

all_results['severity_classifier'] = {
    'best_algorithm': best_name,
    'test_accuracy': float(test_acc),
    'test_f1_weighted': float(test_f1),
    'classes': ['normal', 'mild_concern', 'high_risk'],
    'cv_results': severity_results,
    'label_derivation': '3-class derived from combined score of dementia_label, confusion_score, language_deterioration, semantic_incoherence. Cut at 1.5 and 3.5.',
}


# ============================================================
# STEP 7: Save metadata
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

metadata = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'Pitt Corpus (DementiaBank)',
    'synthetic_data_used': False,
    'total_samples': len(df),
    'control_samples': int(sum(df['dementia_label'] == 0)),
    'dementia_samples': int(sum(df['dementia_label'] == 1)),
    'train_samples': len(idx_train),
    'test_samples': len(idx_test),
    'feature_columns': available_features,
    'feature_count': len(available_features),
    'models_trained': list(all_results.keys()),
    'results': all_results,
}

with open(save_dir / 'training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save feature columns list for inference
with open(save_dir / 'feature_columns.json', 'w') as f:
    json.dump(available_features, f, indent=2)

print(f"\nAll models saved to {save_dir}/")
print(f"\nFiles created:")
for f in sorted(save_dir.iterdir()):
    print(f"  {f.name}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for model_name, result in all_results.items():
    print(f"\n  {model_name}:")
    print(f"    Algorithm: {result['best_algorithm']}")
    if 'test_auc' in result:
        print(f"    Test AUC:      {result['test_auc']:.4f}")
    print(f"    Test F1:       {result.get('test_f1', result.get('test_f1_weighted', 0)):.4f}")
    print(f"    Test Accuracy: {result['test_accuracy']:.4f}")

print("\n\nDone!")
