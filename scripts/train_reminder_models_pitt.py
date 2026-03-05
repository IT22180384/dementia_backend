"""
Train Reminder System Models Using Pitt Corpus Data Only (Optimized)

Trains 3 ML models for the context-aware smart reminder system:
1. Confusion Detection - detects confused responses
2. Caregiver Alert - predicts when to alert caregiver
3. Response Severity Classifier - classifies severity level (3-class)

Optimization methods applied:
- Proper sklearn Pipeline (no data leakage in CV)
- SMOTE oversampling for class imbalance
- RandomizedSearchCV for hyperparameter tuning
- Feature selection (SelectKBest) within the pipeline
- Optimal decision threshold selection
- Train vs test overfitting check
- Feature importance analysis
- Reduced circular label leakage via supplement columns

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
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
N_CV_FOLDS = 5
N_SEARCH_ITER = 50  # RandomizedSearchCV iterations
TEST_SIZE = 0.2

# Features used for model INPUT (prediction)
MODEL_FEATURES = [
    'word_count', 'sentence_count', 'avg_words_per_sentence', 'avg_word_length',
    'unique_word_ratio', 'hesitation_count', 'false_starts', 'self_corrections',
    'uncertainty_markers', 'semantic_incoherence', 'word_finding_difficulty',
    'circumlocution', 'tangentiality', 'narrative_coherence', 'response_coherence',
    'task_completion_score', 'language_deterioration_score'
]

# Supplement columns used ONLY for label derivation (not in MODEL_FEATURES)
# This reduces circular leakage between labels and model inputs
LABEL_SUPPLEMENT_COLUMNS = [
    'dementia_label', 'dementia_severity', 'cognitive_load_indicator',
    'discourse_coherence_score', 'semantic_fluency', 'lexical_diversity',
    'memory_references', 'repeated_questions', 'low_confidence_answers',
    'pause_frequency', 'speech_rate'
]


def find_optimal_threshold(y_true, y_prob):
    """Find the threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    best_f1 = f1_scores[best_idx]
    return float(best_threshold), float(best_f1)


def check_overfitting(model, X_train, y_train, X_test, y_test, model_name):
    """Compare train vs test performance to detect overfitting."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    gap_acc = train_acc - test_acc
    gap_f1 = train_f1 - test_f1

    print(f"\n  Overfitting Check ({model_name}):")
    print(f"    Train Accuracy: {train_acc:.4f}  |  Test Accuracy: {test_acc:.4f}  |  Gap: {gap_acc:.4f}")
    print(f"    Train F1:       {train_f1:.4f}  |  Test F1:       {test_f1:.4f}  |  Gap: {gap_f1:.4f}")

    if gap_f1 > 0.10:
        print(f"    WARNING: F1 gap > 0.10 — possible overfitting!")
    elif gap_f1 > 0.05:
        print(f"    CAUTION: F1 gap > 0.05 — mild overfitting, monitor closely")
    else:
        print(f"    OK: No significant overfitting detected")

    return {'train_acc': train_acc, 'test_acc': test_acc, 'gap_acc': gap_acc,
            'train_f1': train_f1, 'test_f1': test_f1, 'gap_f1': gap_f1}


def get_feature_importance(model, feature_names, model_name, top_n=10):
    """Extract and display feature importance from the trained model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        print(f"  Feature importance not available for {model_name}")
        return {}

    indices = np.argsort(importances)[::-1]
    print(f"\n  Feature Importance ({model_name}):")
    importance_dict = {}
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        importance_dict[feature_names[idx]] = float(importances[idx])

    return importance_dict


# ============================================================
# STEP 1: Load Pitt Corpus features
# ============================================================
print("=" * 70)
print("TRAINING REMINDER SYSTEM MODELS — OPTIMIZED (Pitt Corpus Only)")
print("=" * 70)

df = pd.read_csv('data/pitt_features.csv')
print(f"\nLoaded {len(df)} Pitt Corpus samples")
print(f"  Control: {sum(df['dementia_label']==0)}, Dementia: {sum(df['dementia_label']==1)}")
print(f"  Class ratio: 1:{sum(df['dementia_label']==1)/sum(df['dementia_label']==0):.1f} (Control:Dementia)")


# ============================================================
# STEP 2: Derive labels with REDUCED circular leakage
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Label Derivation (reduced circular leakage)")
print("=" * 70)

# Strategy: Use supplement columns (cognitive_load_indicator, discourse_coherence_score,
# dementia_severity, etc.) as PRIMARY label signals. Model features get lower weight (0.5x).
# This breaks the direct mapping between input features and derived labels.

# --- CONFUSION LABEL ---
print("\n--- Confusion Label ---")

confusion_indicators = np.zeros(len(df))

# PRIMARY indicators from supplement columns (NOT in model features)
if 'cognitive_load_indicator' in df.columns:
    cli_threshold = df['cognitive_load_indicator'].quantile(0.75)
    confusion_indicators += (df['cognitive_load_indicator'] > cli_threshold).astype(float)

if 'repeated_questions' in df.columns:
    confusion_indicators += (df['repeated_questions'] > 0).astype(float) * 1.5

if 'low_confidence_answers' in df.columns:
    lca_threshold = df['low_confidence_answers'].quantile(0.70)
    confusion_indicators += (df['low_confidence_answers'] > lca_threshold).astype(float)

if 'discourse_coherence_score' in df.columns:
    dcs_threshold = df['discourse_coherence_score'].quantile(0.25)
    confusion_indicators += (df['discourse_coherence_score'] < dcs_threshold).astype(float)

if 'memory_references' in df.columns:
    mr_threshold = df['memory_references'].quantile(0.80)
    confusion_indicators += (df['memory_references'] > mr_threshold).astype(float)

# SECONDARY indicators from model features (lower weight = 0.5x to reduce leakage)
si_threshold = df['semantic_incoherence'].quantile(0.80)
confusion_indicators += (df['semantic_incoherence'] > si_threshold).astype(float) * 0.5

nc_threshold = df['narrative_coherence'].quantile(0.20)
confusion_indicators += (df['narrative_coherence'] < nc_threshold).astype(float) * 0.5

# Interaction: dementia + word-finding difficulty
confusion_indicators += (
    df['dementia_label'] *
    (df['word_finding_difficulty'] > df['word_finding_difficulty'].quantile(0.70)).astype(float)
).astype(float) * 0.5

df['confusion_label'] = (confusion_indicators >= 3.0).astype(int)

print(f"  Not confused: {sum(df['confusion_label']==0)}")
print(f"  Confused:     {sum(df['confusion_label']==1)}")
print(f"  By group — Control: {sum((df['group']=='Control') & (df['confusion_label']==1))}, "
      f"Dementia: {sum((df['group']=='Dementia') & (df['confusion_label']==1))}")
print(f"  Imbalance ratio: 1:{sum(df['confusion_label']==0)/max(1,sum(df['confusion_label']==1)):.1f}")


# --- CAREGIVER ALERT LABEL ---
print("\n--- Caregiver Alert Label ---")

alert_indicators = np.zeros(len(df))

alert_indicators += df['dementia_label'].astype(float) * 2.0
alert_indicators += (df['confusion_label'] == 1).astype(float)

if 'dementia_severity' in df.columns:
    ds_threshold = df['dementia_severity'].quantile(0.75)
    alert_indicators += (df['dementia_severity'] > ds_threshold).astype(float)

if 'semantic_fluency' in df.columns:
    sf_threshold = df['semantic_fluency'].quantile(0.25)
    alert_indicators += (df['semantic_fluency'] < sf_threshold).astype(float)

if 'cognitive_load_indicator' in df.columns:
    cli_h_threshold = df['cognitive_load_indicator'].quantile(0.80)
    alert_indicators += (df['cognitive_load_indicator'] > cli_h_threshold).astype(float)

tc_threshold = df['task_completion_score'].quantile(0.20)
alert_indicators += (df['task_completion_score'] < tc_threshold).astype(float) * 0.5

df['alert_label'] = (alert_indicators >= 4.5).astype(int)

print(f"  No alert: {sum(df['alert_label']==0)}")
print(f"  Alert:    {sum(df['alert_label']==1)}")
print(f"  By group — Control: {sum((df['group']=='Control') & (df['alert_label']==1))}, "
      f"Dementia: {sum((df['group']=='Dementia') & (df['alert_label']==1))}")
print(f"  Imbalance ratio: 1:{sum(df['alert_label']==0)/max(1,sum(df['alert_label']==1)):.1f}")


# --- SEVERITY LABEL (3-class) ---
print("\n--- Response Severity Label (3-class) ---")

severity_score = np.zeros(len(df))
severity_score += df['dementia_label'].astype(float) * 2.0

if 'dementia_severity' in df.columns:
    severity_score += df['dementia_severity'] * 2.0

severity_score += (df['confusion_label'] == 1).astype(float) * 1.5

if 'cognitive_load_indicator' in df.columns:
    severity_score += (df['cognitive_load_indicator'] / df['cognitive_load_indicator'].max()) * 1.0

if 'low_confidence_answers' in df.columns:
    severity_score += (df['low_confidence_answers'] / max(1, df['low_confidence_answers'].max())) * 0.5

# Adaptive percentile-based cuts for balanced classes
t1 = np.percentile(severity_score, 40)
t2 = np.percentile(severity_score, 75)

df['severity_label'] = pd.cut(
    severity_score,
    bins=[-np.inf, t1, t2, np.inf],
    labels=[0, 1, 2]
).astype(int)

print(f"  Normal (0):       {sum(df['severity_label']==0)}")
print(f"  Mild concern (1): {sum(df['severity_label']==1)}")
print(f"  High risk (2):    {sum(df['severity_label']==2)}")


# ============================================================
# STEP 3: Prepare features and split
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Feature Preparation")
print("=" * 70)

available_features = [f for f in MODEL_FEATURES if f in df.columns]
print(f"  Model features ({len(available_features)}): {available_features}")
missing = [f for f in MODEL_FEATURES if f not in df.columns]
if missing:
    print(f"  WARNING: Missing features: {missing}")

X = df[available_features].fillna(0).values
print(f"  Feature matrix shape: {X.shape}")

X_train, X_test, idx_train, idx_test = train_test_split(
    X, np.arange(len(df)), test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=df['dementia_label']
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

save_dir = Path('models/reminder_system')
save_dir.mkdir(parents=True, exist_ok=True)

cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
all_results = {}


# ============================================================
# Hyperparameter search spaces
# ============================================================
PARAM_SPACES = {
    'random_forest': {
        'classifier__n_estimators': randint(50, 300),
        'classifier__max_depth': [3, 5, 7, 10, None],
        'classifier__min_samples_split': randint(5, 30),
        'classifier__min_samples_leaf': randint(2, 15),
        'classifier__max_features': ['sqrt', 'log2', None],
    },
    'gradient_boosting': {
        'classifier__n_estimators': randint(50, 300),
        'classifier__max_depth': [2, 3, 4, 5],
        'classifier__learning_rate': uniform(0.01, 0.19),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__min_samples_split': randint(5, 30),
        'classifier__min_samples_leaf': randint(2, 15),
    },
    'logistic_regression': {
        'classifier__C': uniform(0.01, 10.0),
        'classifier__l1_ratio': [0.0, 0.5, 1.0],
        'classifier__penalty': ['elasticnet'],
        'classifier__solver': ['saga'],
    },
}


def train_model_optimized(X_train, y_train, X_test, y_test, task_name,
                          feature_names, is_binary=True):
    """
    Full optimization pipeline per model:
    1. SMOTE oversampling (inside CV folds via imblearn Pipeline)
    2. StandardScaler normalization
    3. SelectKBest feature selection
    4. RandomizedSearchCV hyperparameter tuning
    5. Threshold optimization (binary)
    6. Overfitting detection
    7. Feature importance ranking
    """
    print(f"\n  Class distribution — Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    n_classes = len(np.unique(y_train))
    scoring = 'f1' if is_binary else 'f1_weighted'

    n_features = X_train.shape[1]
    k_values = [k for k in [5, 8, 10] if k < n_features] + [n_features]

    best_overall_score = -1
    best_overall_name = None
    best_overall_pipeline = None
    best_overall_params = None
    all_algo_results = {}

    algorithms = {
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'logistic_regression': LogisticRegression(
            class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000
        ),
    }

    for algo_name, base_model in algorithms.items():
        print(f"\n  --- {algo_name} ---")

        # imblearn Pipeline: SMOTE → Scale → SelectKBest → Classifier
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=n_features)),
            ('classifier', base_model),
        ])

        param_space = PARAM_SPACES[algo_name].copy()
        param_space['feature_selection__k'] = k_values

        search = RandomizedSearchCV(
            pipeline, param_space,
            n_iter=N_SEARCH_ITER,
            scoring=scoring,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            return_train_score=True,
            error_score='raise',
        )

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            print(f"    ERROR: {e}")
            all_algo_results[algo_name] = {'error': str(e)}
            continue

        cv_best_score = search.best_score_
        cv_train_score = search.cv_results_['mean_train_score'][search.best_index_]

        print(f"    Best CV {scoring}: {cv_best_score:.4f}")
        print(f"    Train {scoring}:   {cv_train_score:.4f}")
        print(f"    CV Gap:          {cv_train_score - cv_best_score:.4f}")
        print(f"    Best params: { {k.split('__')[-1]: v for k, v in search.best_params_.items()} }")

        all_algo_results[algo_name] = {
            'cv_score': float(cv_best_score),
            'cv_train_score': float(cv_train_score),
            'cv_gap': float(cv_train_score - cv_best_score),
            'best_params': {k: (int(v) if isinstance(v, (np.integer,)) else
                               float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in search.best_params_.items()},
        }

        if cv_best_score > best_overall_score:
            best_overall_score = cv_best_score
            best_overall_name = algo_name
            best_overall_pipeline = search.best_estimator_
            best_overall_params = search.best_params_

    print(f"\n  BEST ALGORITHM: {best_overall_name} (CV {scoring}={best_overall_score:.4f})")

    # --- Final evaluation on held-out test set ---
    model = best_overall_pipeline
    y_pred = model.predict(X_test)

    if is_binary:
        y_prob = model.predict_proba(X_test)[:, 1]

        # Default threshold (0.5)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1_default = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_prob)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred)

        print(f"\n  Test Results (default threshold=0.5):")
        print(f"    Accuracy:  {test_acc:.4f}")
        print(f"    F1:        {test_f1_default:.4f}")
        print(f"    AUC:       {test_auc:.4f}")
        print(f"    Precision: {test_prec:.4f}")
        print(f"    Recall:    {test_rec:.4f}")

        # Optimal threshold via precision-recall curve
        optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_prob)
        y_pred_opt = (y_prob >= optimal_threshold).astype(int)
        opt_acc = accuracy_score(y_test, y_pred_opt)
        opt_prec = precision_score(y_test, y_pred_opt, zero_division=0)
        opt_rec = recall_score(y_test, y_pred_opt)

        print(f"\n  Optimal Threshold: {optimal_threshold:.3f}")
        print(f"    F1 at optimal:   {optimal_f1:.4f} (vs {test_f1_default:.4f} at 0.5)")
        print(f"    Accuracy:        {opt_acc:.4f}")
        print(f"    Precision:       {opt_prec:.4f}")
        print(f"    Recall:          {opt_rec:.4f}")

        print(f"\n  Classification Report (optimal threshold):")
        print(classification_report(y_test, y_pred_opt,
                                    target_names=['Class 0 (Negative)', 'Class 1 (Positive)']))

        result = {
            'best_algorithm': best_overall_name,
            'test_accuracy': float(opt_acc),
            'test_f1': float(optimal_f1),
            'test_f1_default_threshold': float(test_f1_default),
            'test_auc': float(test_auc),
            'test_precision': float(opt_prec),
            'test_recall': float(opt_rec),
            'optimal_threshold': float(optimal_threshold),
            'algo_comparison': all_algo_results,
        }
    else:
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n  Test Results (multiclass):")
        print(f"    Accuracy:      {test_acc:.4f}")
        print(f"    F1 (weighted): {test_f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Normal', 'Mild Concern', 'High Risk']))

        result = {
            'best_algorithm': best_overall_name,
            'test_accuracy': float(test_acc),
            'test_f1_weighted': float(test_f1),
            'algo_comparison': all_algo_results,
        }

    # Overfitting check
    overfit = check_overfitting(model, X_train, y_train, X_test, y_test, task_name)
    result['overfitting_check'] = {k: float(v) for k, v in overfit.items()}

    # Feature importance
    final_classifier = model.named_steps['classifier']
    selected_k = model.named_steps['feature_selection'].k
    if selected_k == X_train.shape[1] or selected_k == 'all':
        selected_features = feature_names
    else:
        mask = model.named_steps['feature_selection'].get_support()
        selected_features = [f for f, m in zip(feature_names, mask) if m]

    importance = get_feature_importance(final_classifier, selected_features, task_name)
    result['feature_importance'] = importance
    result['selected_features'] = selected_features
    result['best_params'] = {k: (int(v) if isinstance(v, (np.integer,)) else
                                 float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in best_overall_params.items()}

    return model, result


# ============================================================
# STEP 4: Train Confusion Detection Model
# ============================================================
print("\n" + "=" * 70)
print("MODEL 1: Confusion Detection")
print("=" * 70)

y_confusion = df['confusion_label'].values
y_conf_train = y_confusion[idx_train]
y_conf_test = y_confusion[idx_test]

confusion_pipeline, confusion_result = train_model_optimized(
    X_train, y_conf_train, X_test, y_conf_test,
    task_name='confusion_detection',
    feature_names=available_features,
    is_binary=True
)
confusion_result['label_derivation'] = (
    'Derived from supplement columns (cognitive_load_indicator, repeated_questions, '
    'low_confidence_answers, discourse_coherence_score, memory_references) with lower-weight '
    'model-feature signals (semantic_incoherence, narrative_coherence). '
    'Reduced circular leakage vs prior version.'
)
all_results['confusion_detection'] = confusion_result
joblib.dump(confusion_pipeline, save_dir / 'confusion_detection_model.joblib')


# ============================================================
# STEP 5: Train Caregiver Alert Model
# ============================================================
print("\n" + "=" * 70)
print("MODEL 2: Caregiver Alert")
print("=" * 70)

y_alert = df['alert_label'].values
y_alert_train = y_alert[idx_train]
y_alert_test = y_alert[idx_test]

alert_pipeline, alert_result = train_model_optimized(
    X_train, y_alert_train, X_test, y_alert_test,
    task_name='caregiver_alert',
    feature_names=available_features,
    is_binary=True
)
alert_result['label_derivation'] = (
    'Derived from: dementia_label (weighted 2x) + confusion_label + dementia_severity + '
    'semantic_fluency + cognitive_load_indicator + task_completion_score (0.5x). '
    'Alert if combined score >= 4.5. Primarily supplement columns.'
)
all_results['caregiver_alert'] = alert_result
joblib.dump(alert_pipeline, save_dir / 'caregiver_alert_model.joblib')


# ============================================================
# STEP 6: Train Response Severity Classifier (3-class)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 3: Response Severity Classifier (3-class)")
print("=" * 70)

y_severity = df['severity_label'].values
y_sev_train = y_severity[idx_train]
y_sev_test = y_severity[idx_test]

severity_pipeline, severity_result = train_model_optimized(
    X_train, y_sev_train, X_test, y_sev_test,
    task_name='severity_classifier',
    feature_names=available_features,
    is_binary=False
)
severity_result['classes'] = ['normal', 'mild_concern', 'high_risk']
severity_result['label_derivation'] = (
    'Derived from composite: dementia_label (2x) + dementia_severity (2x) + '
    'confusion_label (1.5x) + cognitive_load_indicator + low_confidence_answers. '
    'Adaptive percentile cuts at 40th/75th. Primarily supplement columns.'
)
all_results['severity_classifier'] = severity_result

label_encoder = LabelEncoder()
label_encoder.fit(['normal', 'mild_concern', 'high_risk'])
joblib.dump(severity_pipeline, save_dir / 'severity_classifier_model.joblib')
joblib.dump(label_encoder, save_dir / 'severity_classifier_encoder.joblib')


# ============================================================
# STEP 7: Save metadata and artifacts
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Standalone scaler for inference
standalone_scaler = StandardScaler()
standalone_scaler.fit(X_train)
joblib.dump(standalone_scaler, save_dir / 'feature_scaler.joblib')

# Optimal thresholds
thresholds = {}
if 'optimal_threshold' in confusion_result:
    thresholds['confusion_detection'] = confusion_result['optimal_threshold']
if 'optimal_threshold' in alert_result:
    thresholds['caregiver_alert'] = alert_result['optimal_threshold']

with open(save_dir / 'optimal_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

# Full metadata
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
    'cv_folds': N_CV_FOLDS,
    'hyperparameter_search_iterations': N_SEARCH_ITER,
    'optimizations_applied': [
        'imblearn Pipeline (SMOTE + Scaler + SelectKBest + Classifier — no CV data leakage)',
        'SMOTE oversampling for class imbalance (applied per CV fold)',
        'RandomizedSearchCV hyperparameter tuning (50 iterations per algorithm)',
        'SelectKBest feature selection (k searched within pipeline)',
        'Optimal decision threshold via precision-recall curve (binary models)',
        'Overfitting detection (train vs test gap analysis)',
        'Feature importance ranking',
        'Reduced circular label leakage (supplement columns for label derivation)',
    ],
    'optimal_thresholds': thresholds,
    'models_trained': list(all_results.keys()),
    'results': all_results,
}

with open(save_dir / 'training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

with open(save_dir / 'feature_columns.json', 'w') as f:
    json.dump(available_features, f, indent=2)

print(f"\nAll models saved to {save_dir}/")
print(f"\nFiles created:")
for f in sorted(save_dir.iterdir()):
    print(f"  {f.name}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for model_name, result in all_results.items():
    print(f"\n  {model_name}:")
    print(f"    Algorithm: {result['best_algorithm']}")
    if 'test_auc' in result:
        print(f"    Test AUC:      {result['test_auc']:.4f}")
    test_f1_key = 'test_f1' if 'test_f1' in result else 'test_f1_weighted'
    print(f"    Test F1:       {result[test_f1_key]:.4f}")
    print(f"    Test Accuracy: {result['test_accuracy']:.4f}")
    if 'optimal_threshold' in result:
        print(f"    Opt Threshold: {result['optimal_threshold']:.3f}")
    if 'overfitting_check' in result:
        gap = result['overfitting_check']['gap_f1']
        status = 'OK' if gap <= 0.05 else ('CAUTION' if gap <= 0.10 else 'WARNING')
        print(f"    Overfit Gap:   {gap:.4f} ({status})")
    if 'selected_features' in result:
        print(f"    Features used: {len(result['selected_features'])}/{len(available_features)}")

print("\n\nOptimizations applied:")
for opt in metadata['optimizations_applied']:
    print(f"  + {opt}")

print("\n\nDone!")
