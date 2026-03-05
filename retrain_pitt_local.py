"""
Retrain the Pitt model locally to match the current sklearn/numpy versions.
Uses the same pipeline as Colab but runs on local Pitt data.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import spacy

print("Loading spaCy...")
nlp = spacy.load('en_core_web_sm')


def parse_cha_file(file_path):
    """Extract participant (*PAR:) utterances from CHAT format .cha file."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('*PAR:'):
                    content = line[5:].strip()
                    content = re.sub(r'\x15\d+_\d+\x15', '', content)
                    content = re.sub(r'\[.*?\]', '', content)
                    content = re.sub(r'<.*?>', '', content)
                    content = re.sub(r'&-\w*', '', content)
                    content = re.sub(r'[+/.]', '', content)
                    content = ' '.join(content.split())
                    if content:
                        texts.append(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return ' '.join(texts)


def extract_features(text, task_type='cookie'):
    """Extract linguistic features from participant text."""
    if not text or not text.strip():
        return None

    doc = nlp(text)
    words = text.split()
    sentences = list(doc.sents)

    word_count = len(words)
    sentence_count = max(1, len(sentences))

    # POS distribution
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    total_tokens = max(1, len(doc))

    # Lexical diversity
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / max(1, word_count)

    # Disfluency
    text_lower = text.lower()
    hesitation_count = sum(text_lower.count(h) for h in ['um', 'uh', 'er', 'ah', 'hmm'])
    false_starts = text_lower.count('i mean') + text_lower.count('that is')
    self_corrections = sum(text_lower.count(c) for c in ['no ', 'wait', 'actually', 'i mean', 'sorry'])
    uncertainty = sum(text_lower.count(u) for u in ['maybe', 'i think', 'probably', 'i guess'])

    # Cookie Theft task completion
    cookie_elements = ['cookie', 'jar', 'boy', 'girl', 'stool', 'kitchen',
                       'water', 'sink', 'overflow', 'dishes', 'falling', 'mother', 'woman']
    task_completion = sum(1 for e in cookie_elements if e in text_lower) / len(cookie_elements) if task_type == 'cookie' else 0

    # Connective density
    connectives = ['and', 'but', 'then', 'so', 'because', 'while', 'when', 'after', 'before']
    connective_count = sum(text_lower.count(f' {c} ') for c in connectives)

    # Repetition
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


# ===== PROCESS ALL .cha FILES =====
print("\nProcessing Pitt Corpus .cha files...")
pitt_dir = Path('data/Pitt')
all_data = []

for group in ['Control', 'Dementia']:
    group_dir = pitt_dir / group
    dementia_label = 1 if group == 'Dementia' else 0

    for task_dir in group_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name

        for cha_file in sorted(task_dir.glob('*.cha')):
            text = parse_cha_file(cha_file)
            if not text.strip():
                continue

            features = extract_features(text, task_name)
            if features is None:
                continue

            features['participant_id'] = cha_file.stem
            features['task_type'] = task_name
            features['group'] = group
            features['dementia_label'] = dementia_label
            all_data.append(features)

df = pd.DataFrame(all_data)
print(f"Total samples: {len(df)}")
print(f"Control: {sum(df['dementia_label']==0)}, Dementia: {sum(df['dementia_label']==1)}")

# ===== TRAIN MODEL =====
feature_columns = [
    'word_count', 'sentence_count', 'avg_words_per_sentence', 'avg_word_length',
    'ttr', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'pronoun_ratio',
    'hesitation_count', 'false_starts', 'self_corrections', 'uncertainty_markers',
    'task_completion', 'connective_density', 'repetition_score', 'semantic_density'
]

X = df[feature_columns].fillna(0).values
y = df['dementia_label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=min(10, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_idx = selector.get_support(indices=True)
selected_features = [feature_columns[i] for i in selected_idx]
print(f"\nSelected features: {selected_features}")

# 5-fold CV
print("\n5-Fold Cross-Validation...")
model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, min_samples_split=20, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model, X_train_selected, y_train, cv=cv,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True
)

print(f"  CV Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
print(f"  CV F1:       {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")
print(f"  CV AUC:      {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")

# Train final model
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)

print(f"\nFinal Test Results:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  F1:        {test_f1:.4f}")
print(f"  AUC:       {test_auc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")

# ===== SAVE =====
save_dir = Path('models/improved')
save_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, save_dir / 'best_model_gradient_boosting.joblib')
joblib.dump(scaler, save_dir / 'scaler.joblib')
joblib.dump(selector, save_dir / 'feature_selector.joblib')

metadata = {
    'model_name': 'Gradient Boosting',
    'dataset': 'Pitt Corpus (DementiaBank)',
    'synthetic_data_used': False,
    'total_samples': len(df),
    'control_samples': int(sum(y == 0)),
    'dementia_samples': int(sum(y == 1)),
    'train_samples': len(y_train),
    'test_samples': len(y_test),
    'feature_columns': feature_columns,
    'selected_features': selected_features,
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'cv_f1_mean': float(cv_results['test_f1'].mean()),
}

with open(save_dir / 'training_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to {save_dir}/")
print("Done!")
