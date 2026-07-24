"""
Train a text classifier - with better generalization for real queries
"""

import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("=" * 60)
print("DEATHBED - TRAIN (with improved generalization)")
print("=" * 60)

# 1. Load data
df = pd.read_csv('data/raw/life_decisions_dataset.csv')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 10]

print(f"\n📊 Loaded {len(df)} samples")
print("\nClass distribution:")
print(df['label'].value_counts())

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 Training: {len(X_train)}, 🧪 Test: {len(X_test)}")

# ============================================================
# Choose classifier: try Logistic with stronger regularization,
# or switch to Naive Bayes.
# ============================================================

# --- Option A: Logistic (C=0.1 for stronger regularization) ---
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2),
#                               stop_words='english', sublinear_tf=True,
#                               min_df=2, max_df=0.8)),
#     ('clf', LogisticRegression(solver='lbfgs', max_iter=1000,
#                                C=0.1, class_weight='balanced', random_state=42))
# ])

# --- Option B: Naive Bayes (better calibrated, less overfitting) ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2),
                              stop_words='english', sublinear_tf=True,
                              min_df=2, max_df=0.8)),
    ('clf', MultinomialNB(alpha=1.0))   # alpha=1.0 is default, you can adjust
])

print("\n🔧 Building pipeline with Naive Bayes...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Cross‑validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"\n🔄 Cross‑Val (5‑fold): mean = {cv_scores.mean():.3f}, std = {cv_scores.std():.3f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/decision_classifier.pkl')
joblib.dump(pipeline.classes_, 'models/labels.pkl')
print("\n💾 Model saved to models/")

# Test on sample queries
print("\n" + "=" * 60)
print("🔮 SAMPLE PREDICTIONS (Naive Bayes)")
print("=" * 60)

test_queries = [
    "Should I quit my job to start a tech company?",
    "I'm thinking about proposing to my girlfriend",
    "Should I major in Computer Science or Business?",
    "Is it worth buying a house right now?",
    "Should I move to another city for a job opportunity?",
    "I'm considering becoming a stay-at-home parent",
    "Should I start exercising more to improve my health?",
    "I want to become a minimalist and simplify my life",
]

for q in test_queries:
    cleaned = clean_text(q)
    probs = pipeline.predict_proba([cleaned])[0]
    idx = probs.argmax()
    cat = pipeline.classes_[idx]
    conf = probs[idx]
    print(f"\n📝 '{q[:50]}...'")
    print(f"   → {cat.upper()} (confidence: {conf:.1%})")
    if conf < 0.60:
        print("   ⚠️ LOW CONFIDENCE - user override recommended")

print("\n" + "=" * 60)
print("🎉 Done!")