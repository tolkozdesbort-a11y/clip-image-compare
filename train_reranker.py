"""Train an optional binary relevance reranker from explicit /feedback labels.

Labels 0/1 are treated as not relevant; labels 2/3 as relevant.
The script refuses to train on tiny or single-class datasets.
"""
import os
import sqlite3
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = os.getenv("VISUAL_SEARCH_DB", "visual_search.sqlite3")
OUTPUT_PATH = os.getenv("RERANKER_PATH", "reranker.joblib")

with sqlite3.connect(DB_PATH) as db:
    rows = db.execute(
        "SELECT clip_similarity, dino_similarity, label FROM feedback"
    ).fetchall()

if len(rows) < 20:
    sys.exit(f"Need at least 20 feedback labels; found {len(rows)}.")

data = np.asarray(rows, dtype=np.float32)
X = data[:, :2]
y = (data[:, 2] >= 2).astype(np.int64)
counts = np.bincount(y, minlength=2)
if counts.min() < 5:
    sys.exit(
        "Need at least 5 examples in each class "
        f"(not relevant={counts[0]}, relevant={counts[1]})."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
print("Holdout balanced accuracy:", round(balanced_accuracy_score(y_test, predicted), 4))
print("Holdout ROC AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Holdout samples:", len(y_test))
print("Class counts (all labels):", {"not_relevant": int(counts[0]), "relevant": int(counts[1])})

# Save a final model fitted on all available human feedback.
model.fit(X, y)
joblib.dump(model, OUTPUT_PATH)
print("Saved reranker:", OUTPUT_PATH)
print("The reranker is optional and should only be kept if future held-out tests show a gain.")
