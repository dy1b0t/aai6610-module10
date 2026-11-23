"""
AAI6610 - Module 10: export the deployment artifacts for the Hugging Face Space.

Module 8 saved the tuned network's weights but not the StandardScaler it was
trained behind. This script rebuilds the exact Module 8 preprocessing and
80/20 stratified split (seed = 20), refits the scaler on the training rows
only, and writes its parameters to JSON next to a copy of the weights.
It also writes a handful of real test-set rows as regression fixtures so CI
can check that the deployed model still reproduces Module 8's predictions.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 20
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPACE = HERE / "credit-risk-space"
ART = SPACE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(ROOT / "Module 2" / "default of credit card clients.xls", header=1)
df = df.rename(columns={"default payment next month": "default"}).drop(columns=["ID"])
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

X = df.drop(columns=["default"])
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler = StandardScaler().fit(X_train)
with open(ART / "scaler.json", "w") as f:
    json.dump(
        {
            "features": list(X.columns),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "threshold": 0.5,
            "hidden": [64, 32, 16, 8],
            "dropout": 0.5,
        },
        f,
        indent=2,
    )

shutil.copy(ROOT / "Module 8" / "models" / "nn_credit_risk_tuned.pt", ART / "model.pt")

# Regression fixtures: score the full test set the same way Module 8 did,
# confirm the headline metric, and keep a few rows of each class.
import sys

sys.path.insert(0, str(SPACE))
from model import CreditRiskPredictor  # noqa: E402

pred = CreditRiskPredictor(ART)
proba = pred.predict_proba(X_test.to_numpy(dtype=np.float32))
labels = (proba >= 0.5).astype(int)
tp = int(((labels == 1) & (y_test.to_numpy() == 1)).sum())
fp = int(((labels == 1) & (y_test.to_numpy() == 0)).sum())
fn = int(((labels == 0) & (y_test.to_numpy() == 1)).sum())
f1 = 2 * tp / (2 * tp + fp + fn)
acc = float((labels == y_test.to_numpy()).mean())
print(f"Re-scored test set: accuracy={acc:.4f} F1={f1:.4f} (Module 8: 0.813 / 0.513)")

rng = np.random.default_rng(SEED)
idx = np.concatenate([
    rng.choice(np.where(labels == 1)[0], 5, replace=False),
    rng.choice(np.where(labels == 0)[0], 5, replace=False),
])
fixtures = [
    {
        "features": dict(zip(X.columns, map(int, X_test.iloc[i]))),
        "probability": round(float(proba[i]), 6),
        "label": int(labels[i]),
    }
    for i in idx
]
tests_dir = SPACE / "tests"
tests_dir.mkdir(exist_ok=True)
with open(tests_dir / "fixtures.json", "w") as f:
    json.dump({"test_accuracy": round(acc, 4), "test_f1": round(f1, 4), "rows": fixtures}, f, indent=2)
print("Wrote", ART / "scaler.json", ART / "model.pt", tests_dir / "fixtures.json")
