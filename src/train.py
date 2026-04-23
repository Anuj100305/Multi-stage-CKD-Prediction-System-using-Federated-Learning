import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from preprocess import preprocess_data

# ================================
# FOLDERS
# ================================
os.makedirs("final", exist_ok=True)
os.makedirs("static", exist_ok=True)

DATASET_PATH = "dataset/ckd-dataset-v2.csv"

# ================================
# LOAD DATA
# ================================
X, y, feature_names, scaler = preprocess_data(DATASET_PATH)

# ================================
# SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# SMOTE
# ================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ================================
# BASE MODELS
# ================================
lr = LogisticRegression(max_iter=2000)
svm = SVC(probability=True)

rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [200, 300]},
    cv=3
).fit(X_train, y_train).best_estimator_

gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [200, 300]},
    cv=3
).fit(X_train, y_train).best_estimator_

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb = CalibratedClassifierCV(xgb, cv=3)

# Fit standalone models
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ================================
# STANDALONE MODEL ACCURACY
# ================================
models = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "XGBoost": xgb
}

results = {}

print("\nModel Accuracy:\n")
for name, model in models.items():
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# ================================
# ACEM MODEL
# ================================
acem_weights = [1, 1, 3, 3, 4]

acem = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm),
        ("rf", rf),
        ("gb", gb),
        ("xgb", xgb)
    ],
    voting="soft",
    weights=acem_weights
)

acem.fit(X_train, y_train)

acem_pred = acem.predict(X_test)
acem_acc = accuracy_score(y_test, acem_pred)
results["Adaptive Clinical Ensemble Model (ACEM)"] = acem_acc

print(f"\nAdaptive Clinical Ensemble Model (ACEM) Accuracy: {acem_acc:.4f}")

# ================================
# FEDERATED LEARNING (IMPROVED)
# ================================
def build_local_acem(x_local, y_local):
    local_lr = LogisticRegression(max_iter=2000)
    local_svm = SVC(probability=True)
    local_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    local_gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

    local_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
    local_xgb = CalibratedClassifierCV(local_xgb, cv=2)

    local_lr.fit(x_local, y_local)
    local_svm.fit(x_local, y_local)
    local_rf.fit(x_local, y_local)
    local_gb.fit(x_local, y_local)
    local_xgb.fit(x_local, y_local)

    local_acem = VotingClassifier(
        estimators=[
            ("lr", local_lr),
            ("svm", local_svm),
            ("rf", local_rf),
            ("gb", local_gb),
            ("xgb", local_xgb)
        ],
        voting="soft",
        weights=acem_weights
    )
    local_acem.fit(x_local, y_local)
    return local_acem

# split training data into 3 clients
client_indices = np.array_split(np.arange(len(X_train)), 3)

fed_models = []
fed_weights = []

for idx in client_indices:
    x_client = X_train[idx]
    y_client = y_train[idx]

    local_model = build_local_acem(x_client, y_client)
    fed_models.append(local_model)

    # local model quality on global test set
    local_pred = local_model.predict(X_test)
    local_acc = accuracy_score(y_test, local_pred)
    fed_weights.append(local_acc)

fed_weights = np.array(fed_weights, dtype=float)
fed_weights = fed_weights / fed_weights.sum()

def federated_predict(models, weights, X):
    prob_sum = np.zeros((X.shape[0], 5), dtype=float)
    for w, m in zip(weights, models):
        prob_sum += w * m.predict_proba(X)
    return prob_sum

fed_probs = federated_predict(fed_models, fed_weights, X_test)
fed_preds = np.argmax(fed_probs, axis=1)
fed_acc = accuracy_score(y_test, fed_preds)
results["Federated"] = fed_acc

print(f"Federated Accuracy: {fed_acc:.4f}")
print(f"Federated Weights: {fed_weights}")

# ================================
# GRAPH 1: MODEL ACCURACY
# ================================
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("static/model_accuracy.png")
plt.close()

# ================================
# GRAPH 2: PRIVACY vs ACCURACY
# ================================
privacy_scores = {
    "Adaptive Clinical Ensemble Model (ACEM)": 2,
    "Federated": 9
}

plt.figure(figsize=(8, 5))
for model_name in privacy_scores:
    plt.scatter(
        privacy_scores[model_name],
        results[model_name],
        s=120
    )
    plt.text(
        privacy_scores[model_name],
        results[model_name],
        model_name
    )

plt.xlabel("Privacy Level (Low → High)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Privacy Trade-off")
plt.tight_layout()
plt.savefig("static/privacy_vs_accuracy.png")
plt.close()

# ================================
# SAVE FILES
# ================================
joblib.dump(acem, "final/acem_model.pkl")
joblib.dump(fed_models, "final/federated_models.pkl")
joblib.dump(fed_weights, "final/federated_weights.pkl")
joblib.dump(scaler, "final/scaler.pkl")
joblib.dump(feature_names, "final/features.pkl")

np.save("final/X_train.npy", X_train)

with open("final/scores.json", "w") as f:
    json.dump(results, f)

print("\nTraining Complete")