import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

from preprocess import preprocess_data

os.makedirs("final", exist_ok=True)
os.makedirs("static", exist_ok=True)

DATASET_PATH = "dataset/ckd-dataset-v2.csv"

X, y, feature_names, scaler = preprocess_data(DATASET_PATH)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 🔥 MODELS + TUNING
# ================================

lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)

svm = SVC(probability=True).fit(X_train, y_train)

rf = GridSearchCV(
    RandomForestClassifier(),
    {"n_estimators": [100, 200]},
    cv=3
).fit(X_train, y_train).best_estimator_

gb = GridSearchCV(
    GradientBoostingClassifier(),
    {"n_estimators": [100, 200]},
    cv=3
).fit(X_train, y_train).best_estimator_

models = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

results = {}

print("\n🔍 Model Accuracy:\n")

for name, model in models.items():
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# ================================
# 🔥 ACEM MODEL
# ================================

weights = np.array(list(results.values()))
weights = weights / weights.sum()

acem = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft",
    weights=weights
)

acem.fit(X_train, y_train)

acem_pred = acem.predict(X_test)
acem_acc = accuracy_score(y_test, acem_pred)

print(f"\n🔥 ACEM Accuracy: {acem_acc:.4f}")

results["Adaptive Clinical Ensemble Model (ACEM)"] = acem_acc

# ================================
# 🌐 FEDERATED
# ================================

clients_X = np.array_split(X_train, 3)
clients_y = np.array_split(y_train, 3)

fed_models = []

for i in range(3):
    m = RandomForestClassifier()
    m.fit(clients_X[i], clients_y[i])
    fed_models.append(m)

def fed_predict(models, X):
    preds = [m.predict_proba(X) for m in models]
    return np.mean(preds, axis=0)

fed_preds = np.argmax(fed_predict(fed_models, X_test), axis=1)
fed_acc = accuracy_score(y_test, fed_preds)

print(f"🌐 Federated Accuracy: {fed_acc:.4f}")

results["Federated"] = fed_acc

# ================================
# 📊 GRAPH
# ================================

plt.figure(figsize=(10,5))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=25)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("static/model_accuracy.png")

# ================================
# 💾 SAVE
# ================================

joblib.dump(acem, "final/acem_model.pkl")
joblib.dump(fed_models, "final/federated_models.pkl")
joblib.dump(scaler, "final/scaler.pkl")
joblib.dump(feature_names, "final/features.pkl")

np.save("final/X_train.npy", X_train)

with open("final/scores.json", "w") as f:
    json.dump(results, f)

print("\n✅ Training Complete")