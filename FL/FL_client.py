import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocess import preprocess_data

# =========================
# LOAD DATA
# =========================
X_scaled, y, feature_names, scaler = preprocess_data("dataset/ckd-dataset-v2.csv")

y = np.array(y)
X_scaled = np.array(X_scaled)

# =========================
# STRATIFIED CLIENT SPLIT
# =========================
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(skf.split(X_scaled, y))

client_id = int(input("Enter client id (0,1,2): "))

train_idx, _ = splits[client_id]

X_client = X_scaled[train_idx]
y_client = y[train_idx]

# Multiclass classes
CLASSES = np.array([0, 1, 2, 3, 4])

# =========================
# INCREMENTAL MODEL
# =========================
model = SGDClassifier(
    loss="log_loss",
    max_iter=1,
    learning_rate="optimal",
    random_state=42,
    warm_start=True
)

# Initialize model once with partial_fit
sample_weights = compute_sample_weight(class_weight="balanced", y=y_client)
model.partial_fit(X_client, y_client, classes=CLASSES, sample_weight=sample_weights)

class CKDClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [model.coef_, model.intercept_]

    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        model.classes_ = CLASSES

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_client)

        # incremental update
        model.partial_fit(
            X_client,
            y_client,
            classes=CLASSES,
            sample_weight=sample_weights
        )

        return self.get_parameters(config), len(X_client), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        y_pred = model.predict(X_client)
        accuracy = accuracy_score(y_client, y_pred)

        # dummy loss for Flower
        loss = 1.0 - accuracy

        return loss, len(X_client), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=CKDClient(),
)