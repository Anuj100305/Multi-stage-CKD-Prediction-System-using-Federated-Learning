import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.preprocess import preprocess_data


# Load dataset
X_scaled, _, y, feature_names, scaler = preprocess_data("dataset/ckd-dataset-v2.csv")

# Split dataset into 3 parts (simulate clients)
X_split = np.array_split(X_scaled, 3)
y_split = np.array_split(y.values, 3)

client_id = int(input("Enter client id (0,1,2): "))

X_client = X_split[client_id]
y_client = y_split[client_id]

# Initialize model
model = LogisticRegression(max_iter=1000)

# IMPORTANT: initialize weights
model.fit(X_client, y_client)

class CKDClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [model.coef_, model.intercept_] if hasattr(model, "coef_") else []

    def set_parameters(self, parameters):
        if parameters:
             model.coef_ = parameters[0]
             model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)

        model.fit(X_client, y_client)
            
        return self.get_parameters(config), len(X_client), {}

    def evaluate(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)
        
        loss = 0.0
        accuracy = model.score(X_client, y_client)
        
        return loss, len(X_client), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=CKDClient(),
)