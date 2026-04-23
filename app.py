from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import datetime

app = Flask(__name__)

# Load saved files
model = joblib.load("final/acem_model.pkl")
fed_models = joblib.load("final/federated_models.pkl")
fed_weights = joblib.load("final/federated_weights.pkl")
scaler = joblib.load("final/scaler.pkl")
feature_names = joblib.load("final/features.pkl")
X_train = np.load("final/X_train.npy")

class_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]


def safe_float(val):
    try:
        if val is None or str(val).strip() == "":
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def federated_predict(models, weights, X):
    prob_sum = np.zeros((X.shape[0], 5), dtype=float)
    for w, m in zip(weights, models):
        prob_sum += w * m.predict_proba(X)
    return prob_sum


@app.route("/")
def home():
    return render_template("index.html", values={})


@app.route("/predict", methods=["POST"])
def predict():
    values = {}

    for f in feature_names:
        val = request.form.get(f)
        values[f] = val if val is not None and val != "" else "0"

    data = [safe_float(values[f]) for f in feature_names]
    scaled = scaler.transform([data])

    # ACEM prediction
    pred = model.predict(scaled)[0]
    prob = np.max(model.predict_proba(scaled)) * 100

    # Federated prediction
    fed_probs = federated_predict(fed_models, fed_weights, scaled)
    fed_pred = int(np.argmax(fed_probs))
    fed_prob = float(np.max(fed_probs)) * 100

    return render_template(
        "index.html",
        values=values,
        central_result=f"Adaptive Clinical Ensemble Model (ACEM - Hybrid): {class_names[pred]} ({prob:.2f}%)",
        fed_result=f"Federated: {class_names[fed_pred]} ({fed_prob:.2f}%)",
        accuracy_graph="static/model_accuracy.png"
    )


@app.route("/explain", methods=["POST"])
def explain():
    values = {}

    for f in feature_names:
        val = request.form.get(f)
        values[f] = val if val is not None and val != "" else "0"

    data = [safe_float(values[f]) for f in feature_names]
    scaled = scaler.transform([data])

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    exp = explainer.explain_instance(
        scaled[0],
        model.predict_proba,
        num_features=8
    )

    fig = exp.as_pyplot_figure()
    path = "static/lime.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return render_template(
        "index.html",
        values=values,
        lime_graph=path + "?t=" + str(datetime.datetime.now().timestamp())
    )


if __name__ == "__main__":
    app.run(debug=True)