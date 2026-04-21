from sklearn.metrics import classification_report, roc_auc_score
import joblib

model = joblib.load("final/ckd_model.pkl")

def evaluate(X_test, y_test):
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))