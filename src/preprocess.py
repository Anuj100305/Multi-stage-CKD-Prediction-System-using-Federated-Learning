import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path):

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df[df["stage"].isin(["s1","s2","s3","s4","s5"])]

    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    df.drop(columns=["class", "affected"], errors="ignore", inplace=True)

    stage_map = {"s1":0,"s2":1,"s3":2,"s4":3,"s5":4}
    df["stage"] = df["stage"].map(stage_map)

    y = df["stage"]
    X = df.drop("stage", axis=1)

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, scaler