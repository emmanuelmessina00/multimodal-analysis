import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import glob

def evaluate(ckpt_path="checkpoints/model.joblib", csv_path="data/processed/aligned_data.csv"):
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]
    features = ckpt["features"]
    target = ckpt["target"]

    df = pd.read_csv(csv_path)
    valid = df[features].notna().all(axis=1)
    X = df.loc[valid, features].to_numpy()

    preds = model.predict(X)
    probs = model.predict_proba(X)

    df.loc[valid, f"pred_{target}"] = preds
    df.loc[valid, f"prob_{target}"] = probs[:, 1]

    if target in df.columns:
        acc = accuracy_score(df.loc[valid, target], preds)
        print(f"Accuracy su {csv_path}: {acc:.3f}")
    else:
        print(df[["window_id", f"pred_{target}", f"prob_{target}"]].head(10))

    return df


if __name__ == "__main__":
    for ckpt_path in sorted(glob.glob("checkpoints/*.joblib")):
        print(f"\n=== {os.path.basename(ckpt_path)} ===")
        evaluate(ckpt_path=ckpt_path, csv_path="data/processed/aligned_data.csv")