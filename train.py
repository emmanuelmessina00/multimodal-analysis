import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

FEATURE_SETS = {
    "cardiac": [
        "mu_hr", "rmssd", "pnn50", "hr_range", "sdnn", "sd1", "sd2", "sd1_sd2",
        "delta_mu_hr", "delta_rmssd", "delta_pnn50", "delta_sd1",
    ],
    "visual": [
        "ear_mean", "ear_std", "mar_mean", "mar_max", "brow_raise_mean", "head_motion_std",
        "bs_smile_mean", "bs_smile_max", "bs_jaw_open_mean", "bs_brow_up_mean", "bs_blink_rate",
    ],
}
FEATURE_SETS["feature_fusion"] = FEATURE_SETS["visual"] + FEATURE_SETS["cardiac"]

MODELS = {
    "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
    "rf": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}


def train(csv_path, feature_key="cardiac", target="high_arousal", model_key="rf", out_path="checkpoints/model.joblib"):
    features = FEATURE_SETS[feature_key]
    df = pd.read_csv(csv_path).dropna(subset=features + [target])

    X = df[features].to_numpy()
    y = df[target].to_numpy()
    model = MODELS[model_key]

    if "participant_id" in df.columns:
        y_pred = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=df["participant_id"])
    else:
        y_pred = cross_val_predict(model, X, y, cv=5)

    acc = accuracy_score(y, y_pred)
    print(f"[{feature_key} | {model_key} -> {target}] Accuracy: {acc:.3f}")

    model.fit(X, y)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": model, "features": features, "target": target}, out_path)


if __name__ == "__main__":
    for target in ["high_arousal", "high_valence"]:
        for model_key in MODELS:
            out_path = f"checkpoints/cardiac_{model_key}_{target}.joblib"
            train(
                "data/public/ceap360_20s.csv",
                feature_key="cardiac",
                target=target,
                model_key=model_key,
                out_path=out_path,
            )