import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
FEATURE_COLS = [
    "p_angry",
    "p_happy",
    "p_sad",
    "p_surprise",
    "p_neutral",
    "p_fear",
    "mu_hr",
    "rmssd",
    "pnn50",
]
def run_inference(clf, csv_path: str ) :
    """Carica il file multimodal.csv, seleziona le 9 feature ed esegue

    la predizione su ogni finestra temporale.
    """
    df = pd.read_csv(csv_path)

    # Verifica la presenza di tutte le colonne necessarie
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV per la fusione: {missing}")

    X = df[FEATURE_COLS].to_numpy()

    # Predizioni
    df["predicted_emotion"] = clf.predict(X)

    # Probabilità per classe
    probs = clf.predict_proba(X)
    for idx, class_name in enumerate(clf.classes_):
        df[f"proba_{class_name}"] = probs[:, idx]

    return df


if "__main__" == __name__:


    # [p_angry, p_happy, p_sad, p_surprise, p_neutral, p_fear, mu_HR, RMSSD, pNN50]
    X_train = np.array([
        # Esempi Calmo (0): neutral alto, battito basso, RMSSD alto
        [0.01, 0.02, 0.01, 0.01, 0.94, 0.01, 72.0, 42.0, 15.0],
        [0.00, 0.05, 0.00, 0.00, 0.93, 0.02, 69.5, 38.5, 12.0],
        # Esempi Stress (1): neutral alto (occhiali!), battito elevato, RMSSD basso
        [0.01, 0.01, 0.00, 0.02, 0.95, 0.01, 98.0, 15.2, 1.0],
        [0.00, 0.02, 0.01, 0.01, 0.96, 0.00, 102.5, 12.0, 0.0],
    ])
    y_train = np.array([0, 0, 1, 1])  # 0: Calmo, 1: Stress


    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)


    res=run_inference(clf, f"data/multimodal.csv")
    print(res)