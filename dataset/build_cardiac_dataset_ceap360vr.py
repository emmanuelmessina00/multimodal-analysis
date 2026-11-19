import os
import sys
import json
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cardiac.cardiac_extractor import get_cardiac_metrics_by_bbi


def build_ceap360_dataset(
    ceap_dir: str = os.path.join(PROJECT_ROOT, "data", "public", "ceap360vr"),
    window_sec: float = 20.0,
    step_sec: float = 20.0,
    min_beats: int = 5,
) -> pd.DataFrame:
    """
    Estrae finestre da `window_sec` (default 20s) dai 32 partecipanti di CEAP-360VR,
    calcolando le stesse 8 feature HRV di `cardiac_extractor.py` + medie Valence/Arousal.
    """
    physio_files = sorted(glob.glob(os.path.join(ceap_dir, "5_PhysioData", "Transformed", "P*_Physio_TransData.json")))
    ann_dir = os.path.join(ceap_dir, "3_AnnotationData", "Transformed")

    if not physio_files:
        raise FileNotFoundError(f"Nessun file Physio_TransData trovato in: {ceap_dir}")

    rows = []

    for phy_path in physio_files:
        with open(phy_path, "r", encoding="utf-8") as f:
            phy_root = json.load(f)["Physio_TransData"][0]

        pid = phy_root["ParticipantID"]
        ann_path = os.path.join(ann_dir, f"{pid}_Annotation_TransData.json")
        if not os.path.exists(ann_path):
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann_root = json.load(f)["ContinuousAnnotation_TransData"][0]

        # Mappa VideoID -> lista annotazioni
        ann_by_vid = {
            v["VideoID"]: v["TimeStamp_Valence_Arousal"]
            for v in ann_root["Video_Annotation_TransData"]
        }

        for vid_phy in phy_root["Video_Physio_TransData"]:
            vid = vid_phy["VideoID"]
            ibi_records = vid_phy.get("IBI_TransData", [])
            ann_records = ann_by_vid.get(vid, [])
            if not ibi_records or not ann_records:
                continue

            ibi_ts = np.array([r["TimeStamp"] for r in ibi_records], dtype=np.float64)
            ibi_ms = np.array([r["IBI"] * 1000.0 for r in ibi_records], dtype=np.float64)

            ann_ts = np.array([r["TimeStamp"] for r in ann_records], dtype=np.float64)
            val_arr = np.array([r["Valence"] for r in ann_records], dtype=np.float64)
            aro_arr = np.array([r["Arousal"] for r in ann_records], dtype=np.float64)

            # Finestre [0, 20), [20, 40), [40, 60] (o con overlap se step_sec < window_sec)
            t_start = 0.0
            while t_start + window_sec <= 60.5:
                t_end = t_start + window_sec

                ibi_mask = (ibi_ts >= t_start) & (ibi_ts < t_end)
                bbi_win = ibi_ms[ibi_mask]
                # Filtra eventuali IBI fuori range fisiologico (stesso filtro di cardiac_extractor)
                bbi_win = bbi_win[(bbi_win >= 280.0) & (bbi_win <= 1500.0)]

                ann_mask = (ann_ts >= t_start) & (ann_ts < t_end)
                if len(bbi_win) >= min_beats and ann_mask.sum() > 0:
                    mu_hr, rmssd, pnn50, hr_range, sdnn, sd1, sd2, sd1_sd2 = get_cardiac_metrics_by_bbi(bbi_win)
                    v_mean = float(np.mean(val_arr[ann_mask]))
                    a_mean = float(np.mean(aro_arr[ann_mask]))

                    rows.append({
                        "participant_id": pid,
                        "video_id": vid,
                        "win_start_sec": t_start,
                        "win_end_sec": t_end,
                        "n_beats": int(len(bbi_win)),
                        "mu_hr": mu_hr,
                        "rmssd": rmssd,
                        "pnn50": pnn50,
                        "hr_range": hr_range,
                        "sdnn": sdnn,
                        "sd1": sd1,
                        "sd2": sd2,
                        "sd1_sd2": sd1_sd2,
                        "valence": v_mean,
                        "arousal": a_mean,
                        "high_valence": int(v_mean > 5.0),
                        "high_arousal": int(a_mean > 5.0),
                    })

                t_start += step_sec

    df = pd.DataFrame(rows)

    # Aggiunge gli scostamenti intra-soggetto (delta_*) come in build_multimodal_dataset.py
    for col in ["mu_hr", "rmssd", "pnn50", "sd1"]:
        subj_med = df.groupby("participant_id")[col].transform("median")
        df[f"delta_{col}"] = df[col] - subj_med

    return df


if __name__ == "__main__":
    df_cardiac = build_ceap360_dataset(window_sec=20.0, step_sec=20.0)
    out_path = os.path.join(PROJECT_ROOT, "data", "public", "ceap360_20s.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_cardiac.to_csv(out_path, index=False)

    print(f"[CEAP-360VR] Salvate {len(df_cardiac)} finestre da {df_cardiac['participant_id'].nunique()} partecipanti in {out_path}")
    print("\nDistribuzione classi:")
    print("  High Arousal (>5.0):", df_cardiac["high_arousal"].value_counts().to_dict())
    print("  High Valence (>5.0):", df_cardiac["high_valence"].value_counts().to_dict())