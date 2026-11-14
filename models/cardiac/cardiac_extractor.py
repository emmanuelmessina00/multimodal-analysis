import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from utils import fit2csv

def get_cardiac_metrics_by_bbi(bbi):
    bbi_size=bbi.shape[0]

    #mu of HR
    HR = 60000.0 / bbi
    mu_HR = np.mean(HR)

    #RMSSD
    diffs = np.diff(bbi)
    RMSSD = np.sqrt(np.sum(diffs ** 2) / (bbi_size - 1))

    #pNN50
    pNN50=np.sum(np.abs(diffs) >50)/(bbi_size-1)

    return float(mu_HR), float(RMSSD), float(pNN50*100)

def extract_cardiac_window(df: pd.DataFrame, start_sec: int = 0, duration_sec: int = 20) -> np.ndarray:
    window= df.iloc[start_sec: start_sec + duration_sec]
    target_col = 'hrv' if 'hrv' in window.columns else 'x_hrv_0_1'
    bbi = pd.to_numeric(window[target_col], errors='coerce').dropna()
    bbi_valid = bbi[bbi > 0].to_numpy(dtype=np.float64)


    mu_hr, rmssd, pnn50 = get_cardiac_metrics_by_bbi(bbi_valid)
    return np.array([mu_hr, rmssd, pnn50], dtype=np.float32)

def get_all_cardiac_windows(
        df, duration_sec: int = 20, stride_sec: int = 20
) :

    df_size = len(df)
    rows = []


    for start_t in range(0, df_size - duration_sec + 1, stride_sec):
        mu_hr, rmssd, pnn50 = extract_cardiac_window(
            df, start_sec=start_t, duration_sec=duration_sec
        )

        rows.append(
            {
                "window_start": start_t,
                "window_end": start_t + duration_sec,
                "mu_hr": mu_hr,
                "rmssd": rmssd,
                "pnn50": pnn50,
            }
        )

    return pd.DataFrame(rows)
if __name__=="__main__":
    """
    example = np.array([820, 790, 810, 780, 830, 800])
    print(f"mu_HR, RMSSD, pNN50 by bbi {example} is: {get_cardiac_metrics_by_bbi(example)}")
    """
    fit2csv.main()
    df = pd.read_csv('data/cardiac/example.csv', encoding='latin-1', sep=None, engine='python')
    df_final=get_all_cardiac_windows(df)
    print(df_final.head())