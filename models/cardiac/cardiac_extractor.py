import os
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd


from utils.fit2csv import CST
def get_cardiac_metrics_by_bbi(bbi: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    bbi_size = bbi.shape[0]

    # mu of HR
    HR = 60000.0 / bbi
    mu_HR = np.mean(HR)

    # RMSSD
    diffs = np.diff(bbi)
    RMSSD = np.sqrt(np.sum(diffs ** 2) / (bbi_size - 1))

    # pNN50
    pNN50 = np.sum(np.abs(diffs) > 50) / (bbi_size - 1)

    # Nuova aggiunta dai paper di Shu e Nardelli
    hr_range= float(np.max(HR)) - float(np.min(HR))
    sdnn = float(np.std(bbi, ddof=1))

    x = bbi[:-1]  # RR_n
    y = bbi[1:]  # RR_{n+1}

    sd1 = float(np.std((y - x) / np.sqrt(2.0), ddof=1))
    sd2 = float(np.std((y + x) / np.sqrt(2.0), ddof=1))
    sd1_sd2 = float(sd1 / sd2) if sd2 > 0 else 0.0


    return float(mu_HR), float(RMSSD), float(pNN50 * 100), hr_range, float(sdnn), float(sd1),float(sd2),float(sd1_sd2)

def clean_cardiac_timeline(df_hr: pd.DataFrame, max_rel_dev: float = 0.20) -> pd.DataFrame:

    df = df_hr.copy()

    #(gestisce stringhe 'None' o spazi)
    for col in ['hr', 'hrv', 'exception']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #Scarta righe iniziali/finali completamente prive di entrambi i segnali
    df = df.dropna(subset=['hr', 'hrv'], how='all').reset_index(drop=True)

    # Calcolo incrociato dei None singoli (hr <-> hrv)
    valid_hr = df['hr'].where((df['hr'] >= 40) & (df['hr'] <= 214))
    valid_hrv = df['hrv'].where((df['hrv'] >= 280) & (df['hrv'] <= 1500))

    df['hr'] = valid_hr.fillna(60000.0 / valid_hrv)
    df['hrv'] = valid_hrv.fillna(60000.0 / valid_hr)

    #Rilevamento outlier su hrv (artefatti da movimento / battiti mancati)
    local_med = df['hrv'].rolling(window=5, center=True, min_periods=1).median()
    rel_deviation = (df['hrv'] - local_med).abs() / local_med

    is_artifact = rel_deviation > max_rel_dev
    if 'exception' in df.columns:
        implied_hr = 60000.0 / df['hrv']
        hw_artifact = (df['exception'] == 1) & ((df['hr'] - implied_hr).abs() > 12.0)
        is_artifact = is_artifact | hw_artifact

    n_artifacts = int(is_artifact.sum())
    if n_artifacts > 0:
        df.loc[is_artifact, 'hrv'] = np.nan
        df['hrv'] = df['hrv'].interpolate(method='linear').bfill().ffill()
        df.loc[is_artifact, 'hr'] = 60000.0 / df.loc[is_artifact, 'hrv']
        print(f"[CardiacBranch] Corretti {n_artifacts}/{len(df)} artefatti HRV ({n_artifacts/len(df):.1%}).")

    return df


def extract_cardiac_window(window_df: pd.DataFrame) -> np.ndarray:

    bbi_valid = window_df['hrv'].dropna().to_numpy(dtype=np.float64)

    if len(bbi_valid) >= 2:
        metrics = get_cardiac_metrics_by_bbi(bbi_valid)
        return np.array(metrics, dtype=np.float32)

    return np.zeros(8, dtype=np.float32)

def load_cardiac_folder(cardiac_path: str = "data/cardiac", clean_artifacts: bool = True) -> pd.DataFrame:
    """
    Unisce tutti i .csv cardiaci della registrazione, applica l'eventuale pulizia
    degli artefatti HRV e restituisce la timeline continua.
    """
    if os.path.isfile(cardiac_path):
        csv_files = [cardiac_path]
    elif os.path.isdir(cardiac_path):
        csv_files = [
            p for p in sorted(glob.glob(os.path.join(cardiac_path, "*.csv")))
            if not os.path.basename(p).startswith(("windows_", "merged_", "aligned_", "multimodal"))
        ]
    else:
        raise FileNotFoundError(f"Percorso cardiaco non trovato: {cardiac_path}")

    if not csv_files:
        raise FileNotFoundError(f"Nessun file .csv cardiaco trovato in: {cardiac_path}")

    parts = []
    for p in csv_files:
        df = pd.read_csv(
            p,
            sep=None,
            engine='python',
            encoding='latin-1',
            na_values=['None', 'none', 'NULL', 'null', '']
        )
        df = df.rename(columns={'x_hrv_0_1': 'hrv', 'heart_rate': 'hr'})
        if {'timestamp', 'hr'} <= set(df.columns) and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.tz_convert(CST)
            df = df.dropna(subset=['timestamp'])
            parts.append(df)

    if not parts:
        raise ValueError("Nessun dato cardiaco valido trovato nei file .csv.")

    df_hr = (
        pd.concat(parts, ignore_index=True)
        .sort_values(by='timestamp')
        .drop_duplicates(subset='timestamp', keep='first')
        .reset_index(drop=True)
    )

    if clean_artifacts:
        df_hr = clean_cardiac_timeline(df_hr)
    else:
        df_hr = df_hr.dropna(subset=['hr', 'hrv'], how='all').reset_index(drop=True)

    print(f"[CardiacBranch] Uniti {len(parts)} file -> {len(df_hr)} campioni validi "
          f"(da {df_hr['timestamp'].iloc[0]} a {df_hr['timestamp'].iloc[-1]})")

    gaps = df_hr['timestamp'].diff().dt.total_seconds()
    for i in gaps[gaps > 5].index:
        print(f"  [GAP] {gaps[i]:.0f}s senza dati: {df_hr['timestamp'][i - 1]} -> {df_hr['timestamp'][i]}")

    processed_dir = os.path.join("data", "cardiac", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "cleaned_cardiac_timeline.csv")
    df_hr.to_csv(output_path, index=False)
    return df_hr

def cardiac_features(df_hr: pd.DataFrame, windows: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """
    Calcola mu_hr, rmssd, pnn50 e hr_coverage per ogni finestra [ts_start, ts_end).
    """
    ts = df_hr['timestamp']
    rows = []
    for w in windows.itertuples():
        i0, i1 = ts.searchsorted(w.ts_start), ts.searchsorted(w.ts_end)
        window_df = df_hr.iloc[i0:i1]
        mu_hr, rmssd, pnn50,hr_range,sdnn,sd1,sd2,sd1_sd2 = extract_cardiac_window(window_df)
        rows.append({
            'window_id': w.window_id,
            'mu_hr': float(mu_hr),
            'rmssd': float(rmssd),
            'pnn50': float(pnn50),
            'hr_range': float(hr_range),
            'sdnn': float(sdnn),
            'sd1': float(sd1),
            'sd2': float(sd2),
            'sd1_sd2': float(sd1_sd2),
            'hr_coverage': min(1.0, len(window_df) / window_sec),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "cardiac")
    df_hr = load_cardiac_folder(path)
    print(df_hr.head())