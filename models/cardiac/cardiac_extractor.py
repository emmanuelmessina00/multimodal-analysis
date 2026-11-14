import os
import re
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from utils.fit2csv import read_fit, TIMEZONE


# ----------------------------------------------------------------------------
# Metriche HRV
# ----------------------------------------------------------------------------

def get_cardiac_metrics_by_bbi(bbi):
    bbi_size = bbi.shape[0]

    # mu of HR
    HR = 60000.0 / bbi
    mu_HR = np.mean(HR)

    # RMSSD
    diffs = np.diff(bbi)
    RMSSD = np.sqrt(np.sum(diffs ** 2) / (bbi_size - 1))

    # pNN50
    pNN50 = np.sum(np.abs(diffs) > 50) / (bbi_size - 1)

    return float(mu_HR), float(RMSSD), float(pNN50 * 100)


def extract_cardiac_window(window_df: pd.DataFrame) -> np.ndarray:
    """Estrae le metriche cardiache su una fetta di DataFrame già filtrata per finestra."""
    target_col = 'hrv' if 'hrv' in window_df.columns else 'x_hrv_0_1'
    bbi_valid = np.array([], dtype=np.float64)

    if target_col in window_df.columns:
        bbi = pd.to_numeric(window_df[target_col], errors='coerce').dropna()
        bbi_valid = bbi[bbi > 0].to_numpy(dtype=np.float64)

    # Caso 1: BBI validi presenti
    if len(bbi_valid) >= 2:
        mu_hr, rmssd, pnn50 = get_cardiac_metrics_by_bbi(bbi_valid)
        return np.array([mu_hr, rmssd, pnn50], dtype=np.float32)

    # Caso 2: Fallback su HR a 1 Hz (filtrando subito hr > 0 per evitare divisioni per zero)
    if 'hr' in window_df.columns:
        hr_series = pd.to_numeric(window_df['hr'], errors='coerce').dropna()
        hr_valid = hr_series[hr_series > 0].to_numpy(dtype=np.float64)
        if len(hr_valid) >= 2:
            mu_hr = float(np.mean(hr_valid))
            bbi_approx = 60000.0 / hr_valid
            _, rmssd, pnn50 = get_cardiac_metrics_by_bbi(bbi_approx)
            return np.array([mu_hr, rmssd, pnn50], dtype=np.float32)
        elif len(hr_valid) == 1:
            return np.array([float(hr_valid[0]), 0.0, 0.0], dtype=np.float32)

    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ----------------------------------------------------------------------------
# Caricamento di tutti i file di una registrazione
# ----------------------------------------------------------------------------

def _start_from_filename(path: str) -> pd.Timestamp | None:
    """I file Garmin si chiamano 'YYYY-MM-DD-HH-MM-SS.fit' (ora locale di inizio registrazione)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", os.path.basename(path))
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y-%m-%d-%H-%M-%S").tz_localize(TIMEZONE)


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python', encoding='latin-1')
    df = df.rename(columns={'x_hrv_0_1': 'hrv', 'heart_rate': 'hr'})  # vecchio formato export
    if not {'timestamp', 'hr'} <= set(df.columns):
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.tz_convert(TIMEZONE)
    return df


def load_cardiac_folder(cardiac_path: str = "data/cardiac") -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.isfile(cardiac_path):
        paths = [cardiac_path]
    elif os.path.isdir(cardiac_path):
        fits = sorted(glob.glob(os.path.join(cardiac_path, "*.fit")))
        fit_stems = {os.path.splitext(p)[0] for p in fits}
        # Prende i .csv solo se non c'è alcun .fit corrispondente ed esclude file di output
        csvs = [
            p for p in sorted(glob.glob(os.path.join(cardiac_path, "*.csv")))
            if os.path.splitext(p)[0] not in fit_stems
            and not os.path.basename(p).startswith(("windows_", "merged_", "aligned_", "multimodal"))
        ]
        paths = fits + csvs
    else:
        raise FileNotFoundError(f"Percorso cardiaco non trovato: {cardiac_path}")

def cardiac_features(df_hr: pd.DataFrame, windows: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """
    Calcola mu_hr, rmssd, pnn50 per ogni finestra [ts_start, ts_end) della tabella `windows`.
    `hr_coverage` = frazione della finestra coperta da campioni (1 Hz): serve a scartare
    le finestre che cadono in buchi (es. file .fit illeggibile).
    """
    ts = df_hr['timestamp']  # ordinata: possiamo usare la ricerca binaria
    rows = []
    for w in windows.itertuples():
        i0, i1 = ts.searchsorted(w.ts_start), ts.searchsorted(w.ts_end)
        window_df = df_hr.iloc[i0:i1]
        mu_hr, rmssd, pnn50 = extract_cardiac_window(window_df)
        rows.append({'window_id': w.window_id,
                     'mu_hr': float(mu_hr), 'rmssd': float(rmssd), 'pnn50': float(pnn50),
                     'hr_coverage': min(1.0, len(window_df) / window_sec)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "cardiac")
    df_hr, files = load_cardiac_folder(path)
    print(f"\n{len(df_hr)} campioni validi, da {df_hr['timestamp'].iloc[0]} a {df_hr['timestamp'].iloc[-1]}")
    print(df_hr.head())