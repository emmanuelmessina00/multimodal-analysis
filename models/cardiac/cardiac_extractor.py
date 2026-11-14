import os
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from utils import fit2csv


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


def load_and_merge_cardiac_folder(cardiac_path: str = "data/cardiac") -> pd.DataFrame:
    """
    Accetta il percorso di una cartella (con i file .fit/.csv della registrazione)
    oppure di un singolo file CSV. Converte i .fit saltando quelli corrotti
    e unisce tutti i CSV validi in un unico DataFrame ordinato per timestamp.
    """
    if os.path.isfile(cardiac_path):
        csv_files = [cardiac_path]
    elif os.path.isdir(cardiac_path):
        fit_files = glob.glob(os.path.join(cardiac_path, "*.fit"))
        if fit_files:
            print("Avvio conversione .fit -> .csv con gestione file corrotti...")
            for fit_path in fit_files:
                csv_filename = os.path.join(cardiac_path, os.path.basename(fit_path)[:-4] + '.csv')
                try:
                    fitfile = fitparse.FitFile(fit_path, data_processor=fitparse.StandardUnitsDataProcessor())
                    fit2csv.write_fitfile_to_csv(fitfile, csv_filename)
                except Exception as e:
                    print(f"[SKIP] File FIT non valido o corrotto '{os.path.basename(fit_path)}': {e}")

        csv_files = sorted(glob.glob(os.path.join(cardiac_path, "*.csv")))
        csv_files = [
            f for f in csv_files
            if not os.path.basename(f).startswith("windows_")
               and not os.path.basename(f).startswith("merged_")
        ]
    else:
        raise FileNotFoundError(f"Percorso cardiaco non trovato: {cardiac_path}")

    if not csv_files:
        raise FileNotFoundError(f"Nessun file CSV cardiaco valido trovato in: {cardiac_path}")

    dfs = []
    for f in csv_files:
        try:
            df_part = pd.read_csv(f, encoding='latin-1', sep=None, engine='python')
            if 'hr' in df_part.columns and 'timestamp' in df_part.columns and len(df_part) > 0:
                dfs.append(df_part)
            else:
                print(f"[SKIP] {os.path.basename(f)} ignorato (vuoto o mancano colonne obbligatorie).")
        except Exception as e:
            print(f"[ERRORE] Lettura CSV fallita per {f}: {e}")

    if not dfs:
        raise ValueError("Nessun DataFrame cardiaco valido caricato dalla cartella.")

    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged = df_merged.dropna(subset=['hr', 'timestamp']).reset_index(drop=True)
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'], errors='coerce')
    df_merged = (
        df_merged.dropna(subset=['timestamp'])
        .sort_values('timestamp')
        .drop_duplicates(subset=['timestamp'], keep='first')
        .reset_index(drop=True)
    )

    print(f"[CardiacBranch] Uniti {len(dfs)} file CSV -> {len(df_merged)} campioni validi "
          f"(da {df_merged['timestamp'].iloc[0]} a {df_merged['timestamp'].iloc[-1]})")
    return df_merged


def get_all_cardiac_windows(df: pd.DataFrame, duration_sec: int = 20, stride_sec: int = 20) -> pd.DataFrame:
    """Taglia le finestre sul DataFrame unificato basandosi sui timestamp reali."""
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    t_start_session = df['timestamp'].iloc[0]
    total_seconds = int((df['timestamp'].iloc[-1] - t_start_session).total_seconds())

    for start_sec in range(0, total_seconds - duration_sec + 1, stride_sec):
        window_mask = (
            (df['timestamp'] >= t_start_session + pd.Timedelta(seconds=start_sec)) &
            (df['timestamp'] < t_start_session + pd.Timedelta(seconds=start_sec + duration_sec))
        )
        window_df = df[window_mask]
        mu_hr, rmssd, pnn50 = extract_cardiac_window(window_df)

        rows.append({
            "window_start": start_sec,
            "window_end": start_sec + duration_sec,
            "mu_hr": float(mu_hr),
            "rmssd": float(rmssd),
            "pnn50": float(pnn50),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_unified = load_and_merge_cardiac_folder("data/cardiac")
    df_windows = get_all_cardiac_windows(df_unified)
    print(f"\nCompletato! Estratte {len(df_windows)} finestre dalla registrazione unificata.")
    print(df_windows.head())