import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import pandas as pd
import numpy as np

from models.cardiac.cardiac_extractor import extract_cardiac_window, load_and_merge_cardiac_folder


def build_backward_aligned_windows(
        video_paths: list[str],
        stop_video_index: int,          # 0 se lo stop è nel primo video, 1 se è nel secondo
        stop_time_in_segment: float,    # secondi dall'inizio di quel segmento
        cardiac_csv_path: str,          # percorso alla cartella (es. "data/cardiac") coi 4 file o a un singolo CSV
        window_sec: int = 20
):
    durations = []
    fps_list = []

    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"File video non trovato: {vp}")

        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        durations.append(frames / fps if fps > 0 else 0.0)
        fps_list.append(fps)
        cap.release()

    t_stop_vid = sum(durations[:stop_video_index]) + stop_time_in_segment
    print(f"Istante di stop sul video continuo: {t_stop_vid:.2f} s")

    # Carica e unisce tutti i 4 file CSV della registrazione in un'unica timeline ordinata
    df_hr = load_and_merge_cardiac_folder(cardiac_csv_path)

    t_start_hr = df_hr['timestamp'].iloc[0]
    t_end_hr = df_hr['timestamp'].iloc[-1]
    total_hr_sec = (t_end_hr - t_start_hr).total_seconds()

    if total_hr_sec < t_stop_vid:
        raise ValueError(f"Il battito unificato ({total_hr_sec:.1f}s) è più corto del video fino allo stop ({t_stop_vid:.1f}s)!")

    # Istante esatto sulla timeline Garmin che corrisponde a t=0.0 del video continuo
    video_zero_ts = t_end_hr - pd.Timedelta(seconds=t_stop_vid)

    # Calcolo finestre da 20s a ritroso dallo stop
    n_windows = int(t_stop_vid // window_sec)
    valid_start_vid = t_stop_vid - (n_windows * window_sec)

    windows = []
    for w in range(n_windows):
        v_start = valid_start_vid + w * window_sec
        v_end = v_start + window_sec

        w_ts_start = video_zero_ts + pd.Timedelta(seconds=v_start)
        w_ts_end = video_zero_ts + pd.Timedelta(seconds=v_end)

        window_mask = (df_hr['timestamp'] >= w_ts_start) & (df_hr['timestamp'] < w_ts_end)
        window_df_slice = df_hr[window_mask]

        metrics = extract_cardiac_window(window_df_slice)
        mu_hr, rmssd, pnn50 = float(metrics[0]), float(metrics[1]), float(metrics[2])

        windows.append({
            "window_id": w,
            "window_start": v_start,
            "window_end": v_end,
            "vid_start_sec": v_start,
            "vid_end_sec": v_end,
            "mu_hr": mu_hr,
            "rmssd": rmssd,
            "pnn50": pnn50,
        })

    return pd.DataFrame(windows), durations, fps_list


if __name__ == "__main__":
    print("Test in corso di video_utils.py...")

    test_videos = ["data/video/camera_A000.mp4", "data/video/camera_A001.mp4"]
    test_cardiac_dir = "data/cardiac"

    if os.path.exists(test_videos[0]) and os.path.exists(test_cardiac_dir):
        df_windows, durations, fps_list = build_backward_aligned_windows(
            video_paths=test_videos,
            stop_video_index=1,
            stop_time_in_segment=120.0,
            cardiac_csv_path=test_cardiac_dir,
            window_sec=20
        )
        print("\nTest riuscito con successo! Ecco le prime righe:")
        print(df_windows.head())
        df_windows.to_csv("data/aligned_data.csv", index=False)
    else:
        print("[INFO] File di test non trovati ai path specificati.")