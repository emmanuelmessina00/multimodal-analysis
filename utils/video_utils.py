import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import pandas as pd
import re
from models.cardiac.cardiac_extractor import load_cardiac_folder, cardiac_features


def get_video_info(video_paths: list[str]) -> pd.DataFrame:
    """fps, numero di frame, durata e offset di ogni segmento sulla timeline continua."""
    rows, offset = [], 0.0
    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"File video non trovato: {vp}")
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not fps or fps <= 0 or n_frames <= 0:
            raise ValueError(f"FPS/numero frame non leggibili per {vp} (fps={fps}, frames={n_frames})")
        duration = n_frames / fps
        rows.append({'path': vp, 'fps': fps, 'n_frames': n_frames,
                     'duration': duration, 'offset': offset})
        offset += duration
    return pd.DataFrame(rows)


def build_aligned_windows(
        video_paths: list[str],
        stop_video_index: int,
        stop_time_in_segment: float,
        cardiac_path: str,
        window_sec: int = 20,
        min_hr_coverage: float = 0.8,
) -> pd.DataFrame:
    """
    Allinea video e battito a ritroso dallo stop e restituisce le finestre validate.
    """
    videos = get_video_info(video_paths)
    if stop_time_in_segment > videos['duration'][stop_video_index]:
        raise ValueError("stop_time_in_segment oltre la durata del segmento indicato.")
    t_stop_vid = videos['offset'][stop_video_index] + stop_time_in_segment

    df_hr = load_cardiac_folder(cardiac_path)
    hr_stop_ts = df_hr['timestamp'].iloc[-1]
    video_zero_ts = hr_stop_ts - pd.Timedelta(seconds=t_stop_vid)

    n_windows = int(t_stop_vid // window_sec)
    first_start = t_stop_vid - n_windows * window_sec
    starts = [round(first_start + i * window_sec, 3) for i in range(n_windows)]

    windows = pd.DataFrame({
        'window_id': range(n_windows),
        'vid_start_sec': starts,
        'vid_end_sec': [s + window_sec for s in starts],
    })
    windows['video_idx'] = (videos['offset'].searchsorted(windows['vid_start_sec'], side='right') - 1)
    windows['ts_start'] = video_zero_ts + pd.to_timedelta(windows['vid_start_sec'], unit='s')
    windows['ts_end'] = video_zero_ts + pd.to_timedelta(windows['vid_end_sec'], unit='s')

    windows = windows.merge(cardiac_features(df_hr, windows, window_sec), on='window_id')

    dropped = windows[windows['hr_coverage'] < min_hr_coverage]
    if not dropped.empty:
        print(f"[WARN] Scartate {len(dropped)}/{n_windows} finestre con copertura cardiaca "
              f"< {min_hr_coverage:.0%}: window_id {dropped['window_id'].tolist()}")
    windows = windows[windows['hr_coverage'] >= min_hr_coverage]

    print(f"[Align] Stop video a {t_stop_vid:.2f}s = {hr_stop_ts} | t=0 video = {video_zero_ts} | "
          f"{len(windows)} finestre valide da {window_sec}s")

    cols = [
        'window_id', 'video_idx', 'vid_start_sec', 'vid_end_sec', 'ts_start', 'ts_end',
        'mu_hr', 'rmssd', 'pnn50', 'hr_range', 'sdnn', 'sd1', 'sd2', 'sd1_sd2',
        'hr_coverage'
    ]
    return windows[cols].sort_values(by='ts_start').reset_index(drop=True)

def natural_sort_key(path: str):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', os.path.basename(path))]

if __name__ == "__main__":
    import glob
    import re

    video_dir = os.path.join(PROJECT_ROOT, "data", "video")
    test_videos = sorted(
        [p for p in glob.glob(os.path.join(video_dir, "*")) if p.lower().endswith(".mp4")],
        key=natural_sort_key
    )
    if not test_videos:
        raise FileNotFoundError(f"Nessun file .mp4 trovato in: {video_dir}")

    print(f"Trovati {len(test_videos)} segmenti video: {[os.path.basename(v) for v in test_videos]}")

    test_cardiac_dir = os.path.join(PROJECT_ROOT, "data", "cardiac", "processed")

    df_windows = build_aligned_windows(
        video_paths=test_videos,
        stop_video_index=len(test_videos) - 1,  # di default punta all'ultimo segmento trovato
        stop_time_in_segment=725,
        cardiac_path=test_cardiac_dir,
        window_sec=20,
    )
    print(df_windows.head())

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out = os.path.join(processed_dir, "aligned_data.csv")
    df_windows.to_csv(out, index=False)
    print(f"Salvato {out}")