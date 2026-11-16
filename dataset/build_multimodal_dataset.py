import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from utils.video_utils import build_aligned_windows
from models.visual.visual_extractor import VisualEmotionExtractor


def build_multimodal_dataset(
        video_paths: list[str],
        stop_video_index: int,
        stop_time_in_segment: float,
        cardiac_path: str,
        window_sec: int = 20,
        weights_path: str = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt"),
) -> pd.DataFrame:
    """
    1. Allinea video e battito e genera le finestre da `window_sec` (già filtrate:
       quelle senza dati cardiaci validi non arrivano qui).
    2. Estrae le probabilità di emozione solo su quelle finestre.
    3. Unisce i due rami sul `window_id`.
    """
    windows = build_aligned_windows(
        video_paths=video_paths,
        stop_video_index=stop_video_index,
        stop_time_in_segment=stop_time_in_segment,
        cardiac_path=cardiac_path,
        window_sec=window_sec,
    )
    if windows.empty:
        raise ValueError("Nessuna finestra valida dopo l'allineamento: niente da estrarre dal video.")

    extractor = VisualEmotionExtractor(weights_path=weights_path)
    df_visual = extractor.extract_windows(video_paths, windows)

    return (windows
            .merge(df_visual, on='window_id', how='inner', validate='one_to_one')
            .sort_values('ts_start')
            .reset_index(drop=True))


if __name__ == "__main__":
    import glob
    import re


    def natural_sort_key(path: str):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', os.path.basename(path))]


    video_dir = os.path.join(PROJECT_ROOT, "data", "video")
    test_videos = sorted(
        [p for p in glob.glob(os.path.join(video_dir, "*")) if p.lower().endswith(".mp4")],
        key=natural_sort_key
    )
    if not test_videos:
        raise FileNotFoundError(f"Nessun file .mp4 trovato in: {video_dir}")

    print(f"Trovati {len(test_videos)} segmenti video: {[os.path.basename(v) for v in test_videos]}")

    test_cardiac_dir = os.path.join(PROJECT_ROOT, "data", "cardiac", "processed")
    out = os.path.join("data","processed","aligned_data.csv")
    df_multimodal = build_multimodal_dataset(
        video_paths=test_videos,
        stop_video_index=len(test_videos) - 1,
        stop_time_in_segment=725,
        cardiac_path=test_cardiac_dir,
    )
    df_multimodal.to_csv(out, index=False)
    print(f"Salvate {len(df_multimodal)} finestre in {out}")